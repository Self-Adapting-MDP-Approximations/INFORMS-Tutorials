"""
Delayed-allocation solver built with Dantzig-Wolfe decomposition.

The file contains two main pieces:

1. A separation interface plus an exact pricing implementation for the
   period-level linking constraints.
2. A delayed allocation formulation over component state-action flows that is
   iteratively enriched with new feasible joint actions.

The decomposition works period by period. The delayed allocation problem keeps the
component flow variables ``x`` together with convex-combination variables
``pi`` over a current set of feasible joint actions. The pricing problem uses
the dual information from delayed allocation to search for a new joint action with
positive reduced cost.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from policy import ActionVector, Policy
from wmdp import StateComponent, StateSpace, WMDP

import gurobipy as gp
from gurobipy import GRB


FlowKey = Tuple[int, int, object, int]
DualKey = Tuple[int, int]


class VertexPolicy(Policy):
    """Period-wise policy induced by the dominant delayed-allocation vertex.

    For each period, the policy stores the joint actions with positive
    ``pi`` value. When queried for a state, it selects the largest-``pi``
    action among those whose component actions have positive marginal flow
    at the current component states. If no such action exists, it falls back
    to the largest-``pi`` action overall for that period.
    """

    action_values: Dict[int, List[Tuple[ActionVector, float]]]

    def __init__(
        self,
        positive_pi_actions: Mapping[int, Sequence[Tuple[ActionVector, float]]],
        marginal_flows: Mapping[FlowKey, float],
        tolerance: float = 1e-9,
    ) -> None:
        """Build a vertex policy from a delayed-allocation solution."""
        self.action_values = {}
        self.marginal_flows = dict(marginal_flows)
        self.tolerance = tolerance

        for period, action_values in positive_pi_actions.items():
            if not action_values:
                raise ValueError(
                    f"VertexPolicy requires at least one positive-pi action in period {period}."
                )

            self.action_values[period] = sorted(
                ((tuple(action), value) for action, value in action_values),
                key=lambda item: item[1],
                reverse=True,
            )

    def _has_positive_component_marginals(
        self,
        period: int,
        action: ActionVector,
        state: Sequence[StateComponent],
    ) -> bool:
        """Check whether each component action has positive flow at the given state."""
        for component, component_state in enumerate(state):
            component_action = action[component]
            if (
                self.marginal_flows.get(
                    (period, component, component_state.label, component_action),
                    0.0,
                )
                <= self.tolerance
            ):
                return False
        return True

    def get_action(self, period: int, state: Sequence[StateComponent]) -> ActionVector:
        """Return the selected joint action for the given period."""
        if period not in self.action_values:
            raise ValueError(f"Period {period} is not available in this vertex policy.")
        if not self.action_values[period]:
            raise RuntimeError(f"No actions are available for period {period}.")
        if len(state) != len(self.action_values[period][0][0]):
            raise ValueError("The number of component states must match the number of components.")

        chosen_action: Optional[ActionVector] = None
        for action, _ in self.action_values[period]:
            if self._has_positive_component_marginals(period, action, state):
                chosen_action = action
                break

        if chosen_action is None:
            chosen_action = self.action_values[period][0][0]

        return chosen_action


@dataclass(frozen=True)
class DelayedAllocationDuals:
    """Dual values extracted from one delayed allocation solution."""

    linking: Dict[int, Dict[DualKey, float]]
    simplex: Dict[int, float]


@dataclass
class DelayedAllocationResult:
    """Result after solving or refining delayed allocation."""

    objective_value: float
    marginal_flows: Dict[FlowKey, float]
    positive_pi_actions: Dict[int, List[Tuple[ActionVector, float]]]
    policy: VertexPolicy
    duals: DelayedAllocationDuals
    lower_bound: float
    added_actions: Dict[int, List[Tuple[ActionVector, float]]] = field(default_factory=dict)
    num_new_actions: int = 0


class Separation(ABC):
    """Interface for delayed-allocation separation methods."""

    @abstractmethod
    def solve_pricing_problem(
        self,
        wmdp: WMDP,
        duals_linking: Dict[Tuple[int, int], float],
        dual_simplex: float,
    ) -> Tuple[Tuple[int, ...], float]:
        """Return an improving action and its reduced cost for one period."""


class LinkingIPSeparation(Separation):
    """Exact pricing method based on the linking-constraint integer program."""

    def solve_pricing_problem(
        self,
        wmdp: WMDP,
        duals_linking: Dict[Tuple[int, int], float],
        dual_simplex: float,
    ) -> Tuple[Tuple[int, ...], float]:
        """Solve the pricing problem induced by the current delayed allocation duals.

        The pricing problem searches for one feasible joint action across all
        components that maximizes reduced cost. A positive objective value
        means the action can improve the delayed allocation problem and should
        be added as a new action.
        """
        linkct = wmdp.linking_constraints

        model = gp.Model(f"linking_ip_separation")
        model.setParam("OutputFlag", 0)

        z = model.addVars(
            [(j, a) for j in range(linkct.J) for a in linkct.A[j]],
            vtype=GRB.BINARY,
            name="z",
        )

        # Choose exactly one action for each component.
        for j in range(linkct.J):
            model.addConstr(
                gp.quicksum(z[j, a] for a in linkct.A[j]) == 1,
                name=f"choose_action_{j}",
            )

        # Enforce the global linking/resource constraints on the joint action.
        for k in range(linkct.K):
            model.addConstr(
                gp.quicksum(
                    linkct.C[(k, j, a)] * z[j, a]
                    for j in range(linkct.J)
                    for a in linkct.A[j]
                )
                <= linkct.b[k],
                name=f"linking_{k}",
            )

        # The pricing objective is the reduced cost of the candidate action.
        model.setObjective(
            gp.quicksum(
                duals_linking[(j, a)] * z[j, a]
                for j in range(linkct.J)
                for a in linkct.A[j]
            )
            - dual_simplex,
            GRB.MAXIMIZE,
        )

        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Linking-IP separation did not terminate optimally. "
                f"Gurobi status: {model.Status}"
            )

        action = []
        for j in range(linkct.J):
            chosen_action = None
            for a in linkct.A[j]:
                if z[j, a].X > 0.5:
                    chosen_action = a
                    break
            if chosen_action is None:
                raise RuntimeError(
                    f"Linking-IP separation returned no action for component {j}."
                )
            action.append(chosen_action)

        return tuple(action), model.ObjVal


class DelayedAllocationModel:
    """
    Delayed allocation model.
    """

    def __init__(
        self,
        wmdp: WMDP,
        initial_actions: Mapping[int, Sequence[ActionVector]],
        tolerance: float = 1e-9,
    ) -> None:
        """Build the initial delayed allocation model from the supplied action set."""
        self.wmdp = wmdp
        self.state_space: StateSpace = wmdp.state_space
        self.linkct = wmdp.linking_constraints
        self.J = wmdp.J
        self.T = wmdp.T
        self.Jset = range(self.J)
        self.Tset = range(self.T)
        self.tolerance = tolerance

        self.model = gp.Model("delayed_allocation")
        self.x = None
        self.action_set: Dict[int, List[ActionVector]] = {}
        self.pi_set: Dict[int, List[gp.Var]] = {}
        self.linking_x_constraints: Dict[Tuple[int, int, int], gp.Constr] = {}
        self.probability_simplex_constraints: Dict[int, gp.Constr] = {}

        self.model.setParam("OutputFlag", 0)
        self.model.setParam("FeasibilityTol", self.tolerance)

        validated_actions = self.validate_actions(initial_actions)
        self._build_base_DA_model()
        self.action_set = {t: [] for t in self.Tset}
        self.pi_set = {t: [] for t in self.Tset}
        self._build_linking_constraints()
        for t in self.Tset:
            for action in validated_actions[t]:
                self.add_action(t, action)


    def validate_actions(
        self,
        initial_actions: Mapping[int, Sequence[ActionVector]],
    ) -> Dict[int, List[ActionVector]]:
        """
        Validate the initial joint actions provided for each period.
        """
        
        validated_actions: Dict[int, List[ActionVector]] = {}
        expected_periods = set(self.Tset)
        provided_periods = set(initial_actions.keys())

        missing_periods = expected_periods - provided_periods
        if missing_periods:
            raise ValueError(
                "Initial actions must provide at least one joint action for every period. "
                f"Missing periods: {sorted(missing_periods)}."
            )

        extra_periods = provided_periods - expected_periods
        if extra_periods:
            raise ValueError(
                f"Initial actions contain invalid periods: {sorted(extra_periods)}."
            )

        for t in self.Tset:
            actions_t = initial_actions[t]
            if not actions_t:
                raise ValueError(
                    f"Initial actions for period {t} must contain at least one joint action."
                )

            validated_actions[t] = []
            for action in actions_t:
                action_tuple = tuple(action)
                if len(action_tuple) != self.J:
                    raise ValueError(
                        f"Action {action_tuple} in period {t} does not match the number "
                        f"of components ({self.J})."
                    )

                for j, component_action in enumerate(action_tuple):
                    if component_action not in self.state_space.A[j]:
                        raise ValueError(
                            f"Action {action_tuple} in period {t} uses invalid action "
                            f"{component_action} for component {j}."
                        )

                if not self.wmdp.is_action_feasible(action_tuple):
                    raise ValueError(
                        f"Action {action_tuple} in period {t} is infeasible for the WMDP."
                    )

                validated_actions[t].append(action_tuple)

        return validated_actions


    def _build_base_DA_model(self) -> None:
        """
        Create the delayed allocation variables and constraints that do not depend on actions.
        """

        # Variables (x)
        flow_index: List[FlowKey] = []
        for t in self.Tset:
            for j in self.Jset:
                for state in self.state_space.S[j].states[t]:
                    for action in self.state_space.A[j]:
                        flow_index.append((t, j, state.label, action))
        self.x = self.model.addVars(flow_index, name="x", lb=0.0)

        # Initial states
        for j in self.Jset:
            self.model.addConstr(
                gp.quicksum(
                    self.x[0, j, state.label, action]
                    for state in self.state_space.S[j].states[0]
                    for action in self.state_space.A[j]
                )
                == 1,
                name=f"initial_flow_{j}",
            )

        # Conservation of flow across periods
        for t in range(1, self.T):
            for j in self.Jset:
                for state in self.state_space.S[j].states[t]:
                    self.model.addConstr(
                        gp.quicksum(
                            self.x[t, j, state.label, action]
                            for action in self.state_space.A[j]
                        )
                        == gp.quicksum(
                            self.state_space.S[j].P[t - 1].get(
                                (prev_state.label, action, state.label),
                                0.0,
                            )
                            * self.x[t - 1, j, prev_state.label, action]
                            for prev_state in self.state_space.S[j].states[t - 1]
                            for action in self.state_space.A[j]
                        ),
                        name=f"state_flow_{t}_{j}_{state.label}",
                    )

        # Objective
        self.model.setObjective(
            gp.quicksum(
                state.reward[action] * self.x[t, j, state.label, action]
                for t in self.Tset
                for j in self.Jset
                for state in self.state_space.S[j].states[t]
                for action in self.state_space.A[j]
            ),
            GRB.MAXIMIZE,
        )


    def _build_linking_constraints(self) -> None:
        """
        Create the linking constraints before actions are added.
        """
        for t in self.Tset:
            for j in self.Jset:
                for action in self.state_space.A[j]:
                    self.linking_x_constraints[t, j, action] = self.model.addConstr(
                        gp.quicksum(
                            self.x[t, j, state.label, action]
                            for state in self.state_space.S[j].states[t]
                        )
                        == gp.quicksum(
                            self.pi_set[t][action_index]
                            for action_index, joint_action in enumerate(self.action_set[t])
                            if joint_action[j] == action
                        ),
                        name=f"linking_x_{t}_{j}_{action}",
                    )

            self.probability_simplex_constraints[t] = self.model.addConstr(
                gp.quicksum(
                    self.pi_set[t][action_index]
                    for action_index in range(len(self.action_set[t]))
                )
                == 1,
                name=f"probability_simplex_{t}",
            )


    def add_action(self, t: int, action: ActionVector) -> None:
        """
        Add one feasible joint action to one period of the model.
        """
        action = tuple(action)
        if t not in self.Tset:
            raise ValueError(f"Period {t} is invalid for this delayed allocation model.")
        if action in self.action_set[t]:
            return

        action_index = len(self.action_set[t])
        self.action_set[t].append(action)
        new_var = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            name=f"pi_{t}_{action_index}",
            lb=0.0,
        )
        self.pi_set[t].append(new_var)
        self.model.update()

        for j in self.Jset:
            component_action = action[j]
            self.model.chgCoeff(
                self.linking_x_constraints[t, j, component_action],
                new_var,
                -1.0,
            )
        self.model.chgCoeff(
            self.probability_simplex_constraints[t],
            new_var,
            1.0,
        )


    def _extract_marginal_flows(self) -> Dict[FlowKey, float]:
        """Return the current delayed allocation solution on the x variables."""
        marginal_flows: Dict[FlowKey, float] = {}
        for t in self.Tset:
            for j in self.Jset:
                for state in self.state_space.S[j].states[t]:
                    for action in self.state_space.A[j]:
                        marginal_flows[(t, j, state.label, action)] = self.x[
                            t, j, state.label, action
                        ].X
        return marginal_flows


    def _extract_duals(self) -> DelayedAllocationDuals:
        """
        Collect dual values needed by the pricing problem.
        """
        linking = {
            t: {
                (j, action): self.linking_x_constraints[t, j, action].Pi
                for j in self.Jset
                for action in self.state_space.A[j]
            }
            for t in self.Tset
        }
        simplex = {
            t: self.probability_simplex_constraints[t].Pi
            for t in self.Tset
        }
        return DelayedAllocationDuals(linking=linking, simplex=simplex)


    def _extract_positive_pi_actions(self) -> Dict[int, List[Tuple[ActionVector, float]]]:
        """
        Collect joint actions with positive pi values for each period.
        """
        positive_pi_actions: Dict[int, List[Tuple[ActionVector, float]]] = {}
        for t in self.Tset:
            positive_pi_actions[t] = [
                (action, self.pi_set[t][action_index].X)
                for action_index, action in enumerate(self.action_set[t])
                if self.pi_set[t][action_index].X > self.tolerance
            ]
        return positive_pi_actions


    def optimize(self) -> DelayedAllocationResult:
        """
        Optimize delayed allocation and return primal solution, dual prices, and policy
        """

        # Optimize policy
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                "Optimization ended with status "
                f"{self.model.Status}."
            )

        # Extract result
        objective_value = self.model.ObjVal
        marginal_flows = self._extract_marginal_flows()
        positive_pi_actions = self._extract_positive_pi_actions()

        # Initialize vertex policy
        policy = VertexPolicy(
            positive_pi_actions=positive_pi_actions,
            marginal_flows=marginal_flows,
            tolerance=self.tolerance,
        )

        # Return result and policy
        return DelayedAllocationResult(
            objective_value=objective_value,
            marginal_flows=marginal_flows,
            positive_pi_actions=positive_pi_actions,
            policy=policy,
            duals=self._extract_duals(),
            lower_bound=objective_value,
            num_new_actions=0
        )


    def refine(
        self,
        duals: DelayedAllocationDuals,
        separation_method: Separation,
        verbose: bool = True
    ) -> DelayedAllocationResult:
        """Price new actions, update the model, and return the new delayed allocation solution.

        Parameters
        ----------
        duals
            Dual values returned by :meth:`optimize` or a previous
            :meth:`refine` call.
        separation_method
            Pricing oracle used to search for improving joint actions.
        """

        if verbose:
            print("\n\tRefining DA model...")

        # Check model status
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                "refine requires an existing optimal delayed allocation solution. "
                "Call optimize before refine."
            )

        # Initialize pricing and added actions
        added_actions: Dict[int, List[Tuple[ActionVector, float]]] = {
            t: [] for t in self.Tset
        }
        num_new_actions = 0

        # Check separation problem per period
        for t in self.Tset:
            # Solve pricing based on separation method
            new_action, reduced_cost = separation_method.solve_pricing_problem(
                self.wmdp,
                duals.linking[t],
                duals.simplex[t],
            )

            # Check if action would improve the solution (given a tolerance)
            if reduced_cost < self.tolerance:
                continue

            # Add action to model
            self.add_action(t, new_action)
            added_actions[t].append((new_action, reduced_cost))
            num_new_actions += 1

        for t in self.Tset:
            if not added_actions[t]:
                continue
            
            if verbose:
                print(f"\t\tPeriod {t}: added actions")
                for action, reduced_cost in added_actions[t]:
                    print(f"\t\t\t{action} (reduced cost = {reduced_cost:.6g})")

        result = self.optimize()
        result.num_new_actions = num_new_actions
        result.added_actions = added_actions
        return result
