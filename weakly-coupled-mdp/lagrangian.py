"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Lagrangian-style expectation relaxation for weakly coupled MDPs.

This module solves a relaxation in which linking/resource constraints are
enforced only in expectation through component marginal flows. The extracted
policy samples each component action from those marginals and repairs the
sampled joint action online by replacing infeasible component actions with
action ``0``.
"""

from dataclasses import dataclass
import random
from typing import Dict, Mapping, Sequence, Tuple

from policy import *
from wmdp import *

import gurobipy as gp
from gurobipy import GRB


FlowKey = Tuple[int, int, object, int]


class LagrangianPolicy(Policy):
    """Policy induced by the expectation-relaxed marginal flows.

    For a queried period and joint state, each component action is sampled from
    the conditional distribution implied by ``x[t, j, state, action]``. Actions
    are sampled component by component. If adding the sampled action would make
    the partial joint action infeasible under the original linking constraints,
    the policy uses action ``0`` for that component instead.
    """

    def __init__(
        self,
        marginal_flows: Mapping[FlowKey, float],
        wmdp: WMDP,
        seed: int = 0,
        tolerance: float = 1e-9,
    ) -> None:
        """Initialize the sampling policy from solved marginal flows."""
        self.marginal_flows = dict(marginal_flows)
        self.wmdp = wmdp
        self.linkct = wmdp.linking_constraints
        self.random_generator = random.Random(seed)
        self.tolerance = tolerance

        for component_actions in self.linkct.A:
            if 0 not in component_actions:
                raise ValueError("LagrangianPolicy requires action 0 as the fallback action.")


    def _action_distribution(
        self,
        period: int,
        component: int,
        state_label: object,
    ) -> Tuple[Sequence[int], Sequence[float]]:
        """Return the conditional action distribution for one component state."""
        actions = list(self.linkct.A[component])
        weights = [
            max(
                0.0,
                self.marginal_flows.get((period, component, state_label, action), 0.0),
            )
            for action in actions
        ]
        total_weight = sum(weights)

        if total_weight <= self.tolerance:
            return [0], [1.0]

        return actions, [weight / total_weight for weight in weights]


    def _is_partial_action_feasible(
        self,
        partial_action: Sequence[int],
        candidate_action: int,
    ) -> bool:
        """Check feasibility after appending a candidate and filling the rest with 0."""
        candidate_joint_action = (
            list(partial_action)
            + [candidate_action]
            + [0 for _ in range(self.linkct.J - len(partial_action) - 1)]
        )
        return self.linkct.is_feasible(candidate_joint_action)


    def get_action(self, period: int, state: Sequence[StateComponent]) -> ActionVector:
        """Sample and repair a feasible joint action for the current joint state."""
        if len(state) != self.linkct.J:
            raise ValueError("The number of component states must match the number of components.")

        joint_action = []
        for component, component_state in enumerate(state):
            actions, probabilities = self._action_distribution(
                period=period,
                component=component,
                state_label=component_state.label,
            )
            sampled_action = self.random_generator.choices(
                actions,
                weights=probabilities,
                k=1,
            )[0]

            if self._is_partial_action_feasible(joint_action, sampled_action):
                joint_action.append(sampled_action)
            elif self._is_partial_action_feasible(joint_action, 0):
                joint_action.append(0)
            else:
                raise RuntimeError(
                    "The fallback action 0 does not restore linking feasibility."
                )

        return tuple(joint_action)


@dataclass
class LagrangianResult:
    """Solution returned by the expectation-relaxed Lagrangian model."""

    objective_value: float
    marginal_flows: Dict[FlowKey, float]
    expected_resource_use: Dict[Tuple[int, int], float]
    policy: LagrangianPolicy


class Lagrangian:
    """
    LP model with linking constraints enforced in expectation.
    """

    def __init__(
        self,
        wmdp: WMDP,
        seed: int = 0,
        tolerance: float = 1e-9,
    ) -> None:
        """Build the expectation-relaxed model for the provided WMDP."""
        self.wmdp = wmdp
        self.state_space: StateSpace = wmdp.state_space
        self.linkct = wmdp.linking_constraints
        self.seed = seed
        self.tolerance = tolerance

        self.J = wmdp.J
        self.T = wmdp.T
        self.Jset = range(self.J)
        self.Tset = range(self.T)

        self.model = gp.Model("lagrangian_expectation_relaxation")
        self.x = None

        self._build_model()
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("FeasibilityTol", self.tolerance)


    def _build_model(self) -> None:
        """
        Create the component-flow LP with expected linking constraints.
        """

        # Variables (x)
        flow_index = []
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

        # Linking constraints are satisfied on expectation
        for t in self.Tset:
            for k in self.linkct.Kset:
                self.model.addConstr(
                    gp.quicksum(
                        self.linkct.C[(k, j, action)]
                        * self.x[t, j, state.label, action]
                        for j in self.Jset
                        for state in self.state_space.S[j].states[t]
                        for action in self.state_space.A[j]
                    )
                    <= self.linkct.b[k],
                    name=f"expected_linking_{t}_{k}",
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


    def _extract_marginal_flows(self) -> Dict[FlowKey, float]:
        """Return the solved component state-action flows."""
        marginal_flows: Dict[FlowKey, float] = {}
        for t in self.Tset:
            for j in self.Jset:
                for state in self.state_space.S[j].states[t]:
                    for action in self.state_space.A[j]:
                        marginal_flows[(t, j, state.label, action)] = self.x[
                            t, j, state.label, action
                        ].X
        return marginal_flows


    def _extract_expected_resource_use(self) -> Dict[Tuple[int, int], float]:
        """Return expected resource use keyed by ``(period, constraint)``."""
        expected_resource_use: Dict[Tuple[int, int], float] = {}
        for t in self.Tset:
            for k in self.linkct.Kset:
                expected_resource_use[(t, k)] = sum(
                    self.linkct.C[(k, j, action)]
                    * self.x[t, j, state.label, action].X
                    for j in self.Jset
                    for state in self.state_space.S[j].states[t]
                    for action in self.state_space.A[j]
                )
        return expected_resource_use


    def optimize(self) -> LagrangianResult:
        """Solve the expectation relaxation and return marginal flows and policy."""
        print("Optimizing...")
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                "Lagrangian expectation relaxation did not terminate optimally. "
                f"Gurobi status: {self.model.Status}"
            )

        marginal_flows = self._extract_marginal_flows()
        expected_resource_use = self._extract_expected_resource_use()
        policy = LagrangianPolicy(
            marginal_flows=marginal_flows,
            wmdp=self.wmdp,
            seed=self.seed,
            tolerance=self.tolerance,
        )

        print("\tdone.")
        print(
            "Lagrangian model size:",
            f"vars={self.model.NumVars}",
            f"constrs={self.model.NumConstrs}",
            f"nonzeros={self.model.NumNZs}",
            f"time={self.model.Runtime}s",
        )

        return LagrangianResult(
            objective_value=self.model.ObjVal,
            marginal_flows=marginal_flows,
            expected_resource_use=expected_resource_use,
            policy=policy,
        )


def solve_lagrangian(
    wmdp: WMDP,
    seed: int = 0,
) -> Tuple[float, Dict[FlowKey, float], LagrangianPolicy]:
    """Build and solve the expectation-relaxed Lagrangian LP."""
    lagrangian_model = Lagrangian(wmdp=wmdp, seed=seed)
    result = lagrangian_model.optimize()
    return result.objective_value, result.marginal_flows, result.policy
