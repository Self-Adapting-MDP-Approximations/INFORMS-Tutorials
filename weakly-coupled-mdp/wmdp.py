"""
Core data structures and builder utilities for weakly coupled MDPs.

The intended workflow for users is:

1. define the action set for each component;
2. build each component with ``build_component(...)``;
3. define the linking constraints with ``build_linking_constraints(...)`` or
   ``build_budget_constraint(...)``;
4. assemble the final model with ``build_wmdp(...)``.

The helper functions below are meant to expose the model ingredients directly,
so users can construct their own instances in a package-friendly way.
"""

# =================================================
# Weakly Coupled DP Classes
# =================================================

import sys
from typing import Dict, Hashable, List, Mapping, Sequence, Set, Tuple


class LinkingConstraints:
    """Linking constraints over component actions."""

    # Number of components.
    J:int
    # Action set for each component.
    A:Sequence[Sequence[int]]
    # Number of linking constraints.
    K:int
    # Left-hand-side coefficients, keyed by (k, j, a).
    C:Dict[Tuple[int, int, int], float]
    # Right-hand-side values, keyed by constraint index.
    b:List[float]
    # Convenience set of component indices.
    Jset:Set[int]
    # Convenience set of constraint indices.
    Kset:Set[int]

    def __init__(
        self,
        J: int,
        A: Sequence[Sequence[int]],
        K: int,
        C: Mapping[Tuple[int, int, int], float],
        b: Sequence[float],
    ) -> None:
        """Initialize the linking-constraint system."""
        self.J = J
        self.A = A
        self.K = K
        self.C = dict(C)
        self.b = list(b)
        self.Jset = {j for j in range(J)}
        self.Kset = {k for k in range(K)}

    def is_feasible(self, x: Sequence[int]) -> bool:
        """Return True if action vector ``x`` satisfies every linking constraint."""
        for k in range(self.K):
            if sum([self.C[k,j,x[j]] for j in self.Jset]) > self.b[k]:
                return False
        return True

    def generate_feasible_solutions(
        self,
    ) -> List[Tuple[List[int], Dict[Tuple[int, int], int]]]:
        """Enumerate all feasible action vectors.

        Returns
        -------
        list
            A list of pairs ``(action_vector, indicator_map)`` where
            ``indicator_map[(j, a)]`` is 1 if component ``j`` selects action ``a``
            and 0 otherwise.
        """

        def generate_feasible_solutions_recursive(
            solutions: List[Tuple[List[int], Dict[Tuple[int, int], int]]],
            x: List[int],
            n: int,
        ) -> None:
            if n == self.J:
                if self.is_feasible(x):
                    solutions.append( (x.copy(), {(j,a) : 1 if x[j] == a else 0 for j in self.Jset for a in self.A[j]} ) )
            else:
                for a in self.A[n]:
                    x[n] = a
                    generate_feasible_solutions_recursive(solutions, x, n+1)

        solutions = []
        x = [0 for j in self.Jset]
        generate_feasible_solutions_recursive(solutions, x, 0)
        return solutions

    def add_linking_constraint(
        self,
        pi: Mapping[Tuple[int, int], float],
        pi_0: float,
    ) -> None:
        """Append a new linking constraint to the system."""
        self.K = self.K + 1
        self.Kset = {k for k in range(self.K)}
        for j in self.Jset:
            for a in self.A[j]:
                self.C[(self.K-1, j, a)] = pi[(j,a)]
        self.b.append(pi_0)
        
    def __str__(self):
        """Return a readable string representation of the constraint system."""
        ct_str = ""
        for k in self.Kset:
            if k > 0:
                ct_str += "\n"
            for j in self.Jset:
                for a in self.A[j]:
                    if self.C[k,j,a] != 0:
                    # if True:
                        if self.C[k,j,a] >= 0:
                            ct_str += "+ "
                        else:
                            ct_str += "- "                        
                        ct_str += str(abs(self.C[k,j,a])) + " * y_{" + str(k) + "," + str(j) + "," + str(a) + "} "
            ct_str += "<= " + str(self.b[k])
        return ct_str


class StateComponent:
    """Component-level state description."""

    # State label.
    label:Hashable
    # Component index.
    component:int
    # Reward by action.
    reward:Mapping[int, float]

    def __init__(
        self,
        label: Hashable,
        component: int,
        reward: Mapping[int, float],
    ) -> None:
        """Initialize a component state."""
        self.label = label
        self.component = component
        self.reward = reward

    def __str__(self):        
        """Return a readable string representation of the component state."""
        return "(" + self.label + "," + str(self.component) + "," + str(self.reward) + ")"

    def __repr__(self) -> str:
        """Reuse the readable string representation in containers."""
        return self.__str__()


class StateSpaceComponent:
    """State-space data for a single component."""

    # Component index.
    j:int
    # Number of periods.
    T:int
    # States by period.
    states:Sequence[Sequence[StateComponent]]
    # Action set, assumed common across periods.
    A:Sequence[int]
    # Transition probability data by period.
    P:Sequence[Mapping[Tuple[Hashable, int, Hashable], float]]
    
    def __init__(
        self,
        j: int,
        T: int,
        states: Sequence[Sequence[StateComponent]],
        A: Sequence[int],
        P: Sequence[Mapping[Tuple[Hashable, int, Hashable], float]],
    ) -> None:
        """Initialize the state space for one component."""
        self.j = j
        self.T = T
        self.states = states
        self.A = A
        self.P = P



class StateSpace:
    """Joint state-space wrapper across all components."""

    # Number of components.
    J:int
    # Number of periods.
    T:int
    # Component state spaces.
    S:Sequence[StateSpaceComponent]
    # Component action spaces.
    A:Sequence[Sequence[int]]


    def __init__(
        self,
        J: int,
        T: int,
        S: Sequence[StateSpaceComponent],
        A: Sequence[Sequence[int]],
    ) -> None:
        """Initialize the joint state-space container."""
        self.J = J
        self.T = T
        self.S = S
        self.A = A


    def generate_states(self, t: int) -> List[List[StateComponent]]:
        """Generate all joint states at period ``t``."""
        states: List[List[StateComponent]] = []

        def state_permutations(state: List[StateComponent], t: int, j: int) -> None:
            if j == self.J:
                states.append(state)
            else:
                for s in self.S[j].states[t]:
                    state_permutations(state + [s], t, j+1)

        state_permutations([], t, 0)
        return states


class WMDP:
    """Container for a weakly coupled Markov decision process instance."""

    # Number of components.
    J:int
    # Number of periods.
    T:int
    # Joint state-space description.
    state_space:StateSpace
    # Linking constraints over joint actions.
    linking_constraints:LinkingConstraints

    def __init__(
        self,
        state_space: StateSpace,
        linking_constraints: LinkingConstraints,
    ) -> None:
        """Initialize a WMDP instance and validate basic dimensions."""
        if state_space.J != linking_constraints.J:
            raise ValueError(
                "State space and linking constraints must use the same number of components."
            )

        self.J = state_space.J
        self.T = state_space.T
        self.state_space = state_space
        self.linking_constraints = linking_constraints

    def generate_states(self, t: int) -> List[List[StateComponent]]:
        """Delegate joint-state enumeration to the underlying state space."""
        return self.state_space.generate_states(t)

    def is_action_feasible(self, action: Sequence[int]) -> bool:
        """Return True if ``action`` satisfies the linking constraints."""
        return self.linking_constraints.is_feasible(action)

    def generate_feasible_actions(
        self,
    ) -> List[Tuple[List[int], Dict[Tuple[int, int], int]]]:
        """Enumerate feasible joint actions from the linking-constraint model."""
        return self.linking_constraints.generate_feasible_solutions()

    def __str__(self) -> str:
        """Return a readable string summary of the instance."""
        return (
            f"WMDP(J={self.J}, T={self.T}, "
            f"num_constraints={self.linking_constraints.K})"
        )


def create_binary_action_sets(J: int) -> List[List[int]]:
    """Create binary action sets ``[0, 1]`` for all components.

    Parameters
    ----------
    J
        Number of components.

    Returns
    -------
    list
        A list of action sets, one per component.

    Example
    -------
    ``create_binary_action_sets(3)`` returns ``[[0, 1], [0, 1], [0, 1]]``.
    """
    return [[0, 1] for _ in range(J)]


def build_states_for_periods(
    component: int,
    state_data_by_period: Sequence[Sequence[Tuple[Hashable, Mapping[int, float]]]],
) -> List[List[StateComponent]]:
    """Build period-by-period component states from lightweight input data.

    Parameters
    ----------
    component
        Component index.
    state_data_by_period
        For each period, provide a list of pairs ``(state_label, reward_by_action)``.

    Returns
    -------
    list
        A list of lists of ``StateComponent`` objects, indexed by period.

    Example
    -------
    ``state_data_by_period`` can be written as:

    ``[
        [("healthy", {0: 0.0, 1: 2.0}), ("failed", {0: -1.0, 1: 0.0})],
        [("healthy", {0: 0.0, 1: 2.0}), ("failed", {0: -1.0, 1: 0.0})],
    ]``

    which means that the component has two states in each of two periods, and
    each state stores a reward value for each available action.
    """
    return [
        [
            StateComponent(label=label, component=component, reward=reward)
            for label, reward in states_in_period
        ]
        for states_in_period in state_data_by_period
    ]


def build_component(
    component: int,
    actions: Sequence[int],
    state_data_by_period: Sequence[Sequence[Tuple[Hashable, Mapping[int, float]]]],
    transitions_by_period: Sequence[Mapping[Tuple[Hashable, int, Hashable], float]],
) -> StateSpaceComponent:
    """Build one component model from state, reward, and transition data.

    Parameters
    ----------
    component
        Component index.
    actions
        Action set available to this component.
    state_data_by_period
        For each period, a list of pairs ``(state_label, reward_by_action)``.
    transitions_by_period
        Transition kernel for each nonterminal period. Each kernel is a
        dictionary keyed by ``(current_state, action, next_state)``.

    Returns
    -------
    StateSpaceComponent
        A fully built component model.

    Example
    -------
    ``build_component(...)`` can be called as:

    ``build_component(
        component=0,
        actions=[0, 1],
        state_data_by_period=[
            [("healthy", {0: 0.0, 1: 2.0}), ("failed", {0: -1.0, 1: 0.0})],
            [("healthy", {0: 0.0, 1: 2.0}), ("failed", {0: -1.0, 1: 0.0})],
        ],
        transitions_by_period=[
            {
                ("healthy", 0, "healthy"): 0.8,
                ("healthy", 0, "failed"): 0.2,
                ("healthy", 1, "healthy"): 1.0,
                ("failed", 0, "failed"): 1.0,
                ("failed", 1, "healthy"): 0.7,
                ("failed", 1, "failed"): 0.3,
            }
        ],
    )``

    Here, action ``1`` can be interpreted as a repair action and the transition
    dictionary specifies the probability of each next state.
    """
    states = build_states_for_periods(
        component=component,
        state_data_by_period=state_data_by_period,
    )
    return StateSpaceComponent(
        j=component,
        T=len(state_data_by_period),
        states=states,
        A=list(actions),
        P=list(transitions_by_period),
    )


def build_linking_constraints(
    action_sets: Sequence[Sequence[int]],
    constraint_coefficients: Sequence[Mapping[Tuple[int, int], float]],
    rhs_values: Sequence[float],
) -> LinkingConstraints:
    """Build linking constraints from per-constraint coefficient dictionaries.

    Parameters
    ----------
    action_sets
        Action set for each component.
    constraint_coefficients
        One dictionary per linking constraint. Each dictionary is keyed by
        ``(component, action)`` and stores the associated coefficient.
        Missing keys are interpreted as zero.
    rhs_values
        Right-hand-side value for each linking constraint.

    Returns
    -------
    LinkingConstraints
        A linking-constraint object ready to be passed into ``build_wmdp(...)``.

    Example
    -------
    To impose the constraint
    ``1 * 1{a_0 = 1} + 2 * 1{a_1 = 1} <= 2``, use:

    ``build_linking_constraints(
        action_sets=[[0, 1], [0, 1]],
        constraint_coefficients=[
            {
                (0, 1): 1.0,
                (1, 1): 2.0,
            }
        ],
        rhs_values=[2.0],
    )``

    Coefficients for omitted ``(component, action)`` pairs are treated as zero.
    """
    J = len(action_sets)
    K = len(constraint_coefficients)
    if len(rhs_values) != K:
        raise ValueError("The number of right-hand-side values must match the number of constraints.")

    C: Dict[Tuple[int, int, int], float] = {}
    for k, coefficients in enumerate(constraint_coefficients):
        for j, actions in enumerate(action_sets):
            for a in actions:
                C[(k, j, a)] = coefficients.get((j, a), 0.0)

    return LinkingConstraints(
        J=J,
        A=action_sets,
        K=K,
        C=C,
        b=list(rhs_values),
    )


def build_budget_constraint(
    action_sets: Sequence[Sequence[int]],
    action_costs: Mapping[Tuple[int, int], float],
    budget: float,
) -> LinkingConstraints:
    """Build a single knapsack-style budget constraint.

    Parameters
    ----------
    action_sets
        Action set for each component.
    action_costs
        Cost keyed by ``(component, action)``. Missing keys are interpreted as zero.
    budget
        Total available budget.

    Returns
    -------
    LinkingConstraints
        A single knapsack-style linking constraint.

    Example
    -------
    If choosing action ``1`` for either component costs one unit, then
    ``build_budget_constraint(
        action_sets=[[0, 1], [0, 1]],
        action_costs={(0, 1): 1.0, (1, 1): 1.0},
        budget=1.0,
    )``
    enforces that at most one component can choose action ``1``.
    """
    return build_linking_constraints(
        action_sets=action_sets,
        constraint_coefficients=[action_costs],
        rhs_values=[budget],
    )


def build_wmdp(
    components: Sequence[StateSpaceComponent],
    linking_constraints: LinkingConstraints,
) -> WMDP:
    """Build a WMDP from user-specified components and linking constraints.

    Parameters
    ----------
    components
        A list of component models built with ``build_component(...)``.
    linking_constraints
        A linking-constraint object built with
        ``build_linking_constraints(...)`` or ``build_budget_constraint(...)``.

    Returns
    -------
    WMDP
        A weakly coupled MDP instance.

    Example
    -------
    ``wmdp = build_wmdp(
        components=[component_0, component_1],
        linking_constraints=budget_constraint,
    )``

    After construction, useful methods include:
    ``wmdp.generate_states(t)``, ``wmdp.is_action_feasible(action)``, and
    ``wmdp.generate_feasible_actions()``.
    """
    if not components:
        raise ValueError("At least one component is required to build a WMDP.")

    J = len(components)
    T = components[0].T
    action_sets = [list(component.A) for component in components]

    for component in components:
        if component.T != T:
            raise ValueError("All components must have the same number of periods.")

    state_space = StateSpace(J=J, T=T, S=list(components), A=action_sets)
    return WMDP(state_space=state_space, linking_constraints=linking_constraints)
