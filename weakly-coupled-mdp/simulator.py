"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Simulation utilities for weakly coupled MDP policies.
"""

import random
from typing import Dict, List, Sequence

from policy import Policy
from wmdp import StateComponent, WMDP


class Simulator:
    """Simulate sample paths of a WMDP under a policy."""

    def __init__(self, wmdp: WMDP, policy: Policy) -> None:
        """Initialize the simulator with a WMDP instance and policy."""
        self.wmdp = wmdp
        self.policy = policy
        self.random_generator = random.Random(0)

    def simulate(self) -> Dict[str, Sequence]:
        """Simulate one trajectory from the initial state.

        The initial state is the first joint state generated for period 0.
        For each nonterminal period, the simulator queries the policy for a
        feasible joint action and samples each component's next state from its
        transition probabilities.
        """
        initial_states = self.wmdp.generate_states(0)
        if not initial_states:
            raise RuntimeError("No initial states are available for period 0.")

        state = list(initial_states[0])
        states: List[List[StateComponent]] = [state]
        actions: List[List[int]] = []
        rewards: List[float] = []

        for period in range(self.wmdp.T - 1):
            joint_action = list(self.policy.get_action(period=period, state=state))
            self._validate_joint_action(joint_action)

            actions.append(joint_action)
            rewards.append(self._get_reward(state, joint_action))

            state = [
                self._sample_next_state(
                    period=period,
                    component=component,
                    current_state=state[component],
                    action=joint_action[component],
                )
                for component in range(self.wmdp.J)
            ]
            states.append(state)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "total_reward": sum(rewards),
        }

    def _validate_joint_action(self, joint_action: Sequence[int]) -> None:
        """Validate action dimensions and linking feasibility."""
        if len(joint_action) != self.wmdp.J:
            raise ValueError("The joint action must include one action per component.")
        if not self.wmdp.is_action_feasible(joint_action):
            raise ValueError(f"The policy returned an infeasible action: {joint_action}.")

    def _get_reward(
        self,
        state: Sequence[StateComponent],
        joint_action: Sequence[int],
    ) -> float:
        """Return the total immediate reward for a joint state-action pair."""
        reward = 0.0
        for component, component_state in enumerate(state):
            reward += component_state.reward[joint_action[component]]
        return reward

    def _sample_next_state(
        self,
        period: int,
        component: int,
        current_state: StateComponent,
        action: int,
    ) -> StateComponent:
        """Sample one component's next state from its transition kernel."""
        component_model = self.wmdp.state_space.S[component]
        transition_kernel = component_model.P[period]

        next_state_labels = []
        probabilities = []
        for next_state in component_model.states[period + 1]:
            probability = transition_kernel.get(
                (current_state.label, action, next_state.label),
                0.0,
            )
            if probability > 0.0:
                next_state_labels.append(next_state.label)
                probabilities.append(probability)

        if not next_state_labels:
            raise RuntimeError(
                "No transition probabilities are available for "
                f"component {component}, period {period}, state "
                f"{current_state.label}, action {action}."
            )

        sampled_label = self.random_generator.choices(
            next_state_labels,
            weights=probabilities,
            k=1,
        )[0]

        for next_state in component_model.states[period + 1]:
            if next_state.label == sampled_label:
                return next_state

        raise RuntimeError(f"Sampled unknown next state label: {sampled_label}.")
