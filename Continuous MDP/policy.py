# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Shared policy-evaluation helpers for the tutorial inventory example.

The tutorial uses the same approximate greedy-policy construction for both
FALP and SGALP. Centralizing it here avoids duplicated code and keeps the
focus on the algorithmic differences rather than bookkeeping details.
"""

from __future__ import annotations
from config import PolicyEvaluationConfig
import numpy as np

def _active_num_random_features(model):
    """
    Infer how many nonconstant basis functions are active in a fitted model.

    Args:
        model: FALP, SGALP, or PSMD-like object with basis coefficients.
    """
    if hasattr(model, "current_num_random_features") and model.current_num_random_features is not None:
        return int(model.current_num_random_features)
    if hasattr(model, "num_random_features"):
        return int(model.num_random_features)
    return len(np.asarray(model.coef, dtype=float)) - 1


def _evaluate_basis_batch(model, states):
    """
    Evaluate the model basis on many states using the active basis size.

    Args:
        model: Fitted model whose basis should be evaluated.
        states: One-dimensional collection of state values.
    """
    use_count = _active_num_random_features(model)
    if hasattr(model.basis, "eval_basis_batch"):
        return model.basis.eval_basis_batch(states, num_random_features=use_count)
    return np.asarray(
        [model.basis.eval_basis(np.asarray([state], dtype=float), num_random_features=use_count) for state in states],
        dtype=float,
    )


def build_greedy_policy_lookup(model, config: PolicyEvaluationConfig | None = None):
    """
    Precompute greedy actions on a dense one-dimensional state grid.

    Args:
        model: Fitted value-function approximation used in one-step lookahead.
        config: Policy-evaluation settings controlling the state grid and the
            noise batch used in approximate expectations.
    """
    config = PolicyEvaluationConfig() if config is None else config

    mdp = model.mdp
    actions = mdp.get_discrete_actions()
    noise = mdp.sample_noise_batch(
        num_samples=config.policy_noise_batch_size,
        random_seed=config.policy_noise_seed,
    )

    state_grid = np.linspace(mdp.lower_state_bound, mdp.upper_state_bound, config.state_grid_size)
    policy_actions = np.empty_like(state_grid)

    coef = np.asarray(model.coef, dtype=float)

    for i, state_value in enumerate(state_grid):
        transition_summary = mdp.evaluate_state_action_batch(
            states=np.full(len(actions), state_value, dtype=float),
            actions=actions,
            noise_batch=noise,
        )
        next_states = transition_summary["next_states"]
        expected_cost = transition_summary["expected_cost"]

        basis_vals = _evaluate_basis_batch(model, next_states.reshape(-1)).reshape(len(actions), len(noise), -1)
        expected_future_value = basis_vals.mean(axis=1) @ coef

        q_values = expected_cost + mdp.discount * expected_future_value
        policy_actions[i] = actions[int(np.argmin(q_values))]

    return state_grid, policy_actions


def estimate_upper_bound_fast(
    model,
    config: PolicyEvaluationConfig | None = None,
    return_se: bool = False,
):
    """
    Simulate the discounted cost of the approximate greedy policy.

    Args:
        model: Fitted value-function approximation used to build the policy.
        config: Policy-evaluation settings controlling trajectory simulation.
        return_se: Whether to also return the Monte Carlo standard error.
    """
    config = PolicyEvaluationConfig() if config is None else config
    mdp = model.mdp

    state_grid, policy_actions = build_greedy_policy_lookup(model, config=config)
    total_costs = []

    for traj in range(config.num_trajectories):
        state = np.asarray([config.initial_state], dtype=float)
        discounted_cost = 0.0
        demand_path = mdp.sample_noise_batch(
            num_samples=config.horizon,
            random_seed=config.simulation_seed + traj + 1,
        )

        for time_index in range(config.horizon):
            state_value = float(state[0])
            action_index = np.abs(state_grid - state_value).argmin()
            action = policy_actions[action_index]

            stage_cost = mdp.get_cost_given_noise(state, action, demand_path[time_index])
            discounted_cost += (mdp.discount ** time_index) * stage_cost
            state = mdp.get_next_state_given_noise(state, action, demand_path[time_index])

        total_costs.append(discounted_cost)

    total_costs = np.asarray(total_costs, dtype=float)
    mean_cost = float(np.mean(total_costs))

    if return_se:
        se_cost = float(np.std(total_costs, ddof=1) / np.sqrt(len(total_costs)))
        return mean_cost, se_cost

    return mean_cost