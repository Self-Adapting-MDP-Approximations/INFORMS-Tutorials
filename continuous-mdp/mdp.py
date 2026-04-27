# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Minimal MDP layer for the tutorial examples.

Two things live here:
1. a very small discounted-cost MDP interface
2. a concrete single-product inventory model used throughout the notebook

The implementation is intentionally lightweight so that a reader undrestands 
how mathematical models turn into simulation, sampling, and optimization.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from config import InventoryMDPConfig


class MarkovDecisionProcess:
    """
    Minimal interface for a discounted-cost MDP.

    A concrete MDP should explain:
    - how states evolve after action and exogenous noise
    - how one-step costs are computed
    - how to sample states, actions, and noise for approximation routines
    """

    def __init__(self, mdp_name: str, dim_state: int, dim_act: int, discount: float, random_seed: int):
        """
        Initialize the shared metadata for a discounted-cost MDP.

        Args:
            mdp_name: Short human-readable name of the MDP.
            dim_state: Dimension of the state vector.
            dim_act: Dimension of the action vector.
            discount: Discount factor used in Bellman equations.
            random_seed: Base random seed for reproducible sampling.
        """
        self.mdp_name = mdp_name
        self.dim_state = dim_state
        self.dim_act = dim_act
        self.discount = discount
        self.random_seed = random_seed

    def get_next_state_given_noise(self, cur_state, cur_action, noise):
        """
        Return the next state for one realized noise value.

        Args:
            cur_state: Current state before the action is applied.
            cur_action: Action chosen at the current state.
            noise: Exogenous uncertainty realized after the action.
        """
        raise NotImplementedError

    def get_cost_given_noise(self, cur_state, cur_action, noise):
        """
        Return the one-step cost for one realized noise value.

        Args:
            cur_state: Current state before the action is applied.
            cur_action: Action chosen at the current state.
            noise: Exogenous uncertainty realized after the action.
        """
        raise NotImplementedError

    def get_batch_next_state(self, cur_state, cur_action, noise_list=None):
        """
        Return next states for one state-action pair and many noise samples.

        Args:
            cur_state: Current state before the action is applied.
            cur_action: Action chosen at the current state.
            noise_list: Optional list of noise values to reuse.
        """
        raise NotImplementedError

    def get_expected_cost(self, cur_state, cur_action, noise_list=None):
        """
        Estimate the expected one-step cost for one state-action pair.

        Args:
            cur_state: Current state before the action is applied.
            cur_action: Action chosen at the current state.
            noise_list: Optional list of noise values to reuse.
        """
        raise NotImplementedError

    def sample_noise_batch(self, num_samples=None, random_seed=None):
        """
        Draw a batch of exogenous noise samples.

        Args:
            num_samples: Number of draws to generate.
            random_seed: Optional seed overriding the default seed.
        """
        raise NotImplementedError

    def sample_fixed_noise_batch(self, num_samples=None):
        """
        Draw and cache a reusable batch of exogenous noise samples.

        Args:
            num_samples: Number of draws to generate.
        """
        raise NotImplementedError

    def sample_constraint_state_actions(self, num_samples):
        """
        Draw state-action pairs for Bellman-constraint sampling.

        Args:
            num_samples: Number of state-action pairs to draw.
        """
        raise NotImplementedError

    def sample_state_relevance_states(self, num_samples):
        """
        Draw states used in the ALP objective or guiding constraints.

        Args:
            num_samples: Number of states to draw.
        """
        raise NotImplementedError

    def evaluate_state_action_batch(self, states, actions, noise_batch):
        """
        Evaluate many state-action pairs against a shared noise batch.

        Args:
            states: Collection of states.
            actions: Collection of actions paired with `states`.
            noise_batch: Shared batch of exogenous noise values.
        """
        raise NotImplementedError

    def get_batch_init_state(self, num_traj):
        """
        Draw starting states for Monte Carlo trajectories.

        Args:
            num_traj: Number of trajectories to initialize.
        """
        raise NotImplementedError

    def get_discrete_actions(self):
        """
        Return the discrete action grid used in the tutorial examples.
        """
        raise NotImplementedError

    def is_state_action_feasible(self, state, action):
        """
        Check whether a state-action pair satisfies model bounds.

        Args:
            state: State to check.
            action: Action to check.
        """
        raise NotImplementedError

    # Backward-compatible aliases used by the existing notebook.
    def get_batch_mdp_noise(self, num_samples=None, random_seed=None):
        """
        Backward-compatible alias for `sample_noise_batch`.

        Args:
            num_samples: Number of draws to generate.
            random_seed: Optional seed overriding the default seed.
        """
        return self.sample_noise_batch(num_samples=num_samples, random_seed=random_seed)

    def sample_fix_batch_mdp_noise(self, num_samples=None):
        """
        Backward-compatible alias for `sample_fixed_noise_batch`.

        Args:
            num_samples: Number of draws to generate.
        """
        return self.sample_fixed_noise_batch(num_samples=num_samples)

    def get_state_act_for_ALP_constr(self, num_samples):
        """
        Backward-compatible alias for `sample_constraint_state_actions`.

        Args:
            num_samples: Number of state-action pairs to draw.
        """
        return self.sample_constraint_state_actions(num_samples)

    def get_batch_samples_state_relevance(self, num_samples):
        """
        Backward-compatible alias for `sample_state_relevance_states`.

        Args:
            num_samples: Number of states to draw.
        """
        return self.sample_state_relevance_states(num_samples)


@dataclass
class UniformSampler:
    """
    Tiny sampler wrapper so the code reads like the larger package.
    """

    low: float
    high: float

    def rvs(self, size: int, random_state: int | None = None):
        """
        Draw independent uniform samples.

        Args:
            size: Number of samples to generate.
            random_state: Optional seed for reproducibility.
        """
        rng = np.random.RandomState(random_state)
        return rng.uniform(self.low, self.high, size=size)


def positive_part(x: float) -> float:
    """
    Return `max(x, 0)`.

    Args:
        x: Scalar value to truncate below at zero.
    """
    return max(0.0, x)


def draw_truncated_normal(mean: float, std: float, lower: float, upper: float, size: int, seed: int):
    """
    Simple rejection sampler for a truncated normal distribution.

    Args:
        mean: Mean of the untruncated normal distribution.
        std: Standard deviation of the untruncated normal distribution.
        lower: Lower truncation point.
        upper: Upper truncation point.
        size: Number of samples to return.
        seed: Random seed controlling reproducibility.
    """

    rng = np.random.RandomState(seed)
    accepted = []
    while len(accepted) < size:
        draws = rng.normal(loc=mean, scale=std, size=size)
        accepted.extend(draws[(draws >= lower) & (draws <= upper)].tolist())
    return np.asarray(accepted[:size], dtype=float)


class SingleProductInventoryMDP(MarkovDecisionProcess):
    """
    Inventory MDP with:
    - one-dimensional state: inventory level
    - order-before-demand timing
    - partial backlogging
    - zero lead time
    - bounded storage and bounded backlog

    Negative inventory represents backlog.
    """

    def __init__(self, config: InventoryMDPConfig | dict):
        """
        Build the one-product inventory tutorial MDP.

        Args:
            config: Inventory parameters, either as an `InventoryMDPConfig`
                object or a legacy dictionary.
        """
        if isinstance(config, dict):
            config = _inventory_config_from_dict(config)

        super().__init__(
            mdp_name=config.mdp_name,
            dim_state=1,
            dim_act=1,
            discount=config.discount,
            random_seed=config.random_seed,
        )

        self.config = config

        self.lower_state_bound = config.lower_state_bound
        self.upper_state_bound = config.upper_state_bound
        self.max_order = config.max_order

        self.purchase_cost = config.purchase_cost
        self.holding_cost = config.holding_cost
        self.backlog_cost = config.backlog_cost
        self.disposal_cost = config.disposal_cost
        self.lost_sale_cost = config.lost_sale_cost

        self.demand_mean = config.demand_mean
        self.demand_std = config.demand_std
        self.demand_min = config.demand_min
        self.demand_max = config.demand_max

        self.num_noise_samples = config.num_noise_samples
        self.action_step = config.action_step

        # Legacy attribute names retained for notebook compatibility.
        self.dist_mean = self.demand_mean
        self.dist_std = self.demand_std
        self.dist_min = self.demand_min
        self.dist_max = self.demand_max
        self.num_sample_noise = self.num_noise_samples
        self.action_discrete_param = self.action_step

        self.state_sampler = UniformSampler(self.lower_state_bound, self.upper_state_bound)
        self.action_sampler = UniformSampler(0.0, self.max_order)
        self.init_state_sampler = UniformSampler(self.lower_state_bound, self.upper_state_bound)

        self.list_demand_obs = None

    def clip_inventory(self, raw_inventory: float) -> float:
        """
        Project an inventory level onto the feasible state interval.

        Args:
            raw_inventory: Inventory level before truncation.
        """
        return min(max(raw_inventory, self.lower_state_bound), self.upper_state_bound)

    def _coerce_vector(self, values, name: str):
        """
        Convert a scalar or array-like input into a nonempty float vector.

        Args:
            values: Scalar or iterable input to coerce.
            name: Short label used in error messages.
        """
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        return array

    def _broadcast_state_action_arrays(self, states, actions):
        """
        Align state and action arrays so they describe paired samples.

        Args:
            states: One or more states.
            actions: One or more actions.
        """
        state_array = self._coerce_vector(states, "states")
        action_array = self._coerce_vector(actions, "actions")

        if state_array.size == 1 and action_array.size > 1:
            state_array = np.full(action_array.shape, state_array[0], dtype=float)
        elif action_array.size == 1 and state_array.size > 1:
            action_array = np.full(state_array.shape, action_array[0], dtype=float)
        elif state_array.size != action_array.size:
            raise ValueError("states and actions must have the same length, unless one of them is scalar.")

        return state_array, action_array

    def evaluate_state_action_batch(self, states, actions, noise_batch):
        """
        Evaluate one or more state-action pairs against a shared demand batch.

        Args:
            states: Scalar or vector of inventory states.
            actions: Scalar or vector of order quantities.
            noise_batch: Demand samples used to approximate expectations.
        """
        state_array, action_array = self._broadcast_state_action_arrays(states, actions)
        demand_array = self._coerce_vector(noise_batch, "noise_batch")

        raw_inventory = state_array[:, None] + action_array[:, None] - demand_array[None, :]
        next_states = np.clip(raw_inventory, self.lower_state_bound, self.upper_state_bound)

        purchase_term = self.purchase_cost * action_array
        holding_term = self.holding_cost * np.maximum(next_states, 0.0)
        backlog_term = self.backlog_cost * np.maximum(-next_states, 0.0)
        disposal_term = self.disposal_cost * np.maximum(raw_inventory - self.upper_state_bound, 0.0)
        lost_sale_term = self.lost_sale_cost * np.maximum(self.lower_state_bound - raw_inventory, 0.0)

        expected_cost = purchase_term + (
            holding_term + backlog_term + disposal_term + lost_sale_term
        ).mean(axis=1)

        return {
            "states": state_array,
            "actions": action_array,
            "demand": demand_array,
            "raw_inventory": raw_inventory,
            "next_states": next_states,
            "expected_cost": expected_cost,
        }

    def get_next_state_given_noise(self, cur_state, cur_action, noise):
        """
        Realized transition:
            s' = clip(s + a - demand)

        Args:
            cur_state: Current inventory state.
            cur_action: Order quantity placed before demand.
            noise: Realized demand sample.
        """
        summary = self.evaluate_state_action_batch(cur_state, cur_action, noise)
        return np.asarray([summary["next_states"][0, 0]], dtype=float)

    def get_cost_given_noise(self, cur_state, cur_action, noise):
        """
        One-step cost under one realized demand draw.

        Args:
            cur_state: Current inventory state.
            cur_action: Order quantity placed before demand.
            noise: Realized demand sample.
        """
        summary = self.evaluate_state_action_batch(cur_state, cur_action, noise)
        return float(summary["expected_cost"][0])

    def get_batch_next_state(self, cur_state, cur_action, noise_list=None):
        """
        Batch transition map used inside Bellman expectations.

        Args:
            cur_state: Current inventory state.
            cur_action: Order quantity placed before demand.
            noise_list: Optional demand samples to reuse.
        """
        realized_noise = self.list_demand_obs if noise_list is None else np.asarray(noise_list, dtype=float)
        summary = self.evaluate_state_action_batch(cur_state, cur_action, realized_noise)
        return summary["next_states"].reshape(len(realized_noise), 1)

    def get_expected_cost(self, cur_state, cur_action, noise_list=None):
        """
        Monte Carlo approximation of the expected one-step cost.

        Args:
            cur_state: Current inventory state.
            cur_action: Order quantity placed before demand.
            noise_list: Optional demand samples to reuse.
        """
        realized_noise = self.list_demand_obs if noise_list is None else np.asarray(noise_list, dtype=float)
        summary = self.evaluate_state_action_batch(cur_state, cur_action, realized_noise)
        return float(summary["expected_cost"][0])

    def sample_noise_batch(self, num_samples=None, random_seed=None):
        """
        Draw a demand batch from the truncated normal demand model.

        Args:
            num_samples: Number of demand draws to generate.
            random_seed: Optional seed overriding the model seed.
        """
        sample_size = self.num_noise_samples if num_samples is None else num_samples
        seed = self.random_seed if random_seed is None else random_seed
        return draw_truncated_normal(
            mean=self.demand_mean,
            std=self.demand_std,
            lower=self.demand_min,
            upper=self.demand_max,
            size=sample_size,
            seed=seed,
        )

    def sample_fixed_noise_batch(self, num_samples=None):
        """
        Draw and cache the default demand batch used by Bellman expectations.

        Args:
            num_samples: Number of demand draws to generate.
        """
        self.list_demand_obs = self.sample_noise_batch(num_samples=num_samples)
        return self.list_demand_obs

    def sample_constraint_state_actions(self, num_samples):
        """
        Sample state-action pairs for ALP constraint sampling.

        States are sampled uniformly over the continuous state interval, while
        actions are sampled uniformly from the discrete action grid.

        Args:
            num_samples: Number of state-action pairs to draw.
        """

        state_draws = self.state_sampler.rvs(size=num_samples, random_state=self.random_seed + 1)
        action_rng = np.random.RandomState(self.random_seed + 2)
        action_draws = action_rng.choice(self.get_discrete_actions(), size=num_samples, replace=True)
        state_list = [np.asarray([state_draws[i]], dtype=float) for i in range(num_samples)]
        return state_list, np.asarray(action_draws, dtype=float)

    def sample_state_relevance_states(self, num_samples):
        """
        Sample states for the ALP objective.

        We also include a few boundary states because those are often
        especially informative in inventory problems.

        Args:
            num_samples: Number of random state draws before adding boundary
                points.
        """

        state_draws = self.state_sampler.rvs(size=num_samples, random_state=self.random_seed + 3)
        samples = [np.asarray([state_draws[i]], dtype=float) for i in range(num_samples)]
        return samples

    def get_batch_init_state(self, num_traj):
        """
        Draw initial states for Monte Carlo trajectories.

        Args:
            num_traj: Number of trajectories to initialize.
        """
        draws = self.init_state_sampler.rvs(size=num_traj, random_state=self.random_seed + 4)
        return [np.asarray([draws[i]], dtype=float) for i in range(num_traj)]

    def get_discrete_actions(self):
        """
        Return the discrete action grid used in tutorial policy search.
        """
        action_grid = np.arange(0.0, self.max_order + 0.5 * self.action_step, self.action_step, dtype=float)
        action_grid = action_grid[action_grid <= self.max_order + 1e-10]
        if action_grid[-1] < self.max_order - 1e-10:
            action_grid = np.append(action_grid, self.max_order)
        return np.round(action_grid, decimals=10)

    def is_state_action_feasible(self, state, action):
        """
        Check whether the proposed state-action pair respects model bounds.

        Args:
            state: Inventory level to test.
            action: Order quantity to test.
        """
        state_value = float(np.asarray(state, dtype=float)[0])
        if state_value < self.lower_state_bound or state_value > self.upper_state_bound:
            return False
        if action < 0.0 or action > self.max_order:
            return False
        return True


def _inventory_config_from_dict(config_dict: dict):
    """
    Accept both the new names and the legacy dictionary keys.

    Args:
        config_dict: Dictionary of inventory-model parameters.
    """

    return InventoryMDPConfig(
        mdp_name=config_dict.get("mdp_name", "Inventory"),
        discount=config_dict.get("discount", 0.95),
        random_seed=config_dict.get("random_seed", 12345),
        lower_state_bound=config_dict.get("lower_state_bound", -30.0),
        upper_state_bound=config_dict.get("upper_state_bound", 30.0),
        max_order=config_dict.get("max_order", 10.0),
        purchase_cost=config_dict.get("purchase_cost", 20.0),
        holding_cost=config_dict.get("holding_cost", 2.0),
        backlog_cost=config_dict.get("backlog_cost", 10.0),
        disposal_cost=config_dict.get("disposal_cost", 10.0),
        lost_sale_cost=config_dict.get("lost_sale_cost", 100.0),
        demand_mean=config_dict.get("demand_mean", config_dict.get("dist_mean", 5.0)),
        demand_std=config_dict.get("demand_std", config_dict.get("dist_std", 2.0)),
        demand_min=config_dict.get("demand_min", config_dict.get("dist_min", 0.0)),
        demand_max=config_dict.get("demand_max", config_dict.get("dist_max", 10.0)),
        num_noise_samples=config_dict.get("num_noise_samples", config_dict.get("num_sample_noise", 2000)),
        action_step=config_dict.get("action_step", config_dict.get("action_discrete_param", 1.0)),
    )


def make_inventory_mdp(config: InventoryMDPConfig | dict | None = None):
    """
    Build the tutorial inventory instance and pre-sample the default noise batch.

    The default settings match the existing notebook so the examples still run
    the same way, but the parameters now live in one explicit config object.

    Args:
        config: Optional inventory configuration. If omitted, the tutorial
            defaults are used.
    """

    inventory_config = InventoryMDPConfig() if config is None else config
    mdp = SingleProductInventoryMDP(inventory_config)
    mdp.sample_fixed_noise_batch()
    return mdp
