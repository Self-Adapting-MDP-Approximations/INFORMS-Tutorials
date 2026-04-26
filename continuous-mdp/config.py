# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Centralized parameter bundles for the tutorial code.

The project originally exposed many scalar keyword arguments directly on
constructors and helper functions. That is convenient for quick experiments,
but it becomes harder to read once we want to explain what each parameter is
doing. The dataclasses in this file group related parameters together so a
reader can inspect one object and understand the modeling choices being made.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace


class ConfigMixin:
    """
    Small helper that makes config updates read naturally in notebooks.
    """

    def with_updates(self, **kwargs):
        """
        Return a copy of the config with selected fields replaced.

        Args:
            **kwargs: Field names and replacement values.
        """
        return replace(self, **kwargs)


@dataclass(frozen=True)
class RandomFeatureConfig(ConfigMixin):
    """
    Parameters controlling the random Fourier feature family.

    Attributes:
        bandwidth_choices: Candidate bandwidth values used when sampling
            random Fourier frequencies.
        random_seed: Seed controlling the sampled feature sequence.
    """

    bandwidth_choices: tuple[float, ...] = (1e-3, 1e-4)
    random_seed: int = 111

    def __post_init__(self):
        object.__setattr__(self, "bandwidth_choices", tuple(self.bandwidth_choices))


@dataclass(frozen=True)
class HiGHSSolverConfig(ConfigMixin):
    """
    Numerical settings for SciPy's HiGHS linear-program solver.

    Attributes:
        method: HiGHS backend name passed to SciPy.
        primal_feasibility_tolerance: Solver tolerance for primal feasibility.
        dual_feasibility_tolerance: Solver tolerance for dual feasibility.
    """

    method: str = "highs-ds"
    primal_feasibility_tolerance: float = 1e-7
    dual_feasibility_tolerance: float = 1e-7


@dataclass(frozen=True)
class GuidingConstraintConfig(ConfigMixin):
    """
    Parameters for SGALP guiding constraints.

    Attributes:
        num_guiding_states: Number of sampled states used to build guiding
            inequalities.
        allowed_violation: Additive violation allowance in the guiding rules.
        relax_fraction: Relative violation allowance as a fraction of the
            previous value estimate.
        absolute_floor: Minimum positive allowance used for numerical safety.
        retry_scales: Multipliers tried if the guiding LP is infeasible.
    """

    num_guiding_states: int = 100
    allowed_violation: float = 0.0
    relax_fraction: float = 0.02
    absolute_floor: float = 1e-6
    retry_scales: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

    def __post_init__(self):
        object.__setattr__(self, "retry_scales", tuple(self.retry_scales))


@dataclass(frozen=True)
class FALPConfig(ConfigMixin):
    """
    Core FALP settings.

    Attributes:
        num_random_features: Number of nonconstant random features.
        num_constraints: Number of sampled Bellman inequalities.
        num_state_relevance_samples: Number of states used in the ALP objective.
        random_features: Shared random-feature sampling settings.
        solver: LP solver choice, typically `auto` or `scipy`.
    """

    num_random_features: int = 1
    num_constraints: int = 40
    num_state_relevance_samples: int = 200
    random_features: RandomFeatureConfig = field(default_factory=RandomFeatureConfig)
    solver: str = "auto"


@dataclass(frozen=True)
class SGALPConfig(ConfigMixin):
    """
    Core SGALP settings.

    Attributes:
        max_random_features: Largest nonconstant basis size solved in the
            stage sequence.
        batch_size: Number of new random features added per stage.
        num_constraints: Number of sampled Bellman inequalities per stage.
        num_state_relevance_samples: Number of states used in the ALP objective.
        random_features: Shared random-feature sampling settings.
        guiding: Parameters controlling SGALP guiding inequalities.
        solver: HiGHS solver settings for each SGALP stage LP.
    """

    max_random_features: int = 10
    batch_size: int = 1
    num_constraints: int = 40
    num_state_relevance_samples: int = 200
    random_features: RandomFeatureConfig = field(default_factory=RandomFeatureConfig)
    guiding: GuidingConstraintConfig = field(default_factory=GuidingConstraintConfig)
    solver: HiGHSSolverConfig = field(default_factory=HiGHSSolverConfig)


@dataclass(frozen=True)
class LowerBoundConfig(ConfigMixin):
    """
    Parameters for the sampling-based CVL / LNS lower-bound estimator.

    Attributes:
        num_mc_init_states: Number of initial state-action particles or walkers.
        chain_length: Total number of MCMC steps per chain.
        burn_in: Number of initial MCMC steps discarded from each chain.
        proposal_state_std: Proposal standard deviation for state updates.
        proposal_action_std: Proposal standard deviation for action updates.
        random_seed: Base seed controlling the sampler.
        noise_batch_size: Number of demand samples reused in each Bellman
            residual estimate.
        sampler: Sampling backend, typically `auto`, `metropolis`, or `emcee`.
        num_walkers: Number of walkers for the optional `emcee` sampler.
        initial_state: Initial inventory level at which the lower bound is
            reported.
    """

    num_mc_init_states: int = 64
    chain_length: int = 800
    burn_in: int = 400
    proposal_state_std: float = 0.8
    proposal_action_std: float = 0.8
    random_seed: int = 333
    noise_batch_size: int = 1000
    sampler: str = "auto"
    num_walkers: int = 32
    initial_state: float = 5.0

    def to_kwargs(self):
        """
        Convert the config into keyword arguments for the lower-bound helpers.
        """
        return {
            "num_mc_init_states": self.num_mc_init_states,
            "chain_length": self.chain_length,
            "burn_in": self.burn_in,
            "proposal_state_std": self.proposal_state_std,
            "proposal_action_std": self.proposal_action_std,
            "random_seed": self.random_seed,
            "noise_batch_size": self.noise_batch_size,
            "sampler": self.sampler,
            "num_walkers": self.num_walkers,
            "initial_state": self.initial_state,
        }


@dataclass(frozen=True)
class PolicyEvaluationConfig(ConfigMixin):
    """
    Parameters for approximate greedy-policy construction and simulation.

    Attributes:
        state_grid_size: Number of states in the greedy-policy lookup grid.
        policy_noise_batch_size: Number of demand samples used when comparing
            actions in one-step lookahead.
        policy_noise_seed: Seed controlling the one-step lookahead noise batch.
        num_trajectories: Number of Monte Carlo trajectories used for policy
            evaluation.
        horizon: Number of time periods simulated in each trajectory.
        simulation_seed: Base seed controlling simulated demand paths.
        initial_state: Initial inventory level used in policy simulation.
    """

    state_grid_size: int = 801
    policy_noise_batch_size: int = 128
    policy_noise_seed: int = 123456
    num_trajectories: int = 2000
    horizon: int = 1000
    simulation_seed: int = 2026
    initial_state: float = 5.0


@dataclass(frozen=True)
class PSMDConfig(ConfigMixin):
    """
    Parameters for the lightweight PSMD tutorial implementation.

    Attributes:
        num_iterations: Number of projected-gradient iterations.
        H: Legacy shorthand for the number of sampled state-action particles.
        N: Legacy shorthand for the number of demand samples per iteration.
        eval_interval: Number of iterations between diagnostic evaluations.
        step_size: Initial projected-gradient step size.
        step_size_power: Exponent controlling the diminishing step schedule.
        sampler_steps: Number of Metropolis updates between evaluations.
        proposal_state_std: Proposal standard deviation for state updates.
        proposal_action_std: Proposal standard deviation for action updates.
        sampling_temperature: Temperature controlling how strongly the sampler
            favors highly violated constraints.
        refresh_fraction: Fraction of particles redrawn uniformly each refresh.
        coefficient_clip: L2-norm cap for projected coefficients.
        random_seed: Base seed for all PSMD randomness.
        initial_state: Initial inventory level used in bound reporting.
        snapshot_iterations: Iterations at which the sampler cloud is stored.
        snapshot_sample_size: Number of particles stored in each snapshot.
        snapshot_sampler_steps: Extra Metropolis steps used for snapshot clouds.
        snapshot_refresh_fraction: Refresh fraction used for snapshot clouds.
        lower_bound: Settings for lower-bound estimation on averaged iterates.
        policy_evaluation: Settings for policy-cost simulation on averaged iterates.
    """

    num_iterations: int = 1000
    H: int = 10
    N: int = 50
    eval_interval: int = 50
    step_size: float = 0.2
    step_size_power: float = 0.5
    sampler_steps: int = 20
    proposal_state_std: float = 0.8
    proposal_action_std: float = 0.8
    sampling_temperature: float = 25.0
    refresh_fraction: float = 0.1
    coefficient_clip: float = 500.0
    random_seed: int = 777
    initial_state: float = 5.0
    snapshot_iterations: tuple[int, ...] = (0, 10, 20, 50, 100, 200, 300, 500, 800, 1000)
    snapshot_sample_size: int = 100
    snapshot_sampler_steps: int | None = None
    snapshot_refresh_fraction: float = 0.0
    lower_bound: LowerBoundConfig = field(default_factory=LowerBoundConfig)
    policy_evaluation: PolicyEvaluationConfig = field(default_factory=PolicyEvaluationConfig)

    def __post_init__(self):
        object.__setattr__(self, "snapshot_iterations", tuple(int(x) for x in self.snapshot_iterations))

    @property
    def num_sampler_particles(self) -> int:
        """
        Descriptive alias for the legacy `H` parameter.
        """
        return self.H

    @property
    def num_noise_samples_per_iteration(self) -> int:
        """
        Descriptive alias for the legacy `N` parameter.
        """
        return self.N


@dataclass(frozen=True)
class InventoryMDPConfig(ConfigMixin):
    """
    Concrete inventory-model parameters used in the tutorial notebook.

    Attributes:
        mdp_name: Human-readable name for the model instance.
        discount: Discount factor used in Bellman equations.
        random_seed: Base seed for reproducible sampling.
        lower_state_bound: Lowest feasible inventory level.
        upper_state_bound: Highest feasible inventory level.
        max_order: Largest feasible order quantity.
        purchase_cost: Per-unit procurement cost.
        holding_cost: Per-unit holding cost for positive inventory.
        backlog_cost: Per-unit backlog cost for negative inventory.
        disposal_cost: Per-unit disposal cost beyond the storage limit.
        lost_sale_cost: Per-unit penalty for demand beyond the backlog limit.
        demand_mean: Mean of the truncated normal demand distribution.
        demand_std: Standard deviation of the demand distribution.
        demand_min: Minimum feasible demand draw.
        demand_max: Maximum feasible demand draw.
        num_noise_samples: Default Monte Carlo batch size for expectations.
        action_step: Step size in the discrete order-quantity grid.
    """

    mdp_name: str = "Inventory"
    discount: float = 0.95
    random_seed: int = 12345
    lower_state_bound: float = -10.0
    upper_state_bound: float = 10.0
    max_order: float = 10.0
    purchase_cost: float = 20.0
    holding_cost: float = 2.0
    backlog_cost: float = 10.0
    disposal_cost: float = 10.0
    lost_sale_cost: float = 100.0
    demand_mean: float = 5.0
    demand_std: float = 2.0
    demand_min: float = 0.0
    demand_max: float = 10.0
    num_noise_samples: int = 2000
    action_step: float = 1.0
