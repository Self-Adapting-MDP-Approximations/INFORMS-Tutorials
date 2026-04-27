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


"""
-----------------------------------------------------------------------------
Global experiment controls
-----------------------------------------------------------------------------
Change values in this block to update the continuous-MDP notebooks together.
The config dataclasses below are intentionally built from these constants.
"""

SEEDS: tuple[int, ...] = (111, 222, 333, 444, 555)  # Random seeds used for repeated experiment runs.

"""
Shared sampled-ALP sizes used by the polynomial ALP example, FALP, and SGALP.
"""
NUM_CONSTRAINTS: int = 3000  # Number of sampled Bellman inequalities in sampled ALPs.
NUM_STATE_RELEVANCE_SAMPLES: int = NUM_CONSTRAINTS  # Number of states used to approximate the ALP objective.
NUM_GUIDING_STATES: int = NUM_CONSTRAINTS  # Number of states used for SGALP guiding constraints.
FEATURE_COUNTS: list = [0,1,2,3,4,5,6]  # Random-feature counts tested in FALP and SGALP.
FEATURE_BANDWIDTH_CHOICES = (1e-3, 1e-4)  # Candidate bandwidths for random Fourier features.
GRID_SIZE = 2000  # Shared grid size for policy lookup and plotting defaults.


"""
Inventory instance
"""
MDP_NAME: str = "Inventory"  # Label for the continuous inventory-control model.
DISCOUNT: float = 0.95  # Discount factor in the infinite-horizon cost objective.
MDP_RANDOM_SEED: int = 12345  # Seed for demand samples and MDP sampling routines.
LOWER_STATE_BOUND: float = -4.0  # Minimum inventory position, including backlog.
UPPER_STATE_BOUND: float = 12.0  # Maximum inventory position, including storage capacity.
MAX_ORDER: float = 6.0  # Largest feasible order quantity.
PURCHASE_COST: float = 20.0  # Per-unit ordering cost.
HOLDING_COST: float = 1.0  # Per-unit cost for positive ending inventory.
BACKLOG_COST: float = 25.0  # Per-unit cost for negative ending inventory.
DISPOSAL_COST: float = 5.0  # Per-unit cost for inventory above capacity.
LOST_SALE_COST: float = 3000.0  # Per-unit penalty for demand beyond the backlog limit.
DEMAND_MEAN: float = 6.0  # Mean of the truncated normal demand distribution.
DEMAND_STD: float = 3.5  # Standard deviation of the truncated normal demand distribution.
DEMAND_MIN: float = 0.0  # Lower truncation point for demand.
DEMAND_MAX: float = 10.0  # Upper truncation point for demand.
NUM_NOISE_SAMPLES: int = 1000  # Demand draws used to approximate expectations.
ACTION_STEP: float = .1  # Spacing of the discrete action grid.

"""
Shared policy and lower-bound evaluation
"""
INITIAL_STATE: float = 6.0  # Initial inventory state used for reported policy costs.
POLICY_STATE_GRID_SIZE: int = GRID_SIZE  # Number of grid states used to build greedy-policy lookup.
POLICY_NOISE_BATCH_SIZE: int = NUM_NOISE_SAMPLES  # Demand draws used in one-step greedy lookahead.
POLICY_NOISE_SEED: int = 123456  # Seed for greedy-policy lookahead demand samples.
NUM_POLICY_TRAJECTORIES: int = NUM_NOISE_SAMPLES  # Number of simulated trajectories for policy evaluation.
POLICY_HORIZON: int = 200  # Number of periods simulated for each policy trajectory.
POLICY_SIMULATION_SEED: int = 2026  # Seed for policy-cost simulation paths.

"""
Shared policy and lower-bound evaluation
"""
LOWER_BOUND_NUM_MC_INIT_STATES: int = 128  # Number of initial states used in lower-bound estimation.
LOWER_BOUND_CHAIN_LENGTH: int = 2000  # Markov-chain length for lower-bound sampling.
LOWER_BOUND_BURN_IN: int = 500  # Initial chain samples discarded before estimation.
LOWER_BOUND_PROPOSAL_STATE_STD: float = 2  # State proposal standard deviation for Metropolis sampling.
LOWER_BOUND_PROPOSAL_ACTION_STD: float = 2  # Action proposal standard deviation for Metropolis sampling.
LOWER_BOUND_RANDOM_SEED: int = 654321  # Seed for lower-bound sampling.
LOWER_BOUND_NOISE_BATCH_SIZE: int = 500  # Demand draws used inside lower-bound violation estimates.
LOWER_BOUND_SAMPLER: str = "metropolis"  # Lower-bound sampler backend.
LOWER_BOUND_NUM_WALKERS: int = 10  # Number of parallel chains or walkers in lower-bound sampling.

"""
Polynomial ALP example in how-code-works.ipynb.
"""
POLYNOMIAL_EXPONENTS: tuple[int, ...] = (0, 1, 2)  # Polynomial powers used in the hand-built ALP example.
POLYNOMIAL_ALP_POLICY_GRID_SIZE: int = GRID_SIZE  # Greedy-policy lookup grid for the polynomial ALP example.
POLYNOMIAL_ALP_PROBE_STATES: tuple[float, ...] = (-4.0, 0.0, 4.0, 8.0, 12.0)  # States shown in policy summaries.

"""
Random-feature ALP controls.
"""
RANDOM_FEATURE_BANDWIDTH_CHOICES: tuple[float, ...] = FEATURE_BANDWIDTH_CHOICES  # FALP random-feature bandwidths.
RANDOM_FEATURE_SEED: int = 111  # Seed for random Fourier feature generation.
FALP_FEATURE_COUNTS: tuple[int, ...] = tuple(FEATURE_COUNTS)  # Feature counts used by FALP experiments.
SGALP_FEATURE_COUNTS: tuple[int, ...] = tuple(FEATURE_COUNTS)  # Feature counts used by SGALP experiments.


"""
SGALP-specific controls.
"""
SGALP_BANDWIDTH_CHOICES: tuple[float, ...] = FEATURE_BANDWIDTH_CHOICES  # SGALP random-feature bandwidths.
SGALP_BATCH_SIZE: int = 1  # Number of new random features added per SGALP stage.
GUIDING_VIOLATION: float = 0.0  # Allowed violation in guiding constraints.
GUIDING_RELAX_FRACTION: float = 0.02  # Relative relaxation used if guiding constraints are too tight.
GUIDING_ABS_FLOOR: float = 1e-6  # Minimum absolute relaxation for guiding constraints.
GUIDING_RETRY_SCALES: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)  # Relaxation multipliers tried after infeasibility.

"""
PSMD has no ALP constraint count. These are its analogous sampling controls.
"""
PSMD_NUM_ITERATIONS = 1500  # Total stochastic-gradient iterations.
PSMD_NUM_SAMPLER_PARTICLES = 30  # Number of state-action particles maintained by the sampler.
PSMD_NUM_NOISE_SAMPLES_PER_ITERATION = 50  # Demand draws used in each PSMD gradient estimate.
PSMD_EVAL_INTERVAL = 300  # Iteration spacing for bound and policy evaluations.
PSMD_STEP_SIZE = 0.2  # Initial scale of the stochastic-gradient step size.
PSMD_STEP_SIZE_POWER = 0.5  # Decay exponent for the step-size schedule.
PSMD_SAMPLER_STEPS = 20  # Metropolis moves per PSMD iteration.
PSMD_PROPOSAL_STATE_STD = 0.8  # State proposal standard deviation in the PSMD sampler.
PSMD_PROPOSAL_ACTION_STD = 0.8  # Action proposal standard deviation in the PSMD sampler.
PSMD_SAMPLING_TEMPERATURE = 25.0  # Smoothness of the violation-focused sampling distribution.
PSMD_REFRESH_FRACTION = 0.1  # Fraction of particles randomly refreshed each iteration.
PSMD_COEFFICIENT_CLIP = 5000.0  # Absolute cap used to stabilize VFA coefficients.
PSMD_RANDOM_SEED = 777  # Seed for PSMD randomness.
PSMD_SNAPSHOT_ITERATIONS = (0,300,600,900,1200,1500)  # Iterations saved for sampler plots.
PSMD_SNAPSHOT_SAMPLE_SIZE = 100  # Number of state-action points shown in each snapshot.
PSMD_SNAPSHOT_SAMPLER_STEPS = None  # Extra sampler steps for snapshots; None reuses the PSMD setting.
PSMD_SNAPSHOT_REFRESH_FRACTION = 0.0  # Particle refresh rate used during snapshot collection.



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

    bandwidth_choices: tuple[float, ...] = RANDOM_FEATURE_BANDWIDTH_CHOICES
    random_seed: int = RANDOM_FEATURE_SEED

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

    num_guiding_states: int = NUM_GUIDING_STATES
    allowed_violation: float = GUIDING_VIOLATION
    relax_fraction: float = GUIDING_RELAX_FRACTION
    absolute_floor: float = GUIDING_ABS_FLOOR
    retry_scales: tuple[float, ...] = GUIDING_RETRY_SCALES

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
    num_constraints: int = NUM_CONSTRAINTS
    num_state_relevance_samples: int = NUM_STATE_RELEVANCE_SAMPLES
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
    batch_size: int = SGALP_BATCH_SIZE
    num_constraints: int = NUM_CONSTRAINTS
    num_state_relevance_samples: int = NUM_STATE_RELEVANCE_SAMPLES
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

    num_mc_init_states: int = LOWER_BOUND_NUM_MC_INIT_STATES
    chain_length: int = LOWER_BOUND_CHAIN_LENGTH
    burn_in: int = LOWER_BOUND_BURN_IN
    proposal_state_std: float = LOWER_BOUND_PROPOSAL_STATE_STD
    proposal_action_std: float = LOWER_BOUND_PROPOSAL_ACTION_STD
    random_seed: int = LOWER_BOUND_RANDOM_SEED
    noise_batch_size: int = LOWER_BOUND_NOISE_BATCH_SIZE
    sampler: str = LOWER_BOUND_SAMPLER
    num_walkers: int = LOWER_BOUND_NUM_WALKERS
    initial_state: float = INITIAL_STATE

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

    state_grid_size: int = POLICY_STATE_GRID_SIZE
    policy_noise_batch_size: int = POLICY_NOISE_BATCH_SIZE
    policy_noise_seed: int = POLICY_NOISE_SEED
    num_trajectories: int = NUM_POLICY_TRAJECTORIES
    horizon: int = POLICY_HORIZON
    simulation_seed: int = POLICY_SIMULATION_SEED
    initial_state: float = INITIAL_STATE


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

    num_iterations: int = PSMD_NUM_ITERATIONS
    H: int = PSMD_NUM_SAMPLER_PARTICLES
    N: int = PSMD_NUM_NOISE_SAMPLES_PER_ITERATION
    eval_interval: int = PSMD_EVAL_INTERVAL
    step_size: float = PSMD_STEP_SIZE
    step_size_power: float = PSMD_STEP_SIZE_POWER
    sampler_steps: int = PSMD_SAMPLER_STEPS
    proposal_state_std: float = PSMD_PROPOSAL_STATE_STD
    proposal_action_std: float = PSMD_PROPOSAL_ACTION_STD
    sampling_temperature: float = PSMD_SAMPLING_TEMPERATURE
    refresh_fraction: float = PSMD_REFRESH_FRACTION
    coefficient_clip: float = PSMD_COEFFICIENT_CLIP
    random_seed: int = PSMD_RANDOM_SEED
    initial_state: float = INITIAL_STATE
    snapshot_iterations: tuple[int, ...] = PSMD_SNAPSHOT_ITERATIONS
    snapshot_sample_size: int = PSMD_SNAPSHOT_SAMPLE_SIZE
    snapshot_sampler_steps: int | None = PSMD_SNAPSHOT_SAMPLER_STEPS
    snapshot_refresh_fraction: float = PSMD_SNAPSHOT_REFRESH_FRACTION
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

    mdp_name: str = MDP_NAME
    discount: float = DISCOUNT
    random_seed: int = MDP_RANDOM_SEED
    lower_state_bound: float = LOWER_STATE_BOUND
    upper_state_bound: float = UPPER_STATE_BOUND
    max_order: float = MAX_ORDER
    purchase_cost: float = PURCHASE_COST
    holding_cost: float = HOLDING_COST
    backlog_cost: float = BACKLOG_COST
    disposal_cost: float = DISPOSAL_COST
    lost_sale_cost: float = LOST_SALE_COST
    demand_mean: float = DEMAND_MEAN
    demand_std: float = DEMAND_STD
    demand_min: float = DEMAND_MIN
    demand_max: float = DEMAND_MAX
    num_noise_samples: int = NUM_NOISE_SAMPLES
    action_step: float = ACTION_STEP


@dataclass(frozen=True)
class PolynomialALPExampleConfig(ConfigMixin):
    """
    Shared settings for the hand-built polynomial sampled-ALP example.

    These values are used by `how-code-works.ipynb`. The inventory-model and
    policy-evaluation parameters come from `ContinuousMDPNotebookConfig` so the
    example stays aligned with the FALP, SGALP, and PSMD notebooks.
    """

    seeds: tuple[int, ...] = SEEDS
    polynomial_exponents: tuple[int, ...] = POLYNOMIAL_EXPONENTS
    num_constraints: int = NUM_CONSTRAINTS
    num_state_relevance_samples: int = NUM_STATE_RELEVANCE_SAMPLES
    policy_grid_size: int = POLYNOMIAL_ALP_POLICY_GRID_SIZE
    probe_states: tuple[float, ...] = POLYNOMIAL_ALP_PROBE_STATES

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "polynomial_exponents", tuple(int(power) for power in self.polynomial_exponents))
        object.__setattr__(self, "probe_states", tuple(float(state) for state in self.probe_states))


def _tutorial_lower_bound_config():
    return LowerBoundConfig(
        num_mc_init_states=LOWER_BOUND_NUM_MC_INIT_STATES,
        chain_length=LOWER_BOUND_CHAIN_LENGTH,
        burn_in=LOWER_BOUND_BURN_IN,
        proposal_state_std=LOWER_BOUND_PROPOSAL_STATE_STD,
        proposal_action_std=LOWER_BOUND_PROPOSAL_ACTION_STD,
        random_seed=LOWER_BOUND_RANDOM_SEED,
        noise_batch_size=LOWER_BOUND_NOISE_BATCH_SIZE,
        sampler=LOWER_BOUND_SAMPLER,
        num_walkers=LOWER_BOUND_NUM_WALKERS,
        initial_state=INITIAL_STATE,
    )


def _tutorial_policy_config():
    return PolicyEvaluationConfig(
        state_grid_size=POLICY_STATE_GRID_SIZE,
        policy_noise_batch_size=POLICY_NOISE_BATCH_SIZE,
        policy_noise_seed=POLICY_NOISE_SEED,
        num_trajectories=NUM_POLICY_TRAJECTORIES,
        horizon=POLICY_HORIZON,
        simulation_seed=POLICY_SIMULATION_SEED,
        initial_state=INITIAL_STATE,
    )


def _tutorial_falp_config():
    return FALPConfig(
        num_random_features=0,
        num_constraints=NUM_CONSTRAINTS,
        num_state_relevance_samples=NUM_STATE_RELEVANCE_SAMPLES,
        random_features=RandomFeatureConfig(
            bandwidth_choices=RANDOM_FEATURE_BANDWIDTH_CHOICES,
            random_seed=RANDOM_FEATURE_SEED,
        ),
        solver="auto",
    )


def _tutorial_sgalp_config():
    return SGALPConfig(
        max_random_features=0,
        batch_size=SGALP_BATCH_SIZE,
        num_constraints=NUM_CONSTRAINTS,
        num_state_relevance_samples=NUM_STATE_RELEVANCE_SAMPLES,
        random_features=RandomFeatureConfig(
            bandwidth_choices=SGALP_BANDWIDTH_CHOICES,
            random_seed=RANDOM_FEATURE_SEED,
        ),
        guiding=GuidingConstraintConfig(
            num_guiding_states=NUM_GUIDING_STATES,
            allowed_violation=GUIDING_VIOLATION,
            relax_fraction=GUIDING_RELAX_FRACTION,
            absolute_floor=GUIDING_ABS_FLOOR,
            retry_scales=GUIDING_RETRY_SCALES,
        ),
    )


def _tutorial_psmd_config():
    lower_bound = _tutorial_lower_bound_config()
    policy_evaluation = _tutorial_policy_config()
    return PSMDConfig(
        num_iterations=PSMD_NUM_ITERATIONS,
        H=PSMD_NUM_SAMPLER_PARTICLES,
        N=PSMD_NUM_NOISE_SAMPLES_PER_ITERATION,
        eval_interval=PSMD_EVAL_INTERVAL,
        step_size=PSMD_STEP_SIZE,
        step_size_power=PSMD_STEP_SIZE_POWER,
        sampler_steps=PSMD_SAMPLER_STEPS,
        proposal_state_std=PSMD_PROPOSAL_STATE_STD,
        proposal_action_std=PSMD_PROPOSAL_ACTION_STD,
        sampling_temperature=PSMD_SAMPLING_TEMPERATURE,
        refresh_fraction=PSMD_REFRESH_FRACTION,
        coefficient_clip=PSMD_COEFFICIENT_CLIP,
        random_seed=PSMD_RANDOM_SEED,
        initial_state=policy_evaluation.initial_state,
        snapshot_iterations=PSMD_SNAPSHOT_ITERATIONS,
        snapshot_sample_size=PSMD_SNAPSHOT_SAMPLE_SIZE,
        snapshot_sampler_steps=PSMD_SNAPSHOT_SAMPLER_STEPS,
        snapshot_refresh_fraction=PSMD_SNAPSHOT_REFRESH_FRACTION,
        lower_bound=lower_bound,
        policy_evaluation=policy_evaluation,
    )


@dataclass(frozen=True)
class ContinuousMDPNotebookConfig(ConfigMixin):
    """
    One shared configuration object for the continuous-MDP notebooks.

    `psmd.ipynb`, `self-guided-alp.ipynb`, and `how-code-works.ipynb` should
    import this object instead of redefining experiment parameters locally.
    Changing the values here changes all three notebooks together.
    """

    seeds: tuple[int, ...] = SEEDS
    inventory: InventoryMDPConfig = field(default_factory=InventoryMDPConfig)
    lower_bound: LowerBoundConfig = field(default_factory=_tutorial_lower_bound_config)
    policy_evaluation: PolicyEvaluationConfig = field(default_factory=_tutorial_policy_config)
    polynomial_alp: PolynomialALPExampleConfig = field(default_factory=PolynomialALPExampleConfig)
    falp_feature_counts: tuple[int, ...] = field(default_factory=lambda: tuple(FALP_FEATURE_COUNTS))
    sgalp_feature_counts: tuple[int, ...] = field(default_factory=lambda: tuple(SGALP_FEATURE_COUNTS))
    falp: FALPConfig = field(default_factory=_tutorial_falp_config)
    sgalp: SGALPConfig = field(default_factory=_tutorial_sgalp_config)
    psmd: PSMDConfig = field(default_factory=_tutorial_psmd_config)

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "falp_feature_counts", tuple(int(count) for count in self.falp_feature_counts))
        object.__setattr__(self, "sgalp_feature_counts", tuple(int(count) for count in self.sgalp_feature_counts))


CONTINUOUS_MDP_NOTEBOOK_CONFIG = ContinuousMDPNotebookConfig()


def make_shared_evaluation_configs(
    initial_state: float | None = None,
    lower_bound_sampler: str | None = None,
    notebook_config: ContinuousMDPNotebookConfig | None = None,
):
    """
    Build lower-bound and policy configs shared across the continuous notebooks.

    Args:
        initial_state: Optional override for both lower-bound and policy-cost
            evaluation. If omitted, the value in `CONTINUOUS_MDP_NOTEBOOK_CONFIG`
            is used.
        lower_bound_sampler: Optional lower-bound sampler override.
        notebook_config: Optional complete notebook config to read from.
    """

    notebook_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG if notebook_config is None else notebook_config
    lower_bound_config = notebook_config.lower_bound
    policy_config = notebook_config.policy_evaluation

    if initial_state is not None:
        lower_bound_config = lower_bound_config.with_updates(initial_state=initial_state)
        policy_config = policy_config.with_updates(initial_state=initial_state)

    if lower_bound_sampler is not None:
        lower_bound_config = lower_bound_config.with_updates(sampler=lower_bound_sampler)

    return lower_bound_config, policy_config
