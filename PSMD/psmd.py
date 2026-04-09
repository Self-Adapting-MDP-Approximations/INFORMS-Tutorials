# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Lightweight PSMD implementation for the tutorial inventory example.

This version is intentionally simpler than the research notebook the user
provided:
- it uses a fixed polynomial basis [1, s, s^2]
- it samples state-action pairs with random-walk Metropolis-Hastings
- it performs plain projected gradient updates on the PSMD surrogate

The goal is to give the tutorial project a compact PSMD baseline that fits the
same inventory MDP and reporting workflow as the existing FALP / SGALP code.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from basis import PolynomialBasis1D
from policy import estimate_upper_bound_fast
from Self_Guided_ALP.cvl_lower_bound import SimpleLNSLowerBound
from config import PSMDConfig


@dataclass
class _PSMDModelSnapshot:
    """
    Minimal fitted-model view used during bound evaluation.
    """

    mdp: object
    basis: object
    coef: np.ndarray
    num_random_features: int


class PSMD:
    """
    Small PSMD-style solver for the tutorial inventory MDP.

    The implementation follows the same overall loop as the original notebook:
    1. sample state-action pairs from a distribution that favors Bellman
       constraint violation
    2. estimate the gradient of the PSMD surrogate on those samples
    3. update the value-function coefficients with a diminishing step size
    4. track lower/upper-bound diagnostics using the averaged iterate
    """

    def __init__(
        self,
        mdp,
        config: PSMDConfig | None = None,
        num_iterations=1000,
        H=10,
        N=50,
        eval_interval=50,
        step_size=0.2,
        step_size_power=0.5,
        sampler_steps=20,
        proposal_state_std=0.8,
        proposal_action_std=0.8,
        sampling_temperature=25.0,
        refresh_fraction=0.1,
        coefficient_clip=500.0,
        random_seed=777,
        initial_state=5.0,
    ):
        """
        Build the lightweight PSMD tutorial solver.

        Args:
            mdp: Underlying inventory MDP.
            config: Optional grouped PSMD settings.
            num_iterations: Number of projected-gradient iterations.
            H: Legacy name for the number of sampler particles.
            N: Legacy name for the number of demand samples per iteration.
            eval_interval: Number of iterations between diagnostic evaluations.
            step_size: Initial projected-gradient step size.
            step_size_power: Power used in the diminishing step schedule.
            sampler_steps: Number of Metropolis updates per refresh.
            proposal_state_std: Proposal standard deviation for state moves.
            proposal_action_std: Proposal standard deviation for action moves.
            sampling_temperature: Temperature of the sampler density.
            refresh_fraction: Fraction of particles redrawn uniformly.
            coefficient_clip: L2-norm cap used in projection.
            random_seed: Base seed for reproducible sampling.
            initial_state: Initial inventory level used in diagnostics.
        """
        self.mdp = mdp
        self.config = self._build_config(
            config=config,
            num_iterations=num_iterations,
            H=H,
            N=N,
            eval_interval=eval_interval,
            step_size=step_size,
            step_size_power=step_size_power,
            sampler_steps=sampler_steps,
            proposal_state_std=proposal_state_std,
            proposal_action_std=proposal_action_std,
            sampling_temperature=sampling_temperature,
            refresh_fraction=refresh_fraction,
            coefficient_clip=coefficient_clip,
            random_seed=random_seed,
            initial_state=initial_state,
        )

        self.basis = PolynomialBasis1D(exponents=(0, 1, 2))
        self.num_random_features = self.basis.max_random_features
        self.coef = np.zeros(self.num_random_features + 1, dtype=float)
        self.avg_coef = np.zeros_like(self.coef)
        self.iteration = 0

        self.rng = np.random.RandomState(self.config.random_seed)
        self.sampled_states, self.sampled_actions = self._draw_uniform_state_actions(self.config.num_sampler_particles)
        self.snapshot_states, self.snapshot_actions = self._draw_uniform_state_actions(self.config.snapshot_sample_size)
        self.last_acceptance_rate = 0.0

        self.history = []
        self.state_action_snapshots = {}
        self.solution = None

    @staticmethod
    def _build_config(
        config,
        num_iterations,
        H,
        N,
        eval_interval,
        step_size,
        step_size_power,
        sampler_steps,
        proposal_state_std,
        proposal_action_std,
        sampling_temperature,
        refresh_fraction,
        coefficient_clip,
        random_seed,
        initial_state,
    ):
        """
        Merge legacy scalar arguments into a single `PSMDConfig`.

        Args:
            config: Optional already-built `PSMDConfig`.
            num_iterations: Number of projected-gradient iterations.
            H: Legacy name for the number of sampler particles.
            N: Legacy name for the number of demand samples per iteration.
            eval_interval: Number of iterations between diagnostic evaluations.
            step_size: Initial projected-gradient step size.
            step_size_power: Power used in the diminishing step schedule.
            sampler_steps: Number of Metropolis updates per refresh.
            proposal_state_std: Proposal standard deviation for state moves.
            proposal_action_std: Proposal standard deviation for action moves.
            sampling_temperature: Temperature of the sampler density.
            refresh_fraction: Fraction of particles redrawn uniformly.
            coefficient_clip: L2-norm cap used in projection.
            random_seed: Base seed for reproducible sampling.
            initial_state: Initial inventory level used in diagnostics.
        """
        if config is not None:
            return config
        return PSMDConfig(
            num_iterations=num_iterations,
            H=H,
            N=N,
            eval_interval=eval_interval,
            step_size=step_size,
            step_size_power=step_size_power,
            sampler_steps=sampler_steps,
            proposal_state_std=proposal_state_std,
            proposal_action_std=proposal_action_std,
            sampling_temperature=sampling_temperature,
            refresh_fraction=refresh_fraction,
            coefficient_clip=coefficient_clip,
            random_seed=random_seed,
            initial_state=initial_state,
        )

    def _draw_uniform_state_actions(self, num_samples):
        """
        Draw state-action pairs uniformly from the feasible rectangle.

        Args:
            num_samples: Number of state-action pairs to draw.
        """
        states = self.rng.uniform(self.mdp.lower_state_bound, self.mdp.upper_state_bound, size=num_samples)
        actions = self.rng.uniform(0.0, self.mdp.max_order, size=num_samples)
        return states.astype(float), actions.astype(float)

    def _constraint_statistics(self, states, actions, noise_batch, coef):
        """
        Compute Bellman-residual ingredients for PSMD on a batch of pairs.

        Args:
            states: Collection of inventory states.
            actions: Collection of order quantities.
            noise_batch: Demand samples used in the Bellman expectations.
            coef: Coefficients defining the current value approximation.
        """
        states = np.asarray(states, dtype=float).reshape(-1)
        actions = np.asarray(actions, dtype=float).reshape(-1)
        transition_summary = self.mdp.evaluate_state_action_batch(states, actions, noise_batch=noise_batch)
        next_states = transition_summary["next_states"]
        expected_cost = transition_summary["expected_cost"]

        phi_state = self.basis.eval_basis_batch(states, num_random_features=self.num_random_features)
        expected_phi_next = self.basis.eval_basis_batch(
            next_states.reshape(-1),
            num_random_features=self.num_random_features,
        ).reshape(len(states), len(noise_batch), -1).mean(axis=1)

        vfa_state = phi_state @ coef
        expected_vfa_next = expected_phi_next @ coef
        bellman_residual = expected_cost + self.mdp.discount * expected_vfa_next - vfa_state
        constraint_violation = -bellman_residual

        reference_basis_vector = self.basis.eval_basis(
            np.asarray([0.0], dtype=float),
            num_random_features=self.num_random_features,
        )
        psmd_value = (bellman_residual / (1.0 - self.mdp.discount)) + float(reference_basis_vector @ coef)

        return {
            "expected_cost": expected_cost,
            "phi_state": phi_state,
            "expected_phi_next": expected_phi_next,
            "bellman_residual": bellman_residual,
            "constraint_violation": constraint_violation,
            "psmd_value": psmd_value,
            "reference_basis_vector": reference_basis_vector,
        }

    def _log_violation_density(self, states, actions, noise_batch):
        """
        Evaluate the log density that focuses the sampler on violated regions.

        Args:
            states: Collection of inventory states.
            actions: Collection of order quantities.
            noise_batch: Demand samples used in the Bellman expectations.
        """
        stats = self._constraint_statistics(states, actions, noise_batch=noise_batch, coef=self.coef)
        return -stats["psmd_value"] / self.config.sampling_temperature

    def _metropolis_update_pairs(self, states, actions, noise_batch, sampler_steps, refresh_fraction):
        """
        Run a vectorized Metropolis update on the particle cloud.

        Args:
            states: Current particle states.
            actions: Current particle actions.
            noise_batch: Demand samples used in the Bellman expectations.
            sampler_steps: Number of Metropolis proposals to apply.
            refresh_fraction: Fraction of particles redrawn uniformly.
        """
        states = np.asarray(states, dtype=float).copy()
        actions = np.asarray(actions, dtype=float).copy()
        num_points = len(states)

        num_refresh = int(np.floor(refresh_fraction * num_points))
        if num_refresh > 0:
            refresh_index = self.rng.choice(num_points, size=num_refresh, replace=False)
            new_states, new_actions = self._draw_uniform_state_actions(num_refresh)
            states[refresh_index] = new_states
            actions[refresh_index] = new_actions

        cur_log_density = self._log_violation_density(states, actions, noise_batch=noise_batch)
        accepted = 0

        for _ in range(sampler_steps):
            proposal_states = np.clip(
                states + self.rng.normal(scale=self.config.proposal_state_std, size=num_points),
                self.mdp.lower_state_bound,
                self.mdp.upper_state_bound,
            )
            proposal_actions = np.clip(
                actions + self.rng.normal(scale=self.config.proposal_action_std, size=num_points),
                0.0,
                self.mdp.max_order,
            )

            prop_log_density = self._log_violation_density(proposal_states, proposal_actions, noise_batch=noise_batch)
            log_u = np.log(self.rng.uniform(size=num_points))
            accept = log_u < (prop_log_density - cur_log_density)

            states[accept] = proposal_states[accept]
            actions[accept] = proposal_actions[accept]
            cur_log_density[accept] = prop_log_density[accept]
            accepted += int(np.sum(accept))

        total_moves = max(num_points * sampler_steps, 1)
        return states, actions, accepted / total_moves

    def _sample_violation_pairs(self, noise_batch):
        """
        Refresh the sampler cloud used to estimate PSMD gradients.

        Args:
            noise_batch: Demand samples used in the Bellman expectations.
        """
        self.sampled_states, self.sampled_actions, self.last_acceptance_rate = self._metropolis_update_pairs(
            states=self.sampled_states,
            actions=self.sampled_actions,
            noise_batch=noise_batch,
            sampler_steps=self.config.sampler_steps,
            refresh_fraction=self.config.refresh_fraction,
        )
        return self.sampled_states.copy(), self.sampled_actions.copy()

    def _compute_gradient(self, states, actions, noise_batch):
        """
        Estimate the PSMD gradient on a sampled batch of state-action pairs.

        Args:
            states: Collection of inventory states.
            actions: Collection of order quantities.
            noise_batch: Demand samples used in the Bellman expectations.
        """
        stats = self._constraint_statistics(states, actions, noise_batch=noise_batch, coef=self.coef)
        return np.mean(
            (self.mdp.discount * stats["expected_phi_next"] - stats["phi_state"]) / (1.0 - self.mdp.discount)
            + stats["reference_basis_vector"],
            axis=0,
        )

    def _update_snapshot_particles(self, noise_batch):
        """
        Update the optional particle cloud used for notebook visualizations.

        Args:
            noise_batch: Demand samples used in the Bellman expectations.
        """
        if self.config.snapshot_sample_size <= 0:
            return

        snapshot_sampler_steps = self.config.snapshot_sampler_steps
        if snapshot_sampler_steps is None:
            snapshot_sampler_steps = max(self.config.sampler_steps, 25)

        self.snapshot_states, self.snapshot_actions, _ = self._metropolis_update_pairs(
            states=self.snapshot_states,
            actions=self.snapshot_actions,
            noise_batch=noise_batch,
            sampler_steps=snapshot_sampler_steps,
            refresh_fraction=self.config.snapshot_refresh_fraction,
        )

    def _record_state_action_snapshot(self, iteration):
        """
        Store the current snapshot particle cloud when requested.

        Args:
            iteration: Iteration index being recorded.
        """
        if iteration not in self.config.snapshot_iterations or iteration in self.state_action_snapshots:
            return

        self.state_action_snapshots[iteration] = {
            "states": self.snapshot_states.copy(),
            "actions": self.snapshot_actions.copy(),
        }

    def _project_coef(self, coef):
        """
        Project coefficients onto the tutorial L2 ball, if enabled.

        Args:
            coef: Candidate coefficient vector.
        """
        if self.config.coefficient_clip <= 0.0:
            return coef
        norm = np.linalg.norm(coef, ord=2)
        if norm <= self.config.coefficient_clip:
            return coef
        return coef * (self.config.coefficient_clip / norm)

    def _current_step_size(self, iteration):
        """
        Return the diminishing step size at a given iteration.

        Args:
            iteration: One-based PSMD iteration index.
        """
        return self.config.step_size / ((iteration + 1) ** self.config.step_size_power)

    def _snapshot(self):
        """
        Create a lightweight averaged-iterate view for evaluation routines.
        """
        return _PSMDModelSnapshot(
            mdp=self.mdp,
            basis=self.basis,
            coef=self.avg_coef.copy(),
            num_random_features=self.num_random_features,
        )

    def evaluate_bounds(self):
        """
        Estimate the current lower bound and policy cost of the averaged iterate.
        """
        snapshot = self._snapshot()
        lower_bound_config = self.config.lower_bound.with_updates(initial_state=self.config.initial_state)
        policy_config = self.config.policy_evaluation.with_updates(initial_state=self.config.initial_state)

        lower_bound_stats = SimpleLNSLowerBound(
            mdp=snapshot.mdp,
            basis=snapshot.basis,
            coef=snapshot.coef,
            num_random_features=snapshot.num_random_features,
            **lower_bound_config.to_kwargs(),
        ).estimate_lower_bound_stats()
        policy_cost = estimate_upper_bound_fast(snapshot, config=policy_config)

        return {
            "lower_bound": float(lower_bound_stats["mean"]),
            "policy_cost": float(policy_cost),
            "lower_bound_lambda": float(lower_bound_stats["lambda"]),
            "lower_bound_mc_samples": int(lower_bound_stats["num_samples"]),
        }

    def run(self, verbose: bool = False, show_header: bool = True, show_footer: bool = True):
        """
        Run the PSMD projected-gradient loop and record diagnostics.

        Args:
            verbose: Whether to print progress tables during optimization.
            show_header: Whether to print the progress-table header.
            show_footer: Whether to print the progress-table footer.
        """
        start_time = time.time()
        best_lower_bound = -np.inf
        best_policy_cost = np.inf
        self._record_state_action_snapshot(iteration=0)
        table_width = 108

        if verbose and show_header:
            print("=" * table_width)
            print(
                f"{'seed':>8} {'iter':>8} {'lower bound':>12} {'policy cost':>12} "
                f"{'best lb':>12} {'best pc':>12} {'gap %':>10} {'acc. rate':>12} {'time (sec)':>12}"
            )
            print("-" * table_width)

        for iteration in range(1, self.config.num_iterations + 1):
            self.iteration = iteration
            noise_batch = self.mdp.sample_noise_batch(
                num_samples=self.config.num_noise_samples_per_iteration,
                random_seed=self.config.random_seed + 1000 + iteration,
            )
            sampled_states, sampled_actions = self._sample_violation_pairs(noise_batch=noise_batch)

            grad = self._compute_gradient(sampled_states, sampled_actions, noise_batch=noise_batch)
            step = self._current_step_size(iteration)
            self.coef = self._project_coef(self.coef + step * grad)
            self.avg_coef = ((iteration - 1) * self.avg_coef + self.coef) / iteration
            self._update_snapshot_particles(noise_batch=noise_batch)
            self._record_state_action_snapshot(iteration=iteration)

            should_evaluate = (
                iteration == 1
                or iteration == self.config.num_iterations
                or iteration % self.config.eval_interval == 0
            )
            if should_evaluate:
                metrics = self.evaluate_bounds()
                best_lower_bound = max(best_lower_bound, metrics["lower_bound"])
                best_policy_cost = min(best_policy_cost, metrics["policy_cost"])

                self.history.append(
                    {
                        "iteration": iteration,
                        "lower_bound": metrics["lower_bound"],
                        "policy_cost": metrics["policy_cost"],
                        "best_lower_bound": best_lower_bound,
                        "best_policy_cost": best_policy_cost,
                        "step_size": step,
                        "acceptance_rate": self.last_acceptance_rate,
                        "lower_bound_lambda": metrics["lower_bound_lambda"],
                        "lower_bound_mc_samples": metrics["lower_bound_mc_samples"],
                        "elapsed_seconds": time.time() - start_time,
                    }
                )

                if verbose:
                    gap = 100.0 * (best_policy_cost - best_lower_bound) / best_policy_cost
                    print(
                        f"{self.config.random_seed:8d} {iteration:8d} {metrics['lower_bound']:12.1f} {metrics['policy_cost']:12.1f} "
                        f"{best_lower_bound:12.1f} {best_policy_cost:12.1f} {gap:10.1f} "
                        f"{self.last_acceptance_rate:12.1f} {(time.time() - start_time):12.1f}",
                        flush=True,
                    )
        if verbose:
            print("-" * table_width)
        if verbose and show_footer:
            print("=" * table_width)
            
        self.solution = {
            "coef": self.coef.copy(),
            "avg_coef": self.avg_coef.copy(),
            "history": list(self.history),
            "state_action_snapshots": {
                iteration: {
                    "states": snapshot["states"].copy(),
                    "actions": snapshot["actions"].copy(),
                }
                for iteration, snapshot in self.state_action_snapshots.items()
            },
        }
        return self.solution

    fit = run


SimplePSMD = PSMD
