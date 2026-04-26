# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Small self-guided ALP implementation for teaching.

SGALP uses the same random-feature family as FALP, but it grows the basis set
stage by stage. After each stage it adds guiding constraints so the new value
function approximation does not fall too far below the previous one on sampled
states. Keeping the basis logic shared with FALP makes that distinction much
easier to see.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from basis import RandomFourierBasis1D
from config import GuidingConstraintConfig, HiGHSSolverConfig, RandomFeatureConfig, SGALPConfig


NestedRandomFourierBasis = RandomFourierBasis1D


class SelfGuidedALP:
    """
    Small self-guided ALP class.

    The recommended interface is to pass a single `SGALPConfig` object. The
    older scalar keyword arguments are still accepted so the existing notebook
    remains usable.
    """

    def __init__(
        self,
        mdp,
        config: SGALPConfig | None = None,
        max_random_features=10,
        batch_size=1,
        num_constraints=40,
        num_state_relevance_samples=200,
        num_guiding_states=100,
        basis_seed=111,
        bandwidth_choices=(1e-3, 1e-4),
        guiding_violation=0.0,
        guiding_relax_fraction=0.02,
        guiding_abs_floor=1e-6,
        guiding_retry_scales=(1.0, 2.0, 5.0, 10.0),
        highs_method="highs-ds",
        primal_feasibility_tolerance=1e-7,
        dual_feasibility_tolerance=1e-7,
    ):
        """
        Build the tutorial SGALP solver.

        Args:
            mdp: Underlying inventory MDP.
            config: Optional grouped SGALP settings.
            max_random_features: Largest nonconstant basis size to solve.
            batch_size: Number of new features added at each stage.
            num_constraints: Number of sampled Bellman inequalities per stage.
            num_state_relevance_samples: Number of states used in the ALP
                objective.
            num_guiding_states: Number of states used for guiding constraints.
            basis_seed: Seed controlling the sampled random-feature family.
            bandwidth_choices: Candidate bandwidths used in feature sampling.
            guiding_violation: Additive guiding-constraint allowance.
            guiding_relax_fraction: Relative guiding-constraint allowance.
            guiding_abs_floor: Minimum positive guiding allowance.
            guiding_retry_scales: Relaxation multipliers tried after LP failure.
            highs_method: HiGHS backend passed to SciPy.
            primal_feasibility_tolerance: HiGHS primal-feasibility tolerance.
            dual_feasibility_tolerance: HiGHS dual-feasibility tolerance.
        """
        self.mdp = mdp
        self.config = self._build_config(
            config=config,
            max_random_features=max_random_features,
            batch_size=batch_size,
            num_constraints=num_constraints,
            num_state_relevance_samples=num_state_relevance_samples,
            num_guiding_states=num_guiding_states,
            basis_seed=basis_seed,
            bandwidth_choices=bandwidth_choices,
            guiding_violation=guiding_violation,
            guiding_relax_fraction=guiding_relax_fraction,
            guiding_abs_floor=guiding_abs_floor,
            guiding_retry_scales=guiding_retry_scales,
            highs_method=highs_method,
            primal_feasibility_tolerance=primal_feasibility_tolerance,
            dual_feasibility_tolerance=dual_feasibility_tolerance,
        )

        self.max_random_features = self.config.max_random_features
        self.batch_size = self.config.batch_size
        self.num_constraints = self.config.num_constraints
        self.num_state_relevance_samples = self.config.num_state_relevance_samples
        self.num_guiding_states = self.config.guiding.num_guiding_states

        self.guiding_violation = self.config.guiding.allowed_violation
        self.guiding_relax_fraction = self.config.guiding.relax_fraction
        self.guiding_abs_floor = self.config.guiding.absolute_floor
        self.guiding_retry_scales = self.config.guiding.retry_scales

        self.highs_method = self.config.solver.method
        self.primal_feasibility_tolerance = self.config.solver.primal_feasibility_tolerance
        self.dual_feasibility_tolerance = self.config.solver.dual_feasibility_tolerance

        self.basis = RandomFourierBasis1D(
            max_random_features=self.max_random_features,
            config=self.config.random_features,
        )

        self.current_num_random_features = None
        self.coef = None
        self.solver_result = None
        self.history = []

    @staticmethod
    def _build_config(
        config,
        max_random_features,
        batch_size,
        num_constraints,
        num_state_relevance_samples,
        num_guiding_states,
        basis_seed,
        bandwidth_choices,
        guiding_violation,
        guiding_relax_fraction,
        guiding_abs_floor,
        guiding_retry_scales,
        highs_method,
        primal_feasibility_tolerance,
        dual_feasibility_tolerance,
    ):
        """
        Merge legacy scalar arguments into a single `SGALPConfig`.

        Args:
            config: Optional already-built `SGALPConfig`.
            max_random_features: Largest nonconstant basis size to solve.
            batch_size: Number of new features added at each stage.
            num_constraints: Number of sampled Bellman inequalities per stage.
            num_state_relevance_samples: Number of states used in the ALP
                objective.
            num_guiding_states: Number of states used for guiding constraints.
            basis_seed: Seed controlling the sampled random-feature family.
            bandwidth_choices: Candidate bandwidths used in feature sampling.
            guiding_violation: Additive guiding-constraint allowance.
            guiding_relax_fraction: Relative guiding-constraint allowance.
            guiding_abs_floor: Minimum positive guiding allowance.
            guiding_retry_scales: Relaxation multipliers tried after LP failure.
            highs_method: HiGHS backend passed to SciPy.
            primal_feasibility_tolerance: HiGHS primal-feasibility tolerance.
            dual_feasibility_tolerance: HiGHS dual-feasibility tolerance.
        """
        if config is not None:
            return config
        return SGALPConfig(
            max_random_features=max_random_features,
            batch_size=batch_size,
            num_constraints=num_constraints,
            num_state_relevance_samples=num_state_relevance_samples,
            random_features=RandomFeatureConfig(
                bandwidth_choices=bandwidth_choices,
                random_seed=basis_seed,
            ),
            guiding=GuidingConstraintConfig(
                num_guiding_states=num_guiding_states,
                allowed_violation=guiding_violation,
                relax_fraction=guiding_relax_fraction,
                absolute_floor=guiding_abs_floor,
                retry_scales=guiding_retry_scales,
            ),
            solver=HiGHSSolverConfig(
                method=highs_method,
                primal_feasibility_tolerance=primal_feasibility_tolerance,
                dual_feasibility_tolerance=dual_feasibility_tolerance,
            ),
        )

    def build_sampled_constraint(self, state, action, num_random_features):
        """
        Build one sampled Bellman inequality for the requested basis size.

        Args:
            state: Sampled inventory state.
            action: Sampled order quantity.
            num_random_features: Number of nonconstant random features active at
                the current stage.
        """

        phi_state = self.basis.eval_basis(state, num_random_features=num_random_features)
        next_states = self.mdp.get_batch_next_state(state, action)
        expected_phi_next = self.basis.expected_basis(next_states, num_random_features=num_random_features)
        expected_cost = self.mdp.get_expected_cost(state, action)

        lhs = phi_state - self.mdp.discount * expected_phi_next
        rhs = expected_cost
        return lhs, rhs

    # Backward-compatible name.
    compute_single_constraint = build_sampled_constraint

    def build_falp_lp(self, num_random_features):
        """
        Build the sampled FALP LP for a given stage size.

        Args:
            num_random_features: Number of nonconstant random features active at
                the current stage.
        """

        state_list, action_list = self.mdp.sample_constraint_state_actions(self.num_constraints)

        constraint_rows = []
        rhs_values = []
        for state, action in zip(state_list, action_list):
            lhs, rhs = self.build_sampled_constraint(state, action, num_random_features)
            constraint_rows.append(lhs)
            rhs_values.append(rhs)

        state_relevance_samples = self.mdp.sample_state_relevance_states(self.num_state_relevance_samples)
        objective_coef = np.mean(
            [self.basis.eval_basis(state, num_random_features=num_random_features) for state in state_relevance_samples],
            axis=0,
        )

        return {
            "A": np.asarray(constraint_rows, dtype=float),
            "b": np.asarray(rhs_values, dtype=float),
            "c": np.asarray(objective_coef, dtype=float),
        }

    def build_guiding_constraints(
        self,
        num_random_features,
        prev_coef,
        prev_num_random_features,
        relax_scale=1.0,
    ):
        """
        Build the SGALP guiding inequalities.

        Args:
            num_random_features: Number of nonconstant random features active at
                the current stage.
            prev_coef: Coefficients from the previous SGALP stage.
            prev_num_random_features: Number of nonconstant features used at the
                previous stage.
            relax_scale: Multiplier applied to the guiding allowance.
        """

        guiding_states = self.mdp.sample_state_relevance_states(self.num_guiding_states)

        A_guide = []
        b_guide = []
        for state in guiding_states:
            prev_value = self.basis.get_vfa(
                state,
                prev_coef,
                num_random_features=prev_num_random_features,
            )
            phi_new = self.basis.eval_basis(state, num_random_features=num_random_features).astype(float)

            phi_new[np.abs(phi_new) < 1e-10] = 0.0

            allowable_violation = max(
                self.guiding_abs_floor,
                self.guiding_violation,
                self.guiding_relax_fraction * abs(prev_value),
            )
            allowable_violation *= relax_scale

            A_guide.append(-phi_new)
            b_guide.append(-(prev_value - allowable_violation))

        A_guide = np.asarray(A_guide, dtype=float)
        b_guide = np.asarray(b_guide, dtype=float)

        if len(A_guide) > 0:
            packed = np.round(np.column_stack([A_guide, b_guide]), decimals=10)
            _, unique_idx = np.unique(packed, axis=0, return_index=True)
            unique_idx = np.sort(unique_idx)
            A_guide = A_guide[unique_idx]
            b_guide = b_guide[unique_idx]

        return A_guide, b_guide

    def solve_lp(self, A, b, c):
        """
        Solve:
            maximize c^T r
            subject to A r <= b

        Args:
            A: Left-hand-side matrix of inequality constraints.
            b: Right-hand-side vector of inequality constraints.
            c: Objective coefficients.
        """

        result = linprog(
            c=-c,
            A_ub=A,
            b_ub=b,
            bounds=[(None, None)] * A.shape[1],
            method=self.highs_method,
            options={
                "presolve": True,
                "primal_feasibility_tolerance": self.primal_feasibility_tolerance,
                "dual_feasibility_tolerance": self.dual_feasibility_tolerance,
            },
        )

        self.solver_result = result

        if not result.success:
            raise ValueError(f"LP solve failed: {result.message}")

        return {
            "coef": np.asarray(result.x, dtype=float),
            "objective_value": float(c @ result.x),
            "solver_name": "scipy_highs",
            "solver_status": result.status,
            "solver_message": result.message,
        }

    def fit_stage(self, num_random_features, prev_coef=None, prev_num_random_features=None):
        """
        Solve one SGALP stage.

        If guiding constraints make the LP numerically fragile, retry with
        progressively looser guiding allowances.

        Args:
            num_random_features: Number of nonconstant features active at the
                current stage.
            prev_coef: Coefficients from the previous SGALP stage.
            prev_num_random_features: Number of nonconstant features used at the
                previous stage.
        """

        lp_data = self.build_falp_lp(num_random_features)
        A_base = lp_data["A"]
        b_base = lp_data["b"]
        c = lp_data["c"]

        if prev_coef is None:
            solution = self.solve_lp(A_base, b_base, c)
            solution["num_random_features"] = num_random_features
            solution["num_basis_functions"] = num_random_features + 1
            solution["num_falp_constraints"] = len(lp_data["b"])
            solution["num_guiding_constraints"] = 0
            solution["guiding_relax_used"] = 0.0
            solution["total_constraints"] = len(b_base)
            solution["lp_data"] = {"A": A_base, "b": b_base, "c": c}
            return solution

        last_error = None

        for relax_scale in self.guiding_retry_scales:
            A_guide, b_guide = self.build_guiding_constraints(
                num_random_features=num_random_features,
                prev_coef=prev_coef,
                prev_num_random_features=prev_num_random_features,
                relax_scale=relax_scale,
            )

            A = np.vstack([A_base, A_guide])
            b = np.concatenate([b_base, b_guide])

            try:
                solution = self.solve_lp(A, b, c)
                solution["num_random_features"] = num_random_features
                solution["num_basis_functions"] = num_random_features + 1
                solution["num_falp_constraints"] = len(lp_data["b"])
                solution["num_guiding_constraints"] = len(b_guide)
                solution["guiding_relax_used"] = relax_scale
                solution["total_constraints"] = len(b)
                solution["lp_data"] = {"A": A, "b": b, "c": c}
                return solution
            except ValueError as exc:
                last_error = exc

        raise ValueError(
            "SGALP stage failed even after relaxing guiding constraints. "
            f"Last error: {last_error}"
        )

    def stage_feature_counts(self):
        """
        Return the basis sizes solved during the SGALP sequence.

        We always include the final basis size, even if `batch_size` does not
        divide `max_random_features` exactly.
        """

        if self.max_random_features == 0:
            return [0]

        counts = list(range(self.batch_size, self.max_random_features + 1, self.batch_size))
        if counts[-1] != self.max_random_features:
            counts.append(self.max_random_features)
        return counts

    def fit(self):
        """
        Run the full self-guided ALP sequence.
        """

        self.history = []
        prev_coef = None
        prev_num_random_features = None

        for num_random_features in self.stage_feature_counts():
            solution = self.fit_stage(
                num_random_features=num_random_features,
                prev_coef=prev_coef,
                prev_num_random_features=prev_num_random_features,
            )

            self.coef = solution["coef"]
            self.current_num_random_features = num_random_features
            self.history.append(solution)

            prev_coef = solution["coef"]
            prev_num_random_features = num_random_features

        return self.history[-1]

    def print_history(self):
        """
        Print a compact summary table of the fitted SGALP stages.
        """
        if not self.history:
            print("No SGALP stages have been fit yet.")
            return

        print("=" * 102)
        print("Algorithm name:  Self-Guided ALP")
        print("Basis family:    Random Fourier Features")
        print("=" * 102)
        print(
            f"{'m':>6} {'#basis':>8} {'SGALP obj':>16} {'#FALP constr':>14} "
            f"{'#guide constr':>15} {'relax':>10} {'solver':>12}"
        )
        print("-" * 102)

        for item in self.history:
            print(
                f"{item['num_random_features']:>6} "
                f"{item['num_basis_functions']:>8} "
                f"{item['objective_value']:>16.6f} "
                f"{item['num_falp_constraints']:>14} "
                f"{item['num_guiding_constraints']:>15} "
                f"{item['guiding_relax_used']:>10.2f} "
                f"{item['solver_message'][:12]:>12}"
            )

        print("=" * 102)

    def evaluate_vfa_on_grid(self, num_points=11):
        """
        Evaluate the current SGALP approximation on a simple state grid.

        Args:
            num_points: Number of evenly spaced states in the evaluation grid.
        """

        if self.coef is None:
            raise ValueError("Fit the model before evaluating the value function.")

        grid = np.linspace(self.mdp.lower_state_bound, self.mdp.upper_state_bound, num_points)
        values = [
            self.basis.get_vfa(
                np.asarray([state_value], dtype=float),
                self.coef,
                num_random_features=self.current_num_random_features,
            )
            for state_value in grid
        ]
        return grid, np.asarray(values, dtype=float)


# Backward-compatible alias for older notebooks.
SimpleSelfGuidedALP = SelfGuidedALP
