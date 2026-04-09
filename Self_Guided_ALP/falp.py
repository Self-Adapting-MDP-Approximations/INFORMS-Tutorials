# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Small FALP implementation for teaching.

The tutorial version keeps the algorithm deliberately compact:
1. choose a basis family
2. sample Bellman inequalities
3. solve the approximate linear program
4. interpret the fitted coefficients as a value-function approximation

The surrounding code now stores parameters in `FALPConfig`, so students can
inspect one object and understand the experimental setup without scrolling
through a long constructor call.
"""

from __future__ import annotations

from itertools import combinations
import numpy as np

try:
    from scipy.optimize import linprog
except ImportError:  # pragma: no cover - optional dependency
    linprog = None

from basis import RandomFourierBasis1D
from .cvl_lower_bound import estimate_actual_lower_bound_falp
from config import FALPConfig, RandomFeatureConfig


RandomFourierBasis = RandomFourierBasis1D


class FALP:
    """
    Small FALP class using constraint sampling.

    The public API remains notebook-friendly, but the recommended interface is
    to pass a single `FALPConfig` object instead of many scalar arguments.
    """

    def __init__(
        self,
        mdp,
        config: FALPConfig | None = None,
        num_random_features=1,
        num_constraints=40,
        num_state_relevance_samples=200,
        basis_seed=111,
        bandwidth_choices=(1e-3, 1e-4),
        solver="auto",
    ):
        """
        Build the tutorial FALP solver.

        Args:
            mdp: Underlying inventory MDP.
            config: Optional grouped FALP settings.
            num_random_features: Number of nonconstant random features.
            num_constraints: Number of sampled Bellman inequalities.
            num_state_relevance_samples: Number of states used in the ALP
                objective.
            basis_seed: Seed controlling the sampled random-feature family.
            bandwidth_choices: Candidate bandwidths used in feature sampling.
            solver: LP solver choice, typically `auto` or `scipy`.
        """
        self.mdp = mdp
        self.config = self._build_config(
            config=config,
            num_random_features=num_random_features,
            num_constraints=num_constraints,
            num_state_relevance_samples=num_state_relevance_samples,
            basis_seed=basis_seed,
            bandwidth_choices=bandwidth_choices,
            solver=solver,
        )

        self.num_random_features = self.config.num_random_features
        self.num_constraints = self.config.num_constraints
        self.num_state_relevance_samples = self.config.num_state_relevance_samples
        self.solver = self.config.solver

        self.basis = RandomFourierBasis1D(
            max_random_features=self.num_random_features,
            config=self.config.random_features,
        )

        self.coef = None
        self.lp_data = None
        self.solution = None
        self.solver_result = None

    @staticmethod
    def _build_config(
        config,
        num_random_features,
        num_constraints,
        num_state_relevance_samples,
        basis_seed,
        bandwidth_choices,
        solver,
    ):
        """
        Merge legacy scalar arguments into a single `FALPConfig`.

        Args:
            config: Optional already-built `FALPConfig`.
            num_random_features: Number of nonconstant random features.
            num_constraints: Number of sampled Bellman inequalities.
            num_state_relevance_samples: Number of states used in the ALP
                objective.
            basis_seed: Seed controlling the sampled random-feature family.
            bandwidth_choices: Candidate bandwidths used in feature sampling.
            solver: LP solver choice, typically `auto` or `scipy`.
        """
        if config is not None:
            return config
        return FALPConfig(
            num_random_features=num_random_features,
            num_constraints=num_constraints,
            num_state_relevance_samples=num_state_relevance_samples,
            random_features=RandomFeatureConfig(
                bandwidth_choices=bandwidth_choices,
                random_seed=basis_seed,
            ),
            solver=solver,
        )

    def build_sampled_constraint(self, state, action):
        """
        Build one sampled Bellman inequality:
            (phi(s) - gamma E[phi(s')])^T r <= E[c(s,a)]

        Args:
            state: Sampled inventory state.
            action: Sampled order quantity.
        """

        phi_state = self.basis.eval_basis(state, num_random_features=self.num_random_features)
        next_states = self.mdp.get_batch_next_state(state, action)
        expected_phi_next = self.basis.expected_basis(next_states, num_random_features=self.num_random_features)
        expected_cost = self.mdp.get_expected_cost(state, action)

        lhs = phi_state - self.mdp.discount * expected_phi_next
        rhs = expected_cost
        return lhs, rhs

    # Backward-compatible name.
    compute_single_constraint = build_sampled_constraint

    def build_lp(self):
        """
        Assemble the sampled FALP linear program.

        Args:
            None.
        """

        state_list, action_list = self.mdp.sample_constraint_state_actions(self.num_constraints)

        constraint_rows = []
        rhs_values = []
        for state, action in zip(state_list, action_list):
            lhs, rhs = self.build_sampled_constraint(state, action)
            constraint_rows.append(lhs)
            rhs_values.append(rhs)

        state_relevance_samples = self.mdp.sample_state_relevance_states(self.num_state_relevance_samples)
        objective_coef = np.mean(
            [self.basis.eval_basis(state, num_random_features=self.num_random_features) for state in state_relevance_samples],
            axis=0,
        )

        self.lp_data = {
            "A": np.asarray(constraint_rows, dtype=float),
            "b": np.asarray(rhs_values, dtype=float),
            "c": np.asarray(objective_coef, dtype=float),
        }
        return self.lp_data

    def solve_lp_by_vertex_enumeration(self, tolerance=1e-9):
        """
        Teaching-only fallback solver for very small LPs.

        Args:
            tolerance: Determinant threshold used to skip nearly singular
                candidate bases.
        """

        if self.lp_data is None:
            raise ValueError("Build the LP before solving it.")

        A = self.lp_data["A"]
        b = self.lp_data["b"]
        c = self.lp_data["c"]

        num_constraints, num_variables = A.shape
        candidates = []

        for active_rows in combinations(range(num_constraints), num_variables):
            M = A[list(active_rows), :]
            if abs(np.linalg.det(M)) < tolerance:
                continue
            x = np.linalg.solve(M, b[list(active_rows)])
            if np.all(A @ x <= b + 1e-7):
                candidates.append(x)

        zero = np.zeros(num_variables)
        if np.all(A @ zero <= b + 1e-7):
            candidates.append(zero)

        if not candidates:
            raise ValueError(
                "No feasible candidate point was found. "
                "Increase the number of constraints or reduce the basis size."
            )

        candidate_array = np.asarray(candidates, dtype=float)
        objective_values = candidate_array @ c
        best_index = int(np.argmax(objective_values))

        self.coef = candidate_array[best_index]
        return {
            "coef": self.coef,
            "objective_value": float(objective_values[best_index]),
            "num_candidates_checked": len(candidate_array),
            "solver_name": "vertex_enumeration",
        }

    def solve_lp_with_scipy(self):
        """
        Solve the LP with SciPy HiGHS.
        """

        if linprog is None:
            raise ImportError("scipy is not installed, so linprog is unavailable.")
        if self.lp_data is None:
            raise ValueError("Build the LP before solving it.")

        A = self.lp_data["A"]
        b = self.lp_data["b"]
        c = self.lp_data["c"]

        result = linprog(
            c=-c,
            A_ub=A,
            b_ub=b,
            bounds=[(None, None)] * A.shape[1],
            method="highs",
        )
        self.solver_result = result

        if not result.success:
            raise ValueError(f"LP solve failed: {result.message}")

        self.coef = np.asarray(result.x, dtype=float)
        return {
            "coef": self.coef,
            "objective_value": float(c @ self.coef),
            "solver_name": "scipy_highs",
            "solver_status": result.status,
            "solver_message": result.message,
        }

    def fit(self):
        """
        Build and solve the sampled FALP model.
        """

        self.build_lp()
        use_scipy = self.solver == "scipy" or (self.solver == "auto" and linprog is not None)

        if use_scipy:
            self.solution = self.solve_lp_with_scipy()
        else:
            self.solution = self.solve_lp_by_vertex_enumeration()
        return self.solution

    def get_falp_objective(self):
        """
        Return the FALP LP objective value of the fitted model.
        """

        if self.coef is None or self.lp_data is None:
            raise ValueError("Fit the FALP model before requesting its objective value.")
        return float(self.lp_data["c"] @ self.coef)

    def estimate_cvl_lower_bound(
        self,
        num_mc_init_states=64,
        chain_length=800,
        burn_in=400,
        proposal_state_std=0.8,
        proposal_action_std=0.8,
        random_seed=333,
        noise_batch_size=1000,
        sampler="auto",
        num_walkers=32,
        initial_state=5.0,
    ):
        """
        Estimate a sampling-based CVL / LNS lower bound for the fitted model.

        Args:
            num_mc_init_states: Number of initial particles or walkers.
            chain_length: Number of MCMC steps per chain.
            burn_in: Number of initial MCMC steps to discard.
            proposal_state_std: Proposal standard deviation for state moves.
            proposal_action_std: Proposal standard deviation for action moves.
            random_seed: Base seed for the sampler.
            noise_batch_size: Number of demand draws reused in residual
                calculations.
            sampler: Sampler backend, typically `auto`, `metropolis`, or
                `emcee`.
            num_walkers: Number of walkers used by the optional `emcee`
                sampler.
            initial_state: Initial inventory level at which the bound is
                reported.
        """

        if self.coef is None:
            raise ValueError("Fit the FALP model before estimating the CVL lower bound.")
        return estimate_actual_lower_bound_falp(
            self,
            num_mc_init_states=num_mc_init_states,
            chain_length=chain_length,
            burn_in=burn_in,
            proposal_state_std=proposal_state_std,
            proposal_action_std=proposal_action_std,
            random_seed=random_seed,
            noise_batch_size=noise_batch_size,
            sampler=sampler,
            num_walkers=num_walkers,
            initial_state=initial_state,
        )

    def evaluate_vfa_on_grid(self, num_points=11):
        """
        Evaluate the fitted value-function approximation on a simple grid.

        Args:
            num_points: Number of evenly spaced states in the evaluation grid.
        """

        if self.coef is None:
            raise ValueError("Fit the FALP model before evaluating the value function.")
        grid = np.linspace(self.mdp.lower_state_bound, self.mdp.upper_state_bound, num_points)
        values = [
            self.basis.get_vfa(
                np.asarray([state_value], dtype=float),
                self.coef,
                num_random_features=self.num_random_features,
            )
            for state_value in grid
        ]
        return grid, np.asarray(values, dtype=float)


# Backward-compatible alias for older notebooks.
SimpleFALP = FALP
