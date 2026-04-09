# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Simple sampling-based lower-bound estimator for the tutorial project.

This is a lightweight adaptation of the lower-bound logic in the original
research code. It follows the same main idea:

1. define the saddle / Bellman residual function of the fitted VFA
2. sample state-action pairs with more weight on low residual regions
3. combine those samples into a lower-bound estimate

This version is intentionally much smaller than the research implementation,
but it now follows the original code much more closely:

1. use a fixed batch of exogenous noise when evaluating Bellman residuals
2. use many initial state-action points rather than a single chain
3. run multiple chains together in a vectorized sampler

If `emcee` is available, it can use an ensemble sampler; otherwise it falls
back to a vectorized multi-chain random-walk Metropolis sampler.
"""

from __future__ import annotations

from math import gamma, log, pi
import numpy as np

try:
    import emcee
except ImportError:  # pragma: no cover - optional dependency
    emcee = None


class SimpleLNSLowerBound:
    """
    Lightweight lower-bound estimator inspired by the LNS / CVL lower-bound
    estimator in the original repository.

    Notes:
    - This is still a stochastic estimator.
    - It is much simpler than the full research version.
    - It is tailored to the inventory MDP in the tutorial.
    """

    def __init__(
        self,
        mdp,
        basis,
        coef,
        num_random_features,
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
        Build the tutorial lower-bound estimator around a fitted value model.

        Args:
            mdp: Underlying inventory MDP.
            basis: Basis object used by the fitted value approximation.
            coef: Coefficients of the fitted value approximation.
            num_random_features: Number of nonconstant basis functions in use.
            num_mc_init_states: Number of initial particles or walkers.
            chain_length: Number of MCMC steps per chain.
            burn_in: Number of initial MCMC steps to discard.
            proposal_state_std: Proposal standard deviation for states.
            proposal_action_std: Proposal standard deviation for actions.
            random_seed: Base seed for the sampler.
            noise_batch_size: Number of demand draws reused in residual
                calculations.
            sampler: Sampler backend, typically `auto`, `metropolis`, or
                `emcee`.
            num_walkers: Number of ensemble walkers used by `emcee`.
            initial_state: Initial inventory level at which the bound is
                reported.
        """
        self.mdp = mdp
        self.basis = basis
        self.coef = np.asarray(coef, dtype=float)
        self.num_random_features = num_random_features
        self.num_mc_init_states = num_mc_init_states
        self.chain_length = chain_length
        self.burn_in = burn_in
        self.proposal_state_std = proposal_state_std
        self.proposal_action_std = proposal_action_std
        self.random_seed = random_seed
        self.noise_batch_size = noise_batch_size
        self.sampler = sampler
        self.num_walkers = num_walkers
        self.initial_state = initial_state

        self.dim_state_act = self.mdp.dim_state + self.mdp.dim_act
        self.radius_ball_in_state_action = self.mdp.max_order / 2.0
        self.volume_state_action = (self.mdp.upper_state_bound - self.mdp.lower_state_bound) * self.mdp.max_order
        self.diameter_state_action = np.linalg.norm(
            [self.mdp.upper_state_bound - self.mdp.lower_state_bound, self.mdp.max_order]
        )

        self.lambda_ = self.get_lambda()
        if getattr(self.mdp, "list_demand_obs", None) is not None:
            self.lower_bound_noise = np.asarray(self.mdp.list_demand_obs[: self.noise_batch_size], dtype=float)
        else:
            self.lower_bound_noise = self.mdp.get_batch_mdp_noise(
                num_samples=self.noise_batch_size,
                random_seed=self.random_seed + 999,
            )

    def get_vfa(self, state):
        """
        Evaluate the fitted value approximation at one state.

        Args:
            state: One-dimensional inventory state.
        """
        if hasattr(self.basis, "get_vfa") and self.basis.get_vfa.__code__.co_argcount == 4:
            return self.basis.get_vfa(state, self.coef, self.num_random_features)
        return self.basis.get_vfa(state, self.coef)

    def get_vfa_batch(self, states):
        """
        Vectorized VFA evaluation for the tutorial Fourier bases.

        Args:
            states: Collection of one-dimensional states.
        """
        states = np.asarray(states, dtype=float).reshape(-1)

        if hasattr(self.basis, "eval_basis_batch"):
            basis_vals = self.basis.eval_basis_batch(states, num_random_features=self.num_random_features)
            return basis_vals @ self.coef

        if hasattr(self.basis, "params") and self.basis.params is not None:
            intercepts, thetas = self.basis.params
            use_count = len(self.coef)
            basis_vals = np.cos(np.outer(states, thetas[:use_count]) + intercepts[:use_count])
            return basis_vals @ self.coef

        return np.asarray([self.get_vfa(np.asarray([s], dtype=float)) for s in states], dtype=float)

    def get_expected_vfa_next(self, next_states):
        """
        Average the fitted value approximation across sampled next states.

        Args:
            next_states: Collection of sampled next states.
        """
        next_states = np.asarray(next_states, dtype=float).reshape(-1)
        return float(np.mean(self.get_vfa_batch(next_states)))

    def saddle_func_batch(self, states, actions):
        """
        Vectorized Bellman residual for batches of state-action pairs.
        This is where most of the speedup comes from.

        Args:
            states: Collection of inventory states.
            actions: Collection of order quantities matched to `states`.
        """
        states = np.asarray(states, dtype=float).reshape(-1)
        actions = np.asarray(actions, dtype=float).reshape(-1)
        transition_summary = self.mdp.evaluate_state_action_batch(states, actions, noise_batch=self.lower_bound_noise)
        next_states = transition_summary["next_states"]
        expected_cost = transition_summary["expected_cost"]
        vfa_state = self.get_vfa_batch(states)
        expected_vfa_next = self.get_vfa_batch(next_states.reshape(-1)).reshape(len(states), -1).mean(axis=1)

        return (expected_cost + self.mdp.discount * expected_vfa_next - vfa_state) / (1.0 - self.mdp.discount)

    def saddle_func(self, state, action):
        """
        Bellman residual scaled by 1 / (1 - gamma), following the original code.

        Args:
            state: One-dimensional inventory state.
            action: Order quantity paired with `state`.
        """
        state_val = float(np.asarray(state, dtype=float)[0])
        return float(self.saddle_func_batch(np.asarray([state_val]), np.asarray([action]))[0])

    def get_expected_vfa_on_initial_state(self):
        """
        Evaluate the VFA at the fixed initial state.

        For the inventory example in this tutorial, the lower bound is meant
        to correspond to the specific initial state s=5 rather than an initial
        distribution over states.
        """
        init_state = np.asarray([self.initial_state], dtype=float)
        return float(self.get_vfa(init_state))

    def get_lipschitz_cost_bound(self):
        """
        Small problem-specific analogue of the bound used in the original code.
        """
        return (
            (2.0 * self.mdp.discount**2) * self.mdp.purchase_cost
            + self.mdp.holding_cost
            + self.mdp.backlog_cost
            + self.mdp.disposal_cost
            + self.mdp.lost_sale_cost
        ) * self.mdp.max_order

    def get_lns_constant(self):
        """
        Simplified version of the constant used in the original lower bound.
        """
        norm_vfa_coefs = np.linalg.norm(self.coef, 1)
        lipschitz_cost_func = self.get_lipschitz_cost_bound()
        lipschitz_dual_func = (4.0 * norm_vfa_coefs + lipschitz_cost_func) / (1.0 - self.mdp.discount)

        return (
            log(1.0 / self.volume_state_action)
            - lipschitz_dual_func * (self.radius_ball_in_state_action + self.diameter_state_action)
            + self.dim_state_act * log(self.radius_ball_in_state_action)
            + log(gamma((self.dim_state_act / 2.0) + 1.0) / (pi ** (self.dim_state_act / 2.0)))
        )

    def get_lambda(self):
        """
        Compute the temperature parameter used in the LNS-style density.
        """
        constant = self.get_lns_constant()
        return abs(1.0 / (constant + self.mdp.dim_state + self.mdp.dim_act))

    def log_target_density(self, state, action):
        """
        Evaluate the unnormalized log density at one state-action pair.

        Args:
            state: One-dimensional inventory state.
            action: Order quantity paired with `state`.
        """
        if not self.mdp.is_state_action_feasible(state, action):
            return -np.inf
        return -(self.saddle_func(state, action) / self.lambda_)

    def log_target_density_batch(self, x):
        """
        Vectorized log target density for `emcee`.

        Args:
            x: Two-column array whose rows are `[state, action]`.
        """
        x = np.asarray(x, dtype=float)
        states = x[:, 0]
        actions = x[:, 1]
        feasible = (
            (states >= self.mdp.lower_state_bound)
            & (states <= self.mdp.upper_state_bound)
            & (actions >= 0.0)
            & (actions <= self.mdp.max_order)
        )

        out = np.full(len(x), -np.inf, dtype=float)
        if np.any(feasible):
            saddle = self.saddle_func_batch(states[feasible], actions[feasible])
            out[feasible] = -(saddle / self.lambda_)
        return out

    def emcee_sampler(self):
        """
        Optional faster ensemble sampler when `emcee` is installed.
        """
        if emcee is None:
            raise ImportError("emcee is not installed.")

        num_walkers = max(self.num_walkers, self.num_mc_init_states)
        rand_state_list, rand_action_list = self.mdp.get_state_act_for_ALP_constr(num_walkers)
        initial = np.column_stack(
            [
                np.asarray([float(s[0]) for s in rand_state_list], dtype=float),
                np.asarray(rand_action_list, dtype=float),
            ]
        )

        sampler = emcee.EnsembleSampler(
            nwalkers=num_walkers,
            ndim=2,
            log_prob_fn=self.log_target_density_batch,
            vectorize=True,
        )
        sampler.run_mcmc(initial, self.chain_length, progress=False)
        chain = sampler.get_chain(discard=self.burn_in, flat=True)
        return [(np.asarray([state], dtype=float), float(action)) for state, action in chain]

    def vectorized_metropolis_sampler(self):
        """
        Vectorized multi-chain random-walk Metropolis sampler.

        This mirrors the spirit of the original code more closely than a
        single-chain MH routine: we start from many sampled state-action pairs
        and evolve them together.
        """
        rng = np.random.RandomState(self.random_seed)
        rand_state_list, rand_action_list = self.mdp.get_state_act_for_ALP_constr(self.num_mc_init_states)

        cur_states = np.asarray([float(s[0]) for s in rand_state_list], dtype=float)
        cur_actions = np.asarray(rand_action_list, dtype=float)
        cur_logp = self.log_target_density_batch(np.column_stack([cur_states, cur_actions]))

        kept_states = []
        kept_actions = []

        for step in range(self.chain_length):
            prop_states = cur_states + rng.normal(scale=self.proposal_state_std, size=len(cur_states))
            prop_actions = cur_actions + rng.normal(scale=self.proposal_action_std, size=len(cur_actions))

            prop_states = np.clip(prop_states, self.mdp.lower_state_bound, self.mdp.upper_state_bound)
            prop_actions = np.clip(prop_actions, 0.0, self.mdp.max_order)

            prop_logp = self.log_target_density_batch(np.column_stack([prop_states, prop_actions]))
            log_u = np.log(rng.uniform(size=len(cur_states)))
            accept = log_u < (prop_logp - cur_logp)

            cur_states[accept] = prop_states[accept]
            cur_actions[accept] = prop_actions[accept]
            cur_logp[accept] = prop_logp[accept]

            if step >= self.burn_in:
                kept_states.append(cur_states.copy())
                kept_actions.append(cur_actions.copy())

        kept_states = np.concatenate(kept_states)
        kept_actions = np.concatenate(kept_actions)
        return [(np.asarray([state], dtype=float), float(action)) for state, action in zip(kept_states, kept_actions)]

    def estimate_lower_bound_stats(self):
        """
        Produce the mean lower-bound estimate together with its Monte Carlo
        standard error.
        """
        use_emcee = self.sampler == "emcee" or (self.sampler == "auto" and emcee is not None)
        samples = self.emcee_sampler() if use_emcee else self.vectorized_metropolis_sampler()

        states = np.asarray([float(state[0]) for state, _ in samples], dtype=float)
        actions = np.asarray([float(action) for _, action in samples], dtype=float)
        saddle_values = self.saddle_func_batch(states, actions)
        saddle_mean = float(np.mean(saddle_values))
        saddle_se = float(np.std(saddle_values, ddof=1) / np.sqrt(len(saddle_values))) if len(saddle_values) > 1 else 0.0

        expected_vfa_init = self.get_expected_vfa_on_initial_state()
        lns_constant = self.get_lns_constant()

        lower_bound = (
            saddle_mean
            + self.dim_state_act * self.lambda_ * log(self.lambda_)
            + self.lambda_ * lns_constant
            + expected_vfa_init
        )
        return {
            "mean": lower_bound,
            "saddle_mean": saddle_mean,
            "saddle_se": saddle_se,
            "num_samples": len(saddle_values),
            "lambda": self.lambda_,
        }

    def estimate_lower_bound(self):
        """
        Return just the mean lower-bound estimate.
        """
        return self.estimate_lower_bound_stats()["mean"]


def _estimate_actual_lower_bound(
    model,
    num_random_features,
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
    return_stats=False,
):
    """
    Shared lower-bound wrapper used by FALP and SGALP.

    Args:
        model: Fitted model providing `mdp`, `basis`, and `coef`.
        num_random_features: Number of nonconstant basis functions in use.
        num_mc_init_states: Number of initial particles or walkers.
        chain_length: Number of MCMC steps per chain.
        burn_in: Number of initial MCMC steps to discard.
        proposal_state_std: Proposal standard deviation for states.
        proposal_action_std: Proposal standard deviation for actions.
        random_seed: Base seed for the sampler.
        noise_batch_size: Number of demand draws reused in residual estimates.
        sampler: Sampler backend, typically `auto`, `metropolis`, or `emcee`.
        num_walkers: Number of walkers used by `emcee`.
        initial_state: Initial inventory level at which the bound is reported.
        return_stats: Whether to return the full Monte Carlo summary.
    """
    estimator = SimpleLNSLowerBound(
        mdp=model.mdp,
        basis=model.basis,
        coef=model.coef,
        num_random_features=num_random_features,
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
    return estimator.estimate_lower_bound_stats() if return_stats else estimator.estimate_lower_bound()


def estimate_actual_lower_bound_falp(
    falp_model,
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
    return_stats=False,
):
    """
    Convenience wrapper for the tutorial FALP class.

    Args:
        falp_model: Fitted FALP model whose lower bound is requested.
        num_mc_init_states: Number of initial particles or walkers.
        chain_length: Number of MCMC steps per chain.
        burn_in: Number of initial MCMC steps to discard.
        proposal_state_std: Proposal standard deviation for states.
        proposal_action_std: Proposal standard deviation for actions.
        random_seed: Base seed for the sampler.
        noise_batch_size: Number of demand draws reused in residual estimates.
        sampler: Sampler backend, typically `auto`, `metropolis`, or `emcee`.
        num_walkers: Number of walkers used by `emcee`.
        initial_state: Initial inventory level at which the bound is reported.
        return_stats: Whether to return the full Monte Carlo summary.
    """
    return _estimate_actual_lower_bound(
        model=falp_model,
        num_random_features=falp_model.num_random_features,
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
        return_stats=return_stats,
    )


def estimate_actual_lower_bound_sgalp(
    sgalp_model,
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
    return_stats=False,
):
    """
    Convenience wrapper for the simple SGALP class.

    Args:
        sgalp_model: Fitted SGALP model whose lower bound is requested.
        num_mc_init_states: Number of initial particles or walkers.
        chain_length: Number of MCMC steps per chain.
        burn_in: Number of initial MCMC steps to discard.
        proposal_state_std: Proposal standard deviation for states.
        proposal_action_std: Proposal standard deviation for actions.
        random_seed: Base seed for the sampler.
        noise_batch_size: Number of demand draws reused in residual estimates.
        sampler: Sampler backend, typically `auto`, `metropolis`, or `emcee`.
        num_walkers: Number of walkers used by `emcee`.
        initial_state: Initial inventory level at which the bound is reported.
        return_stats: Whether to return the full Monte Carlo summary.
    """
    return _estimate_actual_lower_bound(
        model=sgalp_model,
        num_random_features=sgalp_model.current_num_random_features,
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
        return_stats=return_stats,
    )
