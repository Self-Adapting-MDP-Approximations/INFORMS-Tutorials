# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

The main goal of this file is pedagogical: keep the notebook readable by
moving repetitive experiment code into well-named helpers. The helpers now use
small config objects so a reader can see the key modeling choices in one place
instead of scanning long parameter lists across multiple functions.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from psmd.psmd import PSMD
from self_guided_alp.cvl_lower_bound import estimate_actual_lower_bound_falp, estimate_actual_lower_bound_sgalp
from self_guided_alp.falp import FALP
from mdp import make_inventory_mdp
from policy import build_greedy_policy_lookup as _build_greedy_policy_lookup
from policy import estimate_upper_bound_fast
from self_guided_alp.sgalp import SelfGuidedALP
from config import (
    CONTINUOUS_MDP_NOTEBOOK_CONFIG,
    FALPConfig,
    InventoryMDPConfig,
    LowerBoundConfig,
    PolynomialALPExampleConfig,
    PolicyEvaluationConfig,
    PSMDConfig,
    RandomFeatureConfig,
    SGALPConfig,
    make_shared_evaluation_configs,
)


def estimate_falp_objective(falp_model):
    """
    Return the fitted FALP objective value.

    Args:
        falp_model: Fitted FALP model.
    """
    return falp_model.get_falp_objective()


def estimate_psmd_alp_objective(psmd_model, num_state_relevance_samples=None):
    """
    Estimate the ALP objective induced by the averaged PSMD value approximation.

    Args:
        psmd_model: Fitted PSMD model.
        num_state_relevance_samples: Number of states used to approximate
            E_nu[V(s)].
    """
    default_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.polynomial_alp
    num_state_relevance_samples = (
        default_config.num_state_relevance_samples
        if num_state_relevance_samples is None
        else num_state_relevance_samples
    )
    relevance_states = psmd_model.mdp.sample_state_relevance_states(num_state_relevance_samples)
    basis_matrix = psmd_model.basis.eval_basis_batch(
        relevance_states,
        num_random_features=psmd_model.num_random_features,
    )
    return float(np.mean(basis_matrix @ psmd_model.avg_coef))


def apply_tutorial_plot_style():
    """
    Apply a single plotting style used across the tutorial notebooks.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "figure.titlesize": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def _format_table_value(value, width=16, precision=1):
    """
    Format one table cell for console progress output.

    Args:
        value: Value to display.
        width: Display width used in the printed table.
        precision: Number of decimal digits for numeric values.
    """
    if value is None:
        return f"{'...':>{width}}"
    if isinstance(value, str):
        return f"{value:>{width}}"
    return f"{value:>{width}.{precision}f}"


def _compute_optimality_gap(lower_bound, upper_bound):
    """
    Compute the percentage gap between a lower bound and an upper bound.

    Args:
        lower_bound: Estimated lower bound.
        upper_bound: Estimated upper bound or policy cost.
    """
    if upper_bound in (None, 0):
        return np.nan
    return 100.0 * (upper_bound - lower_bound) / upper_bound


def _active_num_random_features(model):
    """
    Infer how many nonconstant basis functions are active in a fitted model.

    Args:
        model: FALP-, SGALP-, or PSMD-like fitted model.
    """
    if hasattr(model, "current_num_random_features") and model.current_num_random_features is not None:
        return int(model.current_num_random_features)
    if hasattr(model, "num_random_features"):
        return int(model.num_random_features)
    return len(np.asarray(model.coef, dtype=float)) - 1


def evaluate_vfa_on_grid(model, grid):
    """
    Evaluate a fitted tutorial model on a one-dimensional state grid.

    Args:
        model: Fitted model exposing `basis`, `coef`, and `mdp`.
        grid: One-dimensional state grid.
    """
    use_count = _active_num_random_features(model)
    return np.asarray(
        [
            model.basis.get_vfa(
                np.asarray([state_value], dtype=float),
                model.coef,
                num_random_features=use_count,
            )
            for state_value in np.asarray(grid, dtype=float)
        ],
        dtype=float,
    )


def _polynomial_basis_vector(state):
    """
    Evaluate the quadratic polynomial basis [1, s, s^2].

    Args:
        state: One-dimensional inventory state.
    """
    state_value = float(np.asarray(state, dtype=float).reshape(-1)[0])
    return np.asarray([1.0, state_value, state_value**2], dtype=float)


def fit_representative_quadratic_inventory_alp(
    mdp,
    num_constraints=None,
    num_state_relevance_samples=None,
    representative_linear_coef=10.0,
    representative_quadratic_coef=-4.0,
):
    """
    Fit a small sampled ALP and return a representative quadratic beta.

    The sampled ALP gives a reasonable coefficient scale. The returned
    heat-map beta keeps the fitted constant term, then sets representative
    linear and quadratic terms so the illustrative threshold surface displays
    the nonlinear structure discussed in the tutorial text.

    Args:
        mdp: Inventory MDP instance.
        num_constraints: Number of sampled Bellman inequalities.
        num_state_relevance_samples: Number of sampled objective states.
        representative_linear_coef: Linear coefficient used in the
            illustrative heat-map beta.
        representative_quadratic_coef: Quadratic coefficient used in the
            illustrative heat-map beta.
    """
    default_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.polynomial_alp
    num_constraints = default_config.num_constraints if num_constraints is None else num_constraints
    num_state_relevance_samples = (
        default_config.num_state_relevance_samples
        if num_state_relevance_samples is None
        else num_state_relevance_samples
    )

    state_samples, action_samples = mdp.sample_constraint_state_actions(num_constraints)
    relevance_states = mdp.sample_state_relevance_states(num_state_relevance_samples)

    constraint_rows = []
    rhs_values = []
    for state, action in zip(state_samples, action_samples):
        phi_state = _polynomial_basis_vector(state)
        next_states = mdp.get_batch_next_state(state, action)
        expected_phi_next = np.mean(
            np.column_stack(
                [
                    np.ones(len(next_states)),
                    np.asarray(next_states, dtype=float).reshape(-1),
                    np.asarray(next_states, dtype=float).reshape(-1) ** 2,
                ]
            ),
            axis=0,
        )
        expected_cost = mdp.get_expected_cost(state, action)
        constraint_rows.append(phi_state - mdp.discount * expected_phi_next)
        rhs_values.append(expected_cost)

    A = np.asarray(constraint_rows, dtype=float)
    b = np.asarray(rhs_values, dtype=float)
    c = np.mean([_polynomial_basis_vector(state) for state in relevance_states], axis=0)

    result = linprog(c=-c, A_ub=A, b_ub=b, bounds=[(None, None)] * len(c), method="highs")
    if not result.success:
        raise RuntimeError(result.message)

    alp_beta = np.asarray(result.x, dtype=float)
    sampled_slacks = b - A @ alp_beta
    beta = alp_beta.copy()
    beta[1] = representative_linear_coef
    beta[2] = representative_quadratic_coef

    return SimpleNamespace(
        result=result,
        beta=beta,
        alp_beta=alp_beta,
        alp_objective=float(c @ alp_beta),
        sampled_slacks=sampled_slacks,
        num_constraints=num_constraints,
        num_state_relevance_samples=len(relevance_states),
    )


def evaluate_inventory_constraint_threshold(mdp, beta, state_mesh, action_mesh, chunk_size=500):
    """
    Evaluate f(beta; s, a) on a state-action mesh for the inventory MDP.

    Args:
        mdp: Inventory MDP instance.
        beta: Quadratic VFA coefficient vector [beta_0, beta_1, beta_2].
        state_mesh: Meshgrid of inventory states.
        action_mesh: Meshgrid of order quantities.
        chunk_size: Number of state-action pairs evaluated per vectorized call.
    """
    states = np.asarray(state_mesh, dtype=float).reshape(-1)
    actions = np.asarray(action_mesh, dtype=float).reshape(-1)
    values = np.empty_like(states, dtype=float)

    beta_1 = float(beta[1])
    beta_2 = float(beta[2])
    demand_batch = mdp.list_demand_obs

    for start in range(0, len(states), chunk_size):
        end = min(start + chunk_size, len(states))
        batch_states = states[start:end]
        batch_actions = actions[start:end]

        summary = mdp.evaluate_state_action_batch(batch_states, batch_actions, demand_batch)
        next_states = summary["next_states"]
        expected_cost = summary["expected_cost"]
        expected_next_state = next_states.mean(axis=1)
        expected_next_state_squared = (next_states**2).mean(axis=1)

        linear_term = batch_states - mdp.discount * expected_next_state
        quadratic_term = batch_states**2 - mdp.discount * expected_next_state_squared
        values[start:end] = (
            expected_cost
            - beta_1 * linear_term
            - beta_2 * quadratic_term
        ) / (1.0 - mdp.discount)

    return values.reshape(np.asarray(state_mesh).shape)


def make_inventory_violation_plot_data(
    inventory_config: InventoryMDPConfig | None = None,
    num_constraints=None,
    num_state_relevance_samples=None,
    representative_linear_coef=None,
    representative_quadratic_coef=None,
    state_grid_size=None,
    action_grid_size=None,
):
    """
    Build all data needed for the inventory threshold and Gibbs-density plots.

    Args:
        inventory_config: Inventory configuration used to build the MDP.
        num_constraints: Number of sampled ALP constraints.
        num_state_relevance_samples: Number of sampled objective states.
        representative_linear_coef: Linear beta used in the heat map.
        representative_quadratic_coef: Quadratic beta used in the heat map.
        state_grid_size: Number of state grid points.
        action_grid_size: Number of action grid points.
    """
    config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.inventory if inventory_config is None else inventory_config
    default_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.polynomial_alp
    default_policy_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.policy_evaluation
    num_constraints = default_config.num_constraints if num_constraints is None else num_constraints
    num_state_relevance_samples = (
        default_config.num_state_relevance_samples
        if num_state_relevance_samples is None
        else num_state_relevance_samples
    )
    representative_linear_coef      = 2 if representative_linear_coef is None else representative_linear_coef
    representative_quadratic_coef   = -5.0 if representative_quadratic_coef is None else representative_quadratic_coef
    state_grid_size = default_policy_config.state_grid_size if state_grid_size is None else state_grid_size
    action_grid_size = default_policy_config.state_grid_size if action_grid_size is None else action_grid_size

    mdp = make_inventory_mdp(config)
    fit = fit_representative_quadratic_inventory_alp(
        mdp,
        num_constraints=num_constraints,
        num_state_relevance_samples=num_state_relevance_samples,
        representative_linear_coef=representative_linear_coef,
        representative_quadratic_coef=representative_quadratic_coef,
    )

    state_grid = np.linspace(mdp.lower_state_bound, mdp.upper_state_bound, state_grid_size)
    action_grid = np.linspace(0.0, mdp.max_order, action_grid_size)
    state_mesh, action_mesh = np.meshgrid(state_grid, action_grid)
    threshold_values = evaluate_inventory_constraint_threshold(mdp, fit.beta, state_mesh, action_mesh)
    min_index = np.unravel_index(np.argmin(threshold_values), threshold_values.shape)

    return SimpleNamespace(
        mdp=mdp,
        fit=fit,
        beta=fit.beta,
        alp_beta=fit.alp_beta,
        state_grid=state_grid,
        action_grid=action_grid,
        state_mesh=state_mesh,
        action_mesh=action_mesh,
        threshold_values=threshold_values,
        min_index=min_index,
        min_state=float(state_mesh[min_index]),
        min_action=float(action_mesh[min_index]),
        min_value=float(threshold_values[min_index]),
        contour_levels=np.quantile(
            threshold_values,
            [0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95],
        ),
    )


def gibbs_density_on_grid(function_values, lambda_value, state_grid, action_grid):
    """
    Approximate the Gibbs density y*_{lambda,beta} on a rectangular grid.

    Args:
        function_values: Grid of f(beta; s, a) values.
        lambda_value: Positive KL-regularization parameter.
        state_grid: One-dimensional state grid.
        action_grid: One-dimensional action grid.
    """
    shifted_values = np.asarray(function_values, dtype=float) - np.min(function_values)
    unnormalized_density = np.exp(-shifted_values / float(lambda_value))
    action_integral = np.trapezoid(unnormalized_density, action_grid, axis=0)
    normalizing_constant = np.trapezoid(action_integral, state_grid)
    return unnormalized_density / normalizing_constant


def plot_inventory_constraint_threshold_heatmap(
    plot_data,
    figsize=(12, 3.5),
    fontsize=12,
    cmap="viridis",
):
    """
    Plot the inventory ALP constraint-threshold heat map.

    Args:
        plot_data: Output from `make_inventory_violation_plot_data`.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
        cmap: Matplotlib colormap name.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=figsize)

    heatmap = ax.pcolormesh(
        plot_data.state_mesh,
        plot_data.action_mesh,
        plot_data.threshold_values,
        shading="auto",
        cmap=cmap,
    )
    contours = ax.contour(
        plot_data.state_mesh,
        plot_data.action_mesh,
        plot_data.threshold_values,
        levels=plot_data.contour_levels,
        colors="white",
        linewidths=0.8,
        alpha=0.9,
    )
    ax.clabel(contours, inline=True, fontsize=fontsize, fmt="%.0f")
    ax.contour(
        plot_data.state_mesh,
        plot_data.action_mesh,
        plot_data.threshold_values,
        levels=plot_data.contour_levels[:3],
        colors="black",
        linewidths=1.2,
    )
    ax.scatter(
        [plot_data.min_state],
        [plot_data.min_action],
        color="red",
        edgecolor="white",
        linewidth=0.8,
        s=55,
        label=r"$\min_{s,a} f(\beta;s,a)$",
    )

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label(r"$f(\beta; s, a)$", fontsize=fontsize)
    colorbar.ax.tick_params(labelsize=fontsize)

    ax.set_xlabel(r"Inventory state $s$", fontsize=fontsize)
    ax.set_ylabel(r"Order quantity $a$", fontsize=fontsize)
    ax.set_xlim(plot_data.mdp.lower_state_bound, plot_data.mdp.upper_state_bound)
    ax.set_ylim(0.0, plot_data.mdp.max_order)
    ax.tick_params(labelsize=fontsize)
    ax.legend(loc="lower right", frameon=True, fontsize=fontsize)
    fig.tight_layout()
    plt.show()


def plot_inventory_gibbs_densities(
    plot_data,
    lambda_values=(50.0, 100.0, 200.0),
    figsize=(10, 4),
    fontsize=12,
    cmap="viridis",
):
    """
    Plot Gibbs densities for several KL-regularization values.

    Args:
        plot_data: Output from `make_inventory_violation_plot_data`.
        lambda_values: KL-regularization values to display.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
        cmap: Matplotlib colormap name.
    """
    densities = [
        gibbs_density_on_grid(
            plot_data.threshold_values,
            lambda_value,
            plot_data.state_grid,
            plot_data.action_grid,
        )
        for lambda_value in lambda_values
    ]
    density_max = max(float(density.max()) for density in densities)

    fig, axes = plt.subplots(
        1,
        len(lambda_values),
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    image = None
    for ax, lambda_value, density in zip(axes, lambda_values, densities):
        image = ax.pcolormesh(
            plot_data.state_mesh,
            plot_data.action_mesh,
            density,
            shading="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=density_max,
        )
        ax.contour(
            plot_data.state_mesh,
            plot_data.action_mesh,
            plot_data.threshold_values,
            levels=plot_data.contour_levels,
            colors="white",
            linewidths=0.5,
            alpha=0.9,
        )
        ax.scatter(
            [plot_data.min_state],
            [plot_data.min_action],
            color="orange",
            edgecolor="orange",
            linewidth=0.6,
            s=40,
        )
        ax.set_title(rf"$\lambda = {lambda_value:,.0f}$", fontsize=fontsize)
        ax.set_xlabel(r"Inventory state $s$", fontsize=fontsize)
        ax.set_xlim(plot_data.mdp.lower_state_bound, plot_data.mdp.upper_state_bound)
        ax.set_ylim(0.0, plot_data.mdp.max_order)
        
        ax.set_xticks(np.arange(plot_data.mdp.lower_state_bound, plot_data.mdp.upper_state_bound,4))
        ax.set_yticks(np.arange(0, plot_data.mdp.max_order,2))

        ax.tick_params(labelsize=fontsize)

    axes[0].set_ylabel(r"Order quantity $a$", fontsize=fontsize)
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.9)
    colorbar.set_label(r"$y^*_{\lambda,\beta}(s,a)$", fontsize=fontsize)
    colorbar.ax.tick_params(labelsize=fontsize)
    plt.show()
    return fig, axes, densities


def estimate_cvl_lower_bound(
    falp_model,
    num_mc_init_states=None,
    chain_length=None,
    burn_in=None,
    proposal_state_std=None,
    proposal_action_std=None,
    random_seed=None,
    noise_batch_size=None,
    sampler=None,
    num_walkers=None,
    initial_state=None,
    config: LowerBoundConfig | None = None,
):
    """
    Convenience wrapper for the tutorial lower-bound estimator.

    Args:
        falp_model: Fitted FALP model whose lower bound is requested.
        num_mc_init_states: Number of initial particles or walkers.
        chain_length: Number of MCMC steps per chain.
        burn_in: Number of initial MCMC steps to discard.
        proposal_state_std: Proposal standard deviation for state moves.
        proposal_action_std: Proposal standard deviation for action moves.
        random_seed: Base seed for the sampler.
        noise_batch_size: Number of demand draws reused in residual estimates.
        sampler: Sampler backend, typically `auto`, `metropolis`, or `emcee`.
        num_walkers: Number of walkers used by the optional `emcee` sampler.
        initial_state: Initial inventory level at which the bound is reported.
        config: Optional grouped lower-bound settings.
    """

    default_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.lower_bound
    lower_bound_config = _make_lower_bound_config(
        config=config,
        num_mc_init_states=default_config.num_mc_init_states if num_mc_init_states is None else num_mc_init_states,
        chain_length=default_config.chain_length if chain_length is None else chain_length,
        burn_in=default_config.burn_in if burn_in is None else burn_in,
        proposal_state_std=default_config.proposal_state_std if proposal_state_std is None else proposal_state_std,
        proposal_action_std=default_config.proposal_action_std if proposal_action_std is None else proposal_action_std,
        random_seed=default_config.random_seed if random_seed is None else random_seed,
        noise_batch_size=default_config.noise_batch_size if noise_batch_size is None else noise_batch_size,
        sampler=default_config.sampler if sampler is None else sampler,
        num_walkers=default_config.num_walkers if num_walkers is None else num_walkers,
        initial_state=default_config.initial_state if initial_state is None else initial_state,
    )
    return estimate_actual_lower_bound_falp(falp_model, **lower_bound_config.to_kwargs())


def build_greedy_policy_lookup(
    model,
    state_grid_size=None,
    policy_noise_batch_size=None,
    policy_noise_seed=None,
    config: PolicyEvaluationConfig | None = None,
):
    """
    Precompute greedy actions on a dense one-dimensional state grid.

    Args:
        model: Fitted value-function approximation used in one-step lookahead.
        state_grid_size: Number of states in the lookup grid.
        policy_noise_batch_size: Number of demand samples used in one-step
            lookahead.
        policy_noise_seed: Seed controlling the one-step lookahead noise batch.
        config: Optional grouped policy-evaluation settings.
    """

    default_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.policy_evaluation
    policy_config = _make_policy_config(
        config=config,
        state_grid_size=default_config.state_grid_size if state_grid_size is None else state_grid_size,
        policy_noise_batch_size=default_config.policy_noise_batch_size if policy_noise_batch_size is None else policy_noise_batch_size,
        policy_noise_seed=default_config.policy_noise_seed if policy_noise_seed is None else policy_noise_seed,
        num_trajectories=default_config.num_trajectories,
        horizon=default_config.horizon,
        simulation_seed=default_config.simulation_seed,
        initial_state=default_config.initial_state,
    )
    return _build_greedy_policy_lookup(model, config=policy_config)


def estimate_upper_bound_falp_fast(
    falp_model,
    num_trajectories=None,
    horizon=None,
    sim_seed=2026,
    simulation_seed=None,
    state_grid_size=None,
    policy_noise_batch_size=None,
    initial_state=None,
    return_se=False,
    config: PolicyEvaluationConfig | None = None,
):
    """
    Fast upper-bound estimator using a precomputed greedy-policy lookup table.

    Args:
        falp_model: Fitted FALP model used to build the greedy policy.
        num_trajectories: Number of Monte Carlo policy-evaluation trajectories.
        horizon: Number of periods simulated in each trajectory.
        sim_seed: Legacy seed name for policy simulation.
        simulation_seed: Clearer alias for `sim_seed`.
        state_grid_size: Number of states in the greedy-policy lookup grid.
        policy_noise_batch_size: Number of demand samples used in one-step
            lookahead.
        initial_state: Initial inventory level used in simulation.
        return_se: Whether to also return the Monte Carlo standard error.
        config: Optional grouped policy-evaluation settings.
    """
    if simulation_seed is not None:
        sim_seed = simulation_seed

    default_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.policy_evaluation
    policy_config = _make_policy_config(
        config=config,
        state_grid_size=default_config.state_grid_size if state_grid_size is None else state_grid_size,
        policy_noise_batch_size=default_config.policy_noise_batch_size if policy_noise_batch_size is None else policy_noise_batch_size,
        policy_noise_seed=sim_seed + 500000,
        num_trajectories=default_config.num_trajectories if num_trajectories is None else num_trajectories,
        horizon=default_config.horizon if horizon is None else horizon,
        simulation_seed=sim_seed,
        initial_state=default_config.initial_state if initial_state is None else initial_state,
    )
    return estimate_upper_bound_fast(falp_model, config=policy_config, return_se=return_se)


def run_polynomial_sampled_alp_example(
    seeds=None,
    demand_samples_per_constraint=None,
    action_step=None,
    polynomial_exponents=None,
    num_constraints=None,
    num_state_relevance_samples=None,
    policy_grid_size=None,
    policy_noise_batch_size=None,
    num_policy_trajectories=None,
    policy_horizon=None,
    initial_state=None,
    probe_states=None,
    example_config: PolynomialALPExampleConfig | None = None,
    inventory_config: InventoryMDPConfig | None = None,
    policy_config: PolicyEvaluationConfig | None = None,
):
    """
    Run the tutorial's small polynomial sampled-ALP example across seeds.

    This helper keeps the `how-code-works.ipynb` example readable while still
    exposing the full construct-optimize-evaluate workflow through its
    parameter list and printed summary table.

    Args:
        seeds: Random seeds controlling sampled constraints, relevance states,
            demand batches, lookahead samples, and simulation paths.
        demand_samples_per_constraint: Demand draws used in each Bellman
            expectation approximation.
        action_step: Discrete action-grid spacing used by policy lookup.
        polynomial_exponents: Exponents defining the polynomial VFA basis.
        num_constraints: Number of sampled Bellman inequalities.
        num_state_relevance_samples: Number of sampled objective states before
            boundary/reference states are added.
        policy_grid_size: Number of states in the greedy-policy lookup grid.
        policy_noise_batch_size: Demand draws used for one-step lookahead.
        num_policy_trajectories: Number of simulated policy-evaluation paths.
        policy_horizon: Number of periods simulated in each path.
        initial_state: Initial inventory state for policy evaluation.
        probe_states: States at which to report greedy actions.
    """
    example_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.polynomial_alp if example_config is None else example_config
    inventory_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.inventory if inventory_config is None else inventory_config
    policy_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.policy_evaluation if policy_config is None else policy_config

    seeds = example_config.seeds if seeds is None else seeds
    demand_samples_per_constraint = (
        inventory_config.num_noise_samples
        if demand_samples_per_constraint is None
        else demand_samples_per_constraint
    )
    action_step = inventory_config.action_step if action_step is None else action_step
    polynomial_exponents = (
        example_config.polynomial_exponents
        if polynomial_exponents is None
        else polynomial_exponents
    )
    num_constraints = example_config.num_constraints if num_constraints is None else num_constraints
    num_state_relevance_samples = (
        example_config.num_state_relevance_samples
        if num_state_relevance_samples is None
        else num_state_relevance_samples
    )
    policy_grid_size = example_config.policy_grid_size if policy_grid_size is None else policy_grid_size
    policy_noise_batch_size = (
        policy_config.policy_noise_batch_size
        if policy_noise_batch_size is None
        else policy_noise_batch_size
    )
    num_policy_trajectories = (
        policy_config.num_trajectories
        if num_policy_trajectories is None
        else num_policy_trajectories
    )
    policy_horizon = policy_config.horizon if policy_horizon is None else policy_horizon
    initial_state = policy_config.initial_state if initial_state is None else initial_state
    probe_states = example_config.probe_states if probe_states is None else probe_states
    probe_states = np.asarray(probe_states, dtype=float)

    def solve_one_seed(seed):
        start_time = time.time()
        mdp = make_inventory_mdp(
            inventory_config.with_updates(
                num_noise_samples=demand_samples_per_constraint,
                action_step=action_step,
                random_seed=seed,
            )
        )

        from basis import PolynomialBasis1D

        basis = PolynomialBasis1D(exponents=polynomial_exponents)
        state_samples, action_samples = mdp.sample_constraint_state_actions(num_constraints)
        relevance_states = mdp.sample_state_relevance_states(num_state_relevance_samples)

        constraint_rows = []
        rhs_values = []
        for state, action in zip(state_samples, action_samples):
            phi_state = basis.eval_basis(state)
            next_states = mdp.get_batch_next_state(state, action)
            expected_phi_next = basis.expected_basis(next_states)
            expected_cost = mdp.get_expected_cost(state, action)
            constraint_rows.append(phi_state - mdp.discount * expected_phi_next)
            rhs_values.append(expected_cost)

        A = np.asarray(constraint_rows, dtype=float)
        b = np.asarray(rhs_values, dtype=float)
        c = np.mean([basis.eval_basis(state) for state in relevance_states], axis=0)

        result = linprog(c=-c, A_ub=A, b_ub=b, bounds=[(None, None)] * len(c), method="highs")
        if not result.success:
            raise RuntimeError(result.message)

        coef = np.asarray(result.x, dtype=float)
        alp_objective = float(c @ coef)
        sampled_slacks = b - A @ coef
        min_sampled_slack = 0.0 if abs(sampled_slacks.min()) < 1e-8 else float(sampled_slacks.min())

        alp_model = SimpleNamespace(
            mdp=mdp,
            basis=basis,
            coef=coef,
            num_random_features=len(polynomial_exponents) - 1,
        )
        policy_config = PolicyEvaluationConfig(
            state_grid_size=policy_grid_size,
            policy_noise_batch_size=policy_noise_batch_size,
            policy_noise_seed=seed + 510000,
            num_trajectories=num_policy_trajectories,
            horizon=policy_horizon,
            simulation_seed=seed + 10000,
            initial_state=initial_state,
        )
        policy_cost, policy_se = estimate_upper_bound_fast(alp_model, config=policy_config, return_se=True)
        state_grid, policy_actions = _build_greedy_policy_lookup(alp_model, config=policy_config)
        probe_actions = [policy_actions[np.abs(state_grid - state).argmin()] for state in probe_states]
        elapsed_time = time.time() - start_time

        return {
            "seed": seed,
            "status": result.message,
            "coef": coef,
            "alp_objective": alp_objective,
            "policy_cost": policy_cost,
            "policy_se": policy_se,
            "gap": _compute_optimality_gap(alp_objective, policy_cost),
            "runtime_sec": elapsed_time,
            "min_sampled_slack": min_sampled_slack,
            "binding_constraints": int((sampled_slacks <= 1e-6).sum()),
            "num_relevance_states": len(relevance_states),
            "probe_actions": probe_actions,
        }

    rows = [solve_one_seed(seed) for seed in seeds]

    mean_alp_objective = np.mean([row["alp_objective"] for row in rows])
    mean_policy_cost = np.mean([row["policy_cost"] for row in rows])
    mean_gap = np.mean([row["gap"] for row in rows])
    mean_runtime = np.mean([row["runtime_sec"] for row in rows])

    table_width = 121
    print()
    print("=" * table_width)
    print(
        f"{'seed':>8} {'ALP obj':>16} {'policy cost':>16} "
        f"{'diff %':>12} {'bind constr':>12} {'min slack':>12} {'time (sec)':>12}"
    )
    print("-" * table_width)
    for row in rows:
        print(
            f"{row['seed']:8d} "
            f"{_format_table_value(row['alp_objective'], width=16, precision=1)} "
            f"{_format_table_value(row['policy_cost'], width=16, precision=1)} "
            f"{_format_table_value(row['gap'], width=12, precision=1)} "
            f"{row['binding_constraints']:12d} "
            f"{row['min_sampled_slack']:12.4f} "
            f"{row['runtime_sec']:12.2f}"
        )
        print("-" * table_width)
    print(
        f"{'AVERAGE':>8} "
        f"{_format_table_value(mean_alp_objective, width=16, precision=1)} "
        f"{_format_table_value(mean_policy_cost, width=16, precision=1)} "
        f"{_format_table_value(mean_gap, width=12, precision=1)} "
        f"{'':>12} "
        f"{'':>12} "
        f"{mean_runtime:12.2f}"
    )
    print("=" * table_width)
    print()
    print("Shared ALP example settings")
    print("---------------------------")
    print(f"sampled constraints       : {num_constraints}")
    print(f"demand samples/constraint: {demand_samples_per_constraint}")
    print(f"state-relevance states   : {rows[0]['num_relevance_states']}")
    print("basis                    : [1, s, s^2]")
    print(f"policy lookup states     : {policy_grid_size}")
    print(f"lookahead noise draws    : {policy_noise_batch_size}")
    print(f"simulation paths         : {num_policy_trajectories}")
    print(f"simulation horizon       : {policy_horizon}")
    print(f"initial state            : {initial_state}")
    print()
    print("Greedy policy sample actions by seed")
    print("------------------------------------")
    header = "seed".rjust(8) + "".join([f"{state:>10.1f}" for state in probe_states])
    print(header)
    print("-" * len(header))
    for row in rows:
        action_text = "".join([f"{action:>10.1f}" for action in row["probe_actions"]])
        print(f"{row['seed']:8d}{action_text}")

    return rows


def moving_average_smoother(x, y, window_size=81):
    """
    Smooth a curve with a centered moving average.

    Args:
        x: Grid points associated with the curve.
        y: Function values to smooth.
        window_size: Number of neighboring points used in the moving average.
    """
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    y_smooth = np.zeros_like(y, dtype=float)

    for i in range(len(y)):
        left = max(0, i - half_window)
        right = min(len(y), i + half_window + 1)
        y_smooth[i] = np.mean(y[left:right])

    return x, y_smooth


def run_falp_grid(
    feature_counts,
    seeds=None,
    num_constraints=None,
    num_state_relevance_samples=None,
    bandwidth_choices=None,
    compute_upper_bound=True,
    lower_bound_num_mc_init_states=None,
    lower_bound_burn_in=None,
    lower_bound_noise_batch_size=None,
    lower_bound_sampler=None,
    lower_bound_num_walkers=None,
    lower_bound_chain_length=None,
    upper_bound_num_trajectories=None,
    upper_bound_horizon=None,
    lower_bound_num_mc_samples=None,
    falp_config: FALPConfig | None = None,
    lower_bound_config: LowerBoundConfig | None = None,
    policy_config: PolicyEvaluationConfig | None = None,
    inventory_config: InventoryMDPConfig | None = None,
):
    """
    Fit FALP once for each (#features, seed) pair and cache the outputs.

    The legacy scalar arguments are still supported, but the cleaner pattern is
    to pass `falp_config`, `lower_bound_config`, and `policy_config`.

    Args:
        feature_counts: Numbers of nonconstant random features to fit.
        seeds: Random seeds used for the feature family and evaluation.
        num_constraints: Number of sampled Bellman inequalities.
        num_state_relevance_samples: Number of states used in the ALP objective.
        bandwidth_choices: Candidate bandwidths used in feature sampling.
        compute_upper_bound: Whether to estimate the policy-cost upper bound.
        lower_bound_num_mc_init_states: Number of initial particles or walkers
            used in lower-bound estimation.
        lower_bound_burn_in: Number of initial MCMC steps discarded.
        lower_bound_noise_batch_size: Number of demand draws reused in residual
            estimates.
        lower_bound_sampler: Lower-bound sampler backend.
        lower_bound_num_walkers: Number of walkers used by the optional
            `emcee` sampler.
        lower_bound_chain_length: Number of MCMC steps per chain.
        upper_bound_num_trajectories: Number of policy-evaluation trajectories.
        upper_bound_horizon: Number of periods simulated in each trajectory.
        lower_bound_num_mc_samples: Backward-compatible alias for
            `lower_bound_num_mc_init_states`.
        falp_config: Optional grouped FALP settings.
        lower_bound_config: Optional grouped lower-bound settings.
        policy_config: Optional grouped policy-evaluation settings.
    """
    if lower_bound_num_mc_samples is not None:
        lower_bound_num_mc_init_states = lower_bound_num_mc_samples

    seeds = CONTINUOUS_MDP_NOTEBOOK_CONFIG.seeds if seeds is None else seeds
    inventory_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.inventory if inventory_config is None else inventory_config
    default_falp_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.falp
    num_constraints = default_falp_config.num_constraints if num_constraints is None else num_constraints
    num_state_relevance_samples = (
        default_falp_config.num_state_relevance_samples
        if num_state_relevance_samples is None
        else num_state_relevance_samples
    )
    bandwidth_choices = (
        default_falp_config.random_features.bandwidth_choices
        if bandwidth_choices is None
        else bandwidth_choices
    )
    default_policy_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.policy_evaluation
    default_lower_bound_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.lower_bound
    lower_bound_num_mc_init_states = (
        default_lower_bound_config.num_mc_init_states
        if lower_bound_num_mc_init_states is None
        else lower_bound_num_mc_init_states
    )
    lower_bound_burn_in = default_lower_bound_config.burn_in if lower_bound_burn_in is None else lower_bound_burn_in
    lower_bound_noise_batch_size = (
        default_lower_bound_config.noise_batch_size
        if lower_bound_noise_batch_size is None
        else lower_bound_noise_batch_size
    )
    lower_bound_sampler = default_lower_bound_config.sampler if lower_bound_sampler is None else lower_bound_sampler
    lower_bound_num_walkers = (
        default_lower_bound_config.num_walkers
        if lower_bound_num_walkers is None
        else lower_bound_num_walkers
    )
    lower_bound_chain_length = (
        default_lower_bound_config.chain_length
        if lower_bound_chain_length is None
        else lower_bound_chain_length
    )
    upper_bound_num_trajectories = (
        default_policy_config.num_trajectories
        if upper_bound_num_trajectories is None
        else upper_bound_num_trajectories
    )
    upper_bound_horizon = default_policy_config.horizon if upper_bound_horizon is None else upper_bound_horizon
    results = {}

    base_falp_config = _make_falp_config(
        config=falp_config,
        num_random_features=1,
        num_constraints=num_constraints,
        num_state_relevance_samples=num_state_relevance_samples,
        basis_seed=default_falp_config.random_features.random_seed,
        bandwidth_choices=bandwidth_choices,
        solver=default_falp_config.solver,
    )
    base_lower_bound_config = _make_lower_bound_config(
        config=lower_bound_config,
        num_mc_init_states=lower_bound_num_mc_init_states,
        chain_length=lower_bound_chain_length,
        burn_in=lower_bound_burn_in,
        proposal_state_std=default_lower_bound_config.proposal_state_std,
        proposal_action_std=default_lower_bound_config.proposal_action_std,
        random_seed=default_lower_bound_config.random_seed,
        noise_batch_size=lower_bound_noise_batch_size,
        sampler=lower_bound_sampler,
        num_walkers=lower_bound_num_walkers,
        initial_state=default_lower_bound_config.initial_state,
    )
    base_policy_config = _make_policy_config(
        config=policy_config,
        state_grid_size=default_policy_config.state_grid_size,
        policy_noise_batch_size=default_policy_config.policy_noise_batch_size,
        policy_noise_seed=default_policy_config.policy_noise_seed,
        num_trajectories=upper_bound_num_trajectories,
        horizon=upper_bound_horizon,
        simulation_seed=default_policy_config.simulation_seed,
        initial_state=default_policy_config.initial_state,
    )

    def fmt(value, width=16, precision=1):
        """
        Format one table entry for the FALP progress display.

        Args:
            value: Value to display.
            width: Display width used in the printed table.
            precision: Number of decimal digits for numeric values.
        """
        return _format_table_value(value, width=width, precision=precision)

    table_width = 138

    print("=" * table_width)
    print(
        f"{'seed':>8} {'# features':>12} "
        f"{'FALP obj':>16} {'CVL lb':>16} {'policy cost':>16} "
        f"{'best lb':>16} {'best ub':>16} {'opt gap %':>12} {'time (sec)':>12}"
    )
    print("-" * table_width)

    for seed in seeds:
        best_lower_bound = None
        best_upper_bound = None
        best_gap = None

        for m in feature_counts:
            if m not in results:
                results[m] = {}
            start_time = time.time()

            falp_objective = None
            cvl_lower_bound = None
            upper_bound = None
            gap = None

            def print_progress(end="\r"):
                """
                Print the current FALP progress row.

                Args:
                    end: Line ending used by the progress print.
                """
                elapsed_time = time.time() - start_time
                policy_cost_str = fmt(upper_bound, width=16, precision=1) if upper_bound is not None else f"{'...':>16}"
                best_lb_str = (
                    fmt(best_lower_bound, width=16, precision=1) if best_lower_bound is not None else f"{'...':>16}"
                )
                best_ub_str = (
                    fmt(best_upper_bound, width=16, precision=1) if best_upper_bound is not None else f"{'...':>16}"
                )
                gap_str = fmt(best_gap, width=12, precision=1) if best_gap is not None else f"{'...':>12}"
                print(
                    f"{seed:8d} {m:12d} "
                    f"{fmt(falp_objective, width=16, precision=1)} "
                    f"{fmt(cvl_lower_bound, width=16, precision=1)} "
                    f"{policy_cost_str} {best_lb_str} {best_ub_str} {gap_str} {elapsed_time:12.2f}",
                    end=end,
                    flush=True,
                )

            print_progress()

            model_config = base_falp_config.with_updates(
                num_random_features=m,
                random_features=base_falp_config.random_features.with_updates(random_seed=seed),
            )
            model = FALP(
                mdp=make_inventory_mdp(inventory_config),
                config=model_config,
            )
            solution = model.fit()

            falp_objective = estimate_falp_objective(model)
            print_progress()

            lb_config = base_lower_bound_config.with_updates(random_seed=seed + 20000)
            cvl_lower_bound = estimate_cvl_lower_bound(model, config=lb_config)
            best_lower_bound = (
                cvl_lower_bound
                if best_lower_bound is None
                else max(best_lower_bound, cvl_lower_bound)
            )
            print_progress()

            if compute_upper_bound:
                ub_config = base_policy_config.with_updates(
                    policy_noise_seed=seed + 510000,
                    simulation_seed=seed + 10000,
                )
                upper_bound = estimate_upper_bound_fast(model, config=ub_config)
                best_upper_bound = (
                    upper_bound
                    if best_upper_bound is None
                    else min(best_upper_bound, upper_bound)
                )
                best_gap = _compute_optimality_gap(best_lower_bound, best_upper_bound)
                print_progress()
                gap = best_gap

            elapsed_time = time.time() - start_time

            results[m][seed] = {
                "model": model,
                "solution": solution,
                "falp_objective": falp_objective,
                "lower_bound": cvl_lower_bound,
                "upper_bound": upper_bound,
                "best_lower_bound": best_lower_bound,
                "best_upper_bound": best_upper_bound,
                "gap": gap,
                "runtime_sec": elapsed_time,
            }

            print_progress(end="\n")
        print("-" * table_width)

    print("=" * table_width)
    return results


def extract_boxplot_stats(results_dict, feature_counts, seeds):
    """
    Repackage lower bounds, upper bounds, and gaps for boxplot calls.

    Args:
        results_dict: Nested dictionary returned by a grid runner.
        feature_counts: Feature counts to extract in display order.
        seeds: Random seeds to extract in display order.
    """
    lower = {m: [results_dict[m][seed]["lower_bound"] for seed in seeds] for m in feature_counts}
    upper = {m: [results_dict[m][seed]["upper_bound"] for seed in seeds] for m in feature_counts}
    gap = {m: [results_dict[m][seed]["gap"] for seed in seeds] for m in feature_counts}
    return lower, upper, gap


def _boxplot_style_kwargs():
    """
    Return shared styling used by tutorial boxplots.

    Outliers are shown with `+` markers, and whiskers/caps use black so the
    vertical spread stands out clearly for non-expert readers.
    """
    return {
        "patch_artist": True,
        "showfliers": True,
        "flierprops": {
            "marker": "+",
            "markeredgecolor": "black",
            "markerfacecolor": "none",
            "markersize": 7,
            "linestyle": "none",
        },
        "whiskerprops": {"color": "black", "linewidth": 1.2},
        "capprops": {"color": "black", "linewidth": 1.2},
        "boxprops": {"edgecolor": "black", "linewidth": 1.1},
    }


def _draw_boxplot_min_max_lines(ax, positions, data, linewidth=1.2):
    """
    Draw a black line from the minimum to the maximum value for each boxplot.

    Args:
        ax: Matplotlib axis receiving the overlays.
        positions: Horizontal positions of the boxplots.
        data: Iterable of one-dimensional numeric samples.
        linewidth: Width of the min-to-max line.
    """
    for position, values in zip(positions, data):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            continue
        ax.vlines(
            position,
            float(np.min(values)),
            float(np.max(values)),
            color="black",
            linewidth=linewidth,
            zorder=1,
        )


def plot_value_function_curves(
    results_dict,
    feature_counts,
    seeds,
    algorithm_name,
    comparison_results_dict=None,
    comparison_feature_counts=None,
    comparison_seeds=None,
    comparison_algorithm_name=None,
    grid_size=300,
    colormap="viridis",
    comparison_colormap=None,
    objective_color="#2a6f97",
    figsize=(12, 3.5),
    fontsize=12,
    plot_seed=None,
    comparison_plot_seed=None,
):
    """
    Plot representative value-function curves and mean objective values.

    When `comparison_results_dict` is provided, the figure instead compares
    value-function curves for two methods side by side. This mode is used for
    the FALP-versus-SGALP VFA comparison after both experiment grids have been
    computed.

    Args:
        results_dict: Nested dictionary returned by `run_falp_grid` or
            `run_sgalp_grid`.
        feature_counts: Feature counts to include in the figure.
        seeds: Random seeds included in `results_dict`.
        algorithm_name: Label used in plot titles, such as `FALP` or `SGALP`.
        comparison_results_dict: Optional second result dictionary to compare
            against `results_dict`.
        comparison_feature_counts: Feature counts for the comparison results.
            Defaults to `feature_counts`.
        comparison_seeds: Seeds for the comparison results. Defaults to
            `seeds`.
        comparison_algorithm_name: Label for the comparison method.
        grid_size: Number of states used in the plotting grid.
        colormap: Matplotlib colormap name used for the value curves.
        comparison_colormap: Colormap for the comparison value curves.
        objective_color: Line color used for the objective summary plot.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
        plot_seed: Seed used for the representative value-function curves.
        comparison_plot_seed: Seed used for the comparison curves.
    """
    if plot_seed is None:
        plot_seed = seeds[0]

    def _plot_vfa_panel(ax, result_set, counts, seed, label, cmap_name):
        representative_model = result_set[counts[0]][seed]["model"]
        grid = np.linspace(
            representative_model.mdp.lower_state_bound,
            representative_model.mdp.upper_state_bound,
            grid_size,
        )
        colors = plt.get_cmap(cmap_name)(np.linspace(0.15, 0.9, len(counts)))

        for color, feature_count in zip(colors, counts):
            model = result_set[feature_count][seed]["model"]
            values = evaluate_vfa_on_grid(model, grid)
            shift = max(0.0, 1.0 - np.min(values))
            log_values = np.log(values + shift)
            curve_label = f"N = {feature_count}" + (f" (shift={shift:.2f})" if shift > 0 else "")
            ax.plot(grid, log_values, linewidth=2.3, color=color, label=curve_label)

        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set(
            xlabel="State s",
            ylabel=r"Log value function approximation",
        )
        ax.tick_params(labelsize=fontsize)
        ax.legend(ncol=2, fontsize=fontsize)
        return grid

    if comparison_results_dict is not None:
        comparison_feature_counts = feature_counts if comparison_feature_counts is None else comparison_feature_counts
        comparison_seeds = seeds if comparison_seeds is None else comparison_seeds
        comparison_algorithm_name = (
            "Comparison"
            if comparison_algorithm_name is None
            else comparison_algorithm_name
        )
        comparison_colormap = colormap if comparison_colormap is None else comparison_colormap
        if comparison_plot_seed is None:
            comparison_plot_seed = comparison_seeds[0]

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        _plot_vfa_panel(axes[0], results_dict, feature_counts, plot_seed, algorithm_name, colormap)
        _plot_vfa_panel(
            axes[1],
            comparison_results_dict,
            comparison_feature_counts,
            comparison_plot_seed,
            comparison_algorithm_name,
            comparison_colormap,
        )
        plt.tight_layout()
        plt.show()
        return

    representative_model = results_dict[feature_counts[0]][plot_seed]["model"]
    grid = np.linspace(
        representative_model.mdp.lower_state_bound,
        representative_model.mdp.upper_state_bound,
        grid_size,
    )
    mean_objective_values = []
    colors = plt.get_cmap(colormap)(np.linspace(0.15, 0.9, len(feature_counts)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for color, feature_count in zip(colors, feature_counts):
        model = results_dict[feature_count][plot_seed]["model"]
        mean_objective_values.append(
            np.mean([results_dict[feature_count][seed]["solution"]["objective_value"] for seed in seeds])
        )

        values = evaluate_vfa_on_grid(model, grid)
        shift = max(0.0, 1.0 - np.min(values))
        log_values = np.log(values + shift)

        label = f"N = {feature_count}" + (f" (shift={shift:.2f})" if shift > 0 else "")
        axes[0].plot(grid, log_values, linewidth=2.3, color=color, label=label)

    axes[0].axvline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="State s",
        ylabel=r"Log value function approximation",
        # title=f"{algorithm_name} Value Functions on a Log Scale (seed = {plot_seed})",
    )
    axes[0].legend(ncol=2)
    axes[0].tick_params(labelsize=fontsize)

    axes[1].plot(
        feature_counts,
        mean_objective_values,
        "o--",
        linewidth=2.2,
        markersize=7,
        color=objective_color,
    )
    axes[1].set(
        xlabel="Number of Random Features (m)",
        ylabel=f"Mean {algorithm_name} Objective Value",
        title=f"Average {algorithm_name} Objective vs. Basis Size",
    )
    axes[1].tick_params(labelsize=fontsize)

    plt.tight_layout()
    plt.show()


def plot_bound_boxplots(
    results_dict,
    feature_counts,
    seeds,
    algorithm_name,
    figsize=(12, 3.5),
    fontsize=12,
):
    """
    Plot lower bounds, upper bounds, and optimality gaps across seeds.

    Args:
        results_dict: Nested dictionary returned by a grid runner.
        feature_counts: Feature counts to include in the figure.
        seeds: Random seeds included in `results_dict`.
        algorithm_name: Label used in plot titles, such as `FALP` or `SGALP`.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
    """
    lower_bound_results, upper_bound_results, gap_results = extract_boxplot_stats(results_dict, feature_counts, seeds)

    positions = np.arange(1, len(feature_counts) + 1)
    lb_data = [lower_bound_results[m] for m in feature_counts]
    ub_data = [upper_bound_results[m] for m in feature_counts]
    gap_data = [gap_results[m] for m in feature_counts]

    lb_means = [np.mean(values) for values in lb_data]
    ub_means = [np.mean(values) for values in ub_data]
    gap_means = [np.mean(values) for values in gap_data]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    boxplot_kwargs = _boxplot_style_kwargs()

    bp_lb = axes[0].boxplot(lb_data, positions=positions, widths=0.3, manage_ticks=False, **boxplot_kwargs)
    bp_ub = axes[0].boxplot(ub_data, positions=positions, widths=0.3, manage_ticks=False, **boxplot_kwargs)
    bp_gap = axes[1].boxplot(gap_data, positions=positions, tick_labels=[str(m) for m in feature_counts], **boxplot_kwargs)

    _draw_boxplot_min_max_lines(axes[0], positions, lb_data)
    _draw_boxplot_min_max_lines(axes[0], positions, ub_data)
    _draw_boxplot_min_max_lines(axes[1], positions, gap_data)

    for median in bp_lb["medians"] + bp_ub["medians"] + bp_gap["medians"]:
        median.set_visible(False)

    for patch in bp_lb["boxes"]:
        patch.set_facecolor("#9bd0f5")
    for patch in bp_ub["boxes"]:
        patch.set_facecolor("#f7a072")
    for patch in bp_gap["boxes"]:
        patch.set_facecolor("#d9d9d9")

    axes[0].plot(positions, lb_means, "o--", linewidth=1.8, markersize=6, color="#1565c0", label="Mean lower bound")
    axes[0].plot(positions, ub_means, "o--", linewidth=1.8, markersize=6, color="#d9480f", label="Mean upper bound")
    axes[0].set(
        xticks=positions,
        xticklabels=[str(m) for m in feature_counts],
        xlabel="Number of Random Features (m)",
        ylabel="Bound Value",
        title=f"{algorithm_name} Lower and Upper Bounds Across Random Seeds",
    )
    axes[0].tick_params(labelsize=fontsize)
    axes[0].legend(fontsize=fontsize)

    axes[1].plot(positions, gap_means, "o--", linewidth=1.8, markersize=7, color="black", label="Mean optimality gap")
    axes[1].set(
        xlabel="Number of Random Features (m)",
        ylabel="Optimality Gap (%)",
        title=f"{algorithm_name} Optimality Gap Across Random Seeds",
    )
    axes[1].tick_params(labelsize=fontsize)
    axes[1].legend(fontsize=fontsize)

    plt.tight_layout()
    plt.show()


def plot_falp_vs_sgalp_bound_boxplots(
    falp_results,
    sgalp_results,
    feature_counts,
    seeds,
    figsize=(12, 3.5),
    fontsize=12,
):
    """
    Plot FALP and SGALP lower/upper bound boxplots side by side.

    This comparison intentionally reuses the same boxplot configuration as
    `plot_bound_boxplots`: visible plus-sign outliers, hidden medians,
    min-to-max black line overlays, and the same lower/upper bound colors.

    Args:
        falp_results: Nested dictionary returned by `run_falp_grid`.
        sgalp_results: Nested dictionary returned by `run_sgalp_grid`.
        feature_counts: Feature or stage counts to include in display order.
        seeds: Random seeds included in both result dictionaries.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    boxplot_kwargs = _boxplot_style_kwargs()
    positions = np.arange(1, len(feature_counts) + 1)

    for ax, results, title in [
        (axes[0], falp_results, "FALP bounds"),
        (axes[1], sgalp_results, "SGALP bounds"),
    ]:
        lower_bound_results, upper_bound_results, _ = extract_boxplot_stats(results, feature_counts, seeds)
        lb_data = [lower_bound_results[m] for m in feature_counts]
        ub_data = [upper_bound_results[m] for m in feature_counts]
        lb_means = [np.mean(values) for values in lb_data]
        ub_means = [np.mean(values) for values in ub_data]

        bp_lb = ax.boxplot(lb_data, positions=positions, widths=0.3, manage_ticks=False, **boxplot_kwargs)
        bp_ub = ax.boxplot(ub_data, positions=positions, widths=0.3, manage_ticks=False, **boxplot_kwargs)

        _draw_boxplot_min_max_lines(ax, positions, lb_data)
        _draw_boxplot_min_max_lines(ax, positions, ub_data)

        for median in bp_lb["medians"] + bp_ub["medians"]:
            median.set_visible(False)
        for patch in bp_lb["boxes"]:
            patch.set_facecolor("#9bd0f5")
        for patch in bp_ub["boxes"]:
            patch.set_facecolor("#f7a072")

        ax.plot(positions, lb_means, "o--", linewidth=1.8, markersize=6, color="#1565c0", label="Mean lower bound")
        ax.plot(positions, ub_means, "o--", linewidth=1.8, markersize=6, color="#d9480f", label="Mean upper bound")
        ax.set(
            xticks=positions,
            xticklabels=[str(m) for m in feature_counts],
            xlabel="Number of basis functions",
            # title=title,
        )
        ax.tick_params(labelsize=fontsize)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(
            [bp_lb["boxes"][0], bp_ub["boxes"][0]],
            ["Lower bound", "Upper bound"],
            loc="best",
            fontsize=fontsize,
        )

    axes[0].set_ylabel("Estimated discounted cost", fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def plot_psmd_iteration_diagnostics(
    psmd_results,
    seeds,
    min_iteration=200,
    figsize=(12, 3.5),
    fontsize=12,
):
    """
    Plot PSMD lower-bound and policy-cost traces across seeds.

    Args:
        psmd_results: Dictionary returned by `run_psmd_seed_grid`.
        seeds: Random seeds to include in the multi-seed comparison.
        min_iteration: Earliest iteration shown on the x-axis.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
    """
    histories_by_seed = {
        seed: [row for row in psmd_results[seed]["solution"]["history"] if row["iteration"] >= min_iteration]
        for seed in seeds
    }
    iterations = sorted({row["iteration"] for history in histories_by_seed.values() for row in history})

    current_lb_data = []
    current_pc_data = []
    best_lb_data = []
    best_pc_data = []
    for iteration in iterations:
        rows = [row for history in histories_by_seed.values() for row in history if row["iteration"] == iteration]
        current_lb_data.append(np.asarray([row["lower_bound"] for row in rows], dtype=float))
        current_pc_data.append(np.asarray([row["policy_cost"] for row in rows], dtype=float))
        best_lb_data.append(np.asarray([row["best_lower_bound"] for row in rows], dtype=float))
        best_pc_data.append(np.asarray([row["best_policy_cost"] for row in rows], dtype=float))

    current_lb_means = [values.mean() for values in current_lb_data]
    current_pc_means = [values.mean() for values in current_pc_data]
    best_lb_means = [values.mean() for values in best_lb_data]
    best_pc_means = [values.mean() for values in best_pc_data]

    positions = np.asarray(iterations, dtype=float)
    offset = 0.0
    width = 24.0

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    boxplot_kwargs = _boxplot_style_kwargs()

    bp_current_lb = axes[0].boxplot(current_lb_data, positions=positions - offset, widths=width, manage_ticks=False, **boxplot_kwargs)
    bp_current_pc = axes[0].boxplot(current_pc_data, positions=positions + offset, widths=width, manage_ticks=False, **boxplot_kwargs)
    bp_best_lb = axes[1].boxplot(best_lb_data, positions=positions - offset, widths=width, manage_ticks=False, **boxplot_kwargs)
    bp_best_pc = axes[1].boxplot(best_pc_data, positions=positions + offset, widths=width, manage_ticks=False, **boxplot_kwargs)

    _draw_boxplot_min_max_lines(axes[0], positions - offset, current_lb_data)
    _draw_boxplot_min_max_lines(axes[0], positions + offset, current_pc_data)
    _draw_boxplot_min_max_lines(axes[1], positions - offset, best_lb_data)
    _draw_boxplot_min_max_lines(axes[1], positions + offset, best_pc_data)

    for patch in bp_current_lb["boxes"] + bp_best_lb["boxes"]:
        patch.set(facecolor="#5dade2", edgecolor="#1f4e79", alpha=0.8)
    for patch in bp_current_pc["boxes"] + bp_best_pc["boxes"]:
        patch.set(facecolor="#f1948a", edgecolor="#922b21", alpha=0.8)
    for median in bp_current_lb["medians"] + bp_current_pc["medians"] + bp_best_lb["medians"] + bp_best_pc["medians"]:
        median.set(alpha=0.0, linewidth=0.0)

    for x_position, values in zip(positions - offset, current_lb_data):
        axes[0].scatter(np.full(len(values), x_position), values, color="#1f4e79", s=28, zorder=3)
    for x_position, values in zip(positions + offset, current_pc_data):
        axes[0].scatter(np.full(len(values), x_position), values, color="#922b21", s=28, zorder=3)
    for x_position, values in zip(positions - offset, best_lb_data):
        axes[1].scatter(np.full(len(values), x_position), values, color="#1f4e79", s=28, zorder=3)
    for x_position, values in zip(positions + offset, best_pc_data):
        axes[1].scatter(np.full(len(values), x_position), values, color="#922b21", s=28, zorder=3)

    axes[0].plot(positions - offset, current_lb_means, "o-", linewidth=2.2, markersize=5, color="#1f4e79", zorder=4)
    axes[0].plot(positions + offset, current_pc_means, "o-", linewidth=2.2, markersize=5, color="#922b21", zorder=4)
    axes[1].plot(positions - offset, best_lb_means, "o-", linewidth=2.2, markersize=5, color="#1f4e79", zorder=4)
    axes[1].plot(positions + offset, best_pc_means, "o-", linewidth=2.2, markersize=5, color="#922b21", zorder=4)

    axes[0].set(
        xticks=iterations,
        xticklabels=[str(int(t)) for t in iterations],
        xlabel="Iteration",
        ylabel="Value",
        title="(a) Bounds at the Current Evaluated Iterate",
    )
    axes[0].tick_params(labelsize=fontsize)
    axes[0].grid(alpha=0.25)
    axes[0].legend(
        [bp_current_lb["boxes"][0], bp_current_pc["boxes"][0]],
        ["PSMD lower bound", "PSMD policy cost"],
        loc="lower right",
        fontsize=fontsize,
    )

    axes[1].set(
        xticks=iterations,
        xticklabels=[str(int(t)) for t in iterations],
        xlabel="Iteration",
        title="(b) Best Historical Bounds",
    )
    axes[1].tick_params(labelsize=fontsize)
    axes[1].grid(alpha=0.25)
    axes[1].legend(
        [bp_best_lb["boxes"][0], bp_best_pc["boxes"][0]],
        ["PSMD lower bound", "PSMD policy cost"],
        loc="lower right",
        fontsize=fontsize,
    )

    plt.tight_layout()
    plt.show()


def plot_psmd_acceptance_and_value(
    psmd_result,
    plot_seed,
    grid_size=300,
    figsize=(12, 3.5),
    fontsize=12,
):
    """
    Plot PSMD sampler acceptance rates and the final averaged value curve.

    Args:
        psmd_result: One seed entry from `run_psmd_seed_grid`.
        plot_seed: Seed label shown in the titles.
        grid_size: Number of states used in the value-function grid.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
    """
    solver = psmd_result["model"]
    solution = psmd_result["solution"]
    history = solution["history"]

    state_grid = np.linspace(solver.mdp.lower_state_bound, solver.mdp.upper_state_bound, grid_size)
    avg_coef = solution["avg_coef"]
    vfa_values = np.asarray(
        [
            solver.basis.get_vfa(
                np.asarray([state_value], dtype=float),
                avg_coef,
                num_random_features=solver.num_random_features,
            )
            for state_value in state_grid
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(
        [row["iteration"] for row in history],
        [row["acceptance_rate"] for row in history],
        "o--",
        color="#2a6f97",
        linewidth=2,
    )
    axes[0].set(
        xlabel="Iteration",
        ylabel="Acceptance Rate",
        title=f"MH Sampler Acceptance Rate (seed = {plot_seed})",
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(labelsize=fontsize)

    axes[1].plot(state_grid, vfa_values, color="#5b2c6f", linewidth=2.5)
    axes[1].axvline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="State s",
        ylabel=r"$\hat V(s)$",
        title=f"Final Averaged PSMD Value Approximation (seed = {plot_seed})",
    )
    axes[1].tick_params(labelsize=fontsize)

    plt.tight_layout()
    plt.show()


def plot_psmd_sampling_snapshots(
    psmd_result,
    plot_seed,
    ncols=6,
    offset=0.5,
    figsize=(12, 3),
    fontsize=12,
):
    """
    Plot the PSMD state-action sampler cloud at stored iterations.

    Args:
        psmd_result: One seed entry from `run_psmd_seed_grid`.
        plot_seed: Seed label shown in the title.
        ncols: Number of one-row snapshot panels to show.
        offset: Extra margin added around the feasible state-action rectangle.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
    """
    solver = psmd_result["model"]
    solution = psmd_result["solution"]
    snapshot_iterations = [
        iteration
        for iteration in solver.config.snapshot_iterations
        if iteration in solution["state_action_snapshots"]
    ]
    snapshot_iterations = snapshot_iterations[:ncols]

    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, iteration in zip(axes, snapshot_iterations):
        snapshot = solution["state_action_snapshots"][iteration]
        ax.scatter(
            snapshot["states"],
            snapshot["actions"],
            s=50,
            color="#1f4ae0",
            alpha=0.5,
            edgecolors="none",
        )
        ax.set_title(f"Iteration {iteration:,}", fontsize=fontsize)
        ax.set_xlim(solver.mdp.lower_state_bound - offset, solver.mdp.upper_state_bound + offset)
        ax.set_ylim(-offset + 0.0, offset + solver.mdp.max_order)
        ax.set_xticks(np.arange(solver.mdp.lower_state_bound, solver.mdp.upper_state_bound,4))
        ax.set_yticks(np.arange(0, solver.mdp.max_order,2))
        ax.set_xlabel("State", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(True, alpha=0.3)

    if snapshot_iterations:
        axes[0].set_ylabel("Action", fontsize=fontsize)

    for ax in axes[len(snapshot_iterations):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def run_psmd_seed_grid(
    seeds=None,
    psmd_config: PSMDConfig | None = None,
    inventory_config: InventoryMDPConfig | None = None,
    verbose=True,
):
    """
    Fit PSMD once for each random seed and cache the outputs.

    The returned dictionary is keyed by seed so the notebook can choose one
    representative run for iteration-level plots while still reporting
    multi-seed comparisons.

    Args:
        seeds: Random seeds used for PSMD runs.
        psmd_config: Optional grouped PSMD settings.
        verbose: Whether to print progress tables during optimization.
    """

    seeds = CONTINUOUS_MDP_NOTEBOOK_CONFIG.seeds if seeds is None else seeds
    inventory_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.inventory if inventory_config is None else inventory_config
    base_psmd_config = _make_psmd_config(config=psmd_config)
    results = {}

    def fmt(value, width=16, precision=1):
        """
        Format one table entry for the PSMD summary display.

        Args:
            value: Value to display.
            width: Display width used in the printed table.
            precision: Number of decimal digits for numeric values.
        """
        return _format_table_value(value, width=width, precision=precision)

    table_width = 104

    for index, seed in enumerate(seeds):
        run_config = base_psmd_config.with_updates(
            random_seed=seed,
            lower_bound=base_psmd_config.lower_bound.with_updates(random_seed=seed + 20000),
            policy_evaluation=base_psmd_config.policy_evaluation.with_updates(
                policy_noise_seed=seed + 510000,
                simulation_seed=seed + 10000,
            ),
        )

        start_time = time.time()
        model = PSMD(mdp=make_inventory_mdp(inventory_config), config=run_config)
        solution = model.run(
            verbose=verbose,
            show_header=(index == 0),
            show_footer=(index == len(seeds) - 1),
        )
        history = solution["history"]
        final_row = history[-1]
        best_lower_bound = history[-1]["best_lower_bound"]
        best_policy_cost = history[-1]["best_policy_cost"]
        best_gap = _compute_optimality_gap(best_lower_bound, best_policy_cost)
        alp_objective = estimate_psmd_alp_objective(model)
        elapsed_time = time.time() - start_time

        results[seed] = {
            "model": model,
            "solution": solution,
            "alp_objective": alp_objective,
            "current_lower_bound": final_row["lower_bound"],
            "current_upper_bound": final_row["policy_cost"],
            "best_lower_bound": best_lower_bound,
            "best_upper_bound": best_policy_cost,
            "current_gap": (
                _compute_optimality_gap(final_row["lower_bound"], final_row["policy_cost"])
            ),
            "best_gap": best_gap,
            "runtime_sec": elapsed_time,
        }

    # mean_current_lb = np.mean([results[seed]["current_lower_bound"] for seed in seeds])
    # mean_current_ub = np.mean([results[seed]["current_upper_bound"] for seed in seeds])
    # mean_alp_objective = np.mean([results[seed]["alp_objective"] for seed in seeds])
    # mean_best_lb = np.mean([results[seed]["best_lower_bound"] for seed in seeds])
    # mean_best_ub = np.mean([results[seed]["best_upper_bound"] for seed in seeds])
    # mean_best_gap = np.mean([results[seed]["best_gap"] for seed in seeds])
    # mean_runtime = np.mean([results[seed]["runtime_sec"] for seed in seeds])

    # print()
    # print("=" * table_width)
    # print(
    #     f"{'seed':>8} {'ALP obj':>16} {'current lb':>16} {'current ub':>16} "
    #     f"{'best lb':>16} {'best ub':>16} {'best gap %':>12} {'time (sec)':>12}"
    # )
    # print("-" * table_width)
    # for seed in seeds:
    #     print(
    #         f"{seed:8d} "
    #         f"{fmt(results[seed]['alp_objective'], width=16, precision=1)} "
    #         f"{fmt(results[seed]['current_lower_bound'], width=16, precision=1)} "
    #         f"{fmt(results[seed]['current_upper_bound'], width=16, precision=1)} "
    #         f"{fmt(results[seed]['best_lower_bound'], width=16, precision=1)} "
    #         f"{fmt(results[seed]['best_upper_bound'], width=16, precision=1)} "
    #         f"{fmt(results[seed]['best_gap'], width=12, precision=1)} "
    #         f"{results[seed]['runtime_sec']:12.2f}"
    #     )
    #     print("-" * table_width)
    print("=" * table_width)
    return results


def run_sgalp_grid(
    feature_counts,
    sgalp_class=None,
    seeds=None,
    num_constraints=None,
    num_state_relevance_samples=None,
    num_guiding_states=None,
    bandwidth_choices=None,
    guiding_violation=None,
    compute_upper_bound=True,
    lower_bound_num_mc_init_states=None,
    lower_bound_chain_length=None,
    lower_bound_burn_in=None,
    lower_bound_noise_batch_size=None,
    lower_bound_sampler=None,
    lower_bound_num_walkers=None,
    upper_bound_num_trajectories=None,
    upper_bound_horizon=None,
    sgalp_config: SGALPConfig | None = None,
    lower_bound_config: LowerBoundConfig | None = None,
    policy_config: PolicyEvaluationConfig | None = None,
    inventory_config: InventoryMDPConfig | None = None,
):
    """
    Fit SGALP once for each (#features, seed) pair and cache the outputs.

    Unlike the older helper, this version respects the passed parameters
    instead of silently using hard-coded SGALP settings.

    Args:
        feature_counts: Numbers of nonconstant random features to fit.
        sgalp_class: Optional SGALP-compatible class to instantiate.
        seeds: Random seeds used for the feature family and evaluation.
        num_constraints: Number of sampled Bellman inequalities per stage.
        num_state_relevance_samples: Number of states used in the ALP objective.
        num_guiding_states: Number of states used for guiding constraints.
        bandwidth_choices: Candidate bandwidths used in feature sampling.
        guiding_violation: Additive guiding-constraint allowance.
        compute_upper_bound: Whether to estimate the policy-cost upper bound.
        lower_bound_num_mc_init_states: Number of initial particles or walkers
            used in lower-bound estimation.
        lower_bound_chain_length: Number of MCMC steps per chain.
        lower_bound_burn_in: Number of initial MCMC steps discarded.
        lower_bound_noise_batch_size: Number of demand draws reused in residual
            estimates.
        lower_bound_sampler: Lower-bound sampler backend.
        lower_bound_num_walkers: Number of walkers used by the optional
            `emcee` sampler.
        upper_bound_num_trajectories: Number of policy-evaluation trajectories.
        upper_bound_horizon: Number of periods simulated in each trajectory.
        sgalp_config: Optional grouped SGALP settings.
        lower_bound_config: Optional grouped lower-bound settings.
        policy_config: Optional grouped policy-evaluation settings.
    """

    sgalp_class = SelfGuidedALP if sgalp_class is None else sgalp_class
    seeds = CONTINUOUS_MDP_NOTEBOOK_CONFIG.seeds if seeds is None else seeds
    inventory_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.inventory if inventory_config is None else inventory_config
    default_sgalp_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.sgalp
    num_constraints = default_sgalp_config.num_constraints if num_constraints is None else num_constraints
    num_state_relevance_samples = (
        default_sgalp_config.num_state_relevance_samples
        if num_state_relevance_samples is None
        else num_state_relevance_samples
    )
    num_guiding_states = (
        default_sgalp_config.guiding.num_guiding_states
        if num_guiding_states is None
        else num_guiding_states
    )
    bandwidth_choices = (
        default_sgalp_config.random_features.bandwidth_choices
        if bandwidth_choices is None
        else bandwidth_choices
    )
    guiding_violation = (
        default_sgalp_config.guiding.allowed_violation
        if guiding_violation is None
        else guiding_violation
    )
    default_policy_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.policy_evaluation
    default_lower_bound_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.lower_bound
    lower_bound_num_mc_init_states = (
        default_lower_bound_config.num_mc_init_states
        if lower_bound_num_mc_init_states is None
        else lower_bound_num_mc_init_states
    )
    lower_bound_chain_length = (
        default_lower_bound_config.chain_length
        if lower_bound_chain_length is None
        else lower_bound_chain_length
    )
    lower_bound_burn_in = default_lower_bound_config.burn_in if lower_bound_burn_in is None else lower_bound_burn_in
    lower_bound_noise_batch_size = (
        default_lower_bound_config.noise_batch_size
        if lower_bound_noise_batch_size is None
        else lower_bound_noise_batch_size
    )
    lower_bound_sampler = default_lower_bound_config.sampler if lower_bound_sampler is None else lower_bound_sampler
    lower_bound_num_walkers = (
        default_lower_bound_config.num_walkers
        if lower_bound_num_walkers is None
        else lower_bound_num_walkers
    )
    upper_bound_num_trajectories = (
        default_policy_config.num_trajectories
        if upper_bound_num_trajectories is None
        else upper_bound_num_trajectories
    )
    upper_bound_horizon = default_policy_config.horizon if upper_bound_horizon is None else upper_bound_horizon
    results = {}

    base_sgalp_config = _make_sgalp_config(
        config=sgalp_config,
        max_random_features=1,
        batch_size=1,
        num_constraints=num_constraints,
        num_state_relevance_samples=num_state_relevance_samples,
        num_guiding_states=num_guiding_states,
        basis_seed=default_sgalp_config.random_features.random_seed,
        bandwidth_choices=bandwidth_choices,
        guiding_violation=guiding_violation,
        guiding_relax_fraction=default_sgalp_config.guiding.relax_fraction,
        guiding_abs_floor=default_sgalp_config.guiding.absolute_floor,
        guiding_retry_scales=default_sgalp_config.guiding.retry_scales,
        highs_method=default_sgalp_config.solver.method,
        primal_feasibility_tolerance=default_sgalp_config.solver.primal_feasibility_tolerance,
        dual_feasibility_tolerance=default_sgalp_config.solver.dual_feasibility_tolerance,
    )
    base_lower_bound_config = _make_lower_bound_config(
        config=lower_bound_config,
        num_mc_init_states=lower_bound_num_mc_init_states,
        chain_length=lower_bound_chain_length,
        burn_in=lower_bound_burn_in,
        proposal_state_std=default_lower_bound_config.proposal_state_std,
        proposal_action_std=default_lower_bound_config.proposal_action_std,
        random_seed=default_lower_bound_config.random_seed,
        noise_batch_size=lower_bound_noise_batch_size,
        sampler=lower_bound_sampler,
        num_walkers=lower_bound_num_walkers,
        initial_state=default_lower_bound_config.initial_state,
    )
    base_policy_config = _make_policy_config(
        config=policy_config,
        state_grid_size=default_policy_config.state_grid_size,
        policy_noise_batch_size=default_policy_config.policy_noise_batch_size,
        policy_noise_seed=default_policy_config.policy_noise_seed,
        num_trajectories=upper_bound_num_trajectories,
        horizon=upper_bound_horizon,
        simulation_seed=default_policy_config.simulation_seed,
        initial_state=default_policy_config.initial_state,
    )

    def fmt(value, width=16, precision=1):
        """
        Format one table entry for the SGALP progress display.

        Args:
            value: Value to display.
            width: Display width used in the printed table.
            precision: Number of decimal digits for numeric values.
        """
        return _format_table_value(value, width=width, precision=precision)

    table_width = 138

    print("=" * table_width)
    print(
        f"{'seed':>8} {'# features':>12} "
        f"{'SGALP obj':>16} {'CVL lb':>16} {'policy cost':>16} "
        f"{'best lb':>16} {'best ub':>16} {'opt gap %':>12} {'time (sec)':>12}"
    )
    print("-" * table_width)

    for seed in seeds:
        best_lower_bound = None
        best_upper_bound = None
        best_gap = None

        for m in feature_counts:
            if m not in results:
                results[m] = {}
            start_time = time.time()

            sgalp_objective = None
            cvl_lower_bound = None
            upper_bound = None
            gap = None

            def print_progress(end="\r"):
                """
                Print the current SGALP progress row.

                Args:
                    end: Line ending used by the progress print.
                """
                elapsed_time = time.time() - start_time
                policy_cost_str = fmt(upper_bound, width=16, precision=1) if upper_bound is not None else f"{'...':>16}"
                best_lb_str = (
                    fmt(best_lower_bound, width=16, precision=1) if best_lower_bound is not None else f"{'...':>16}"
                )
                best_ub_str = (
                    fmt(best_upper_bound, width=16, precision=1) if best_upper_bound is not None else f"{'...':>16}"
                )
                gap_str = fmt(best_gap, width=12, precision=1) if best_gap is not None else f"{'...':>12}"
                print(
                    f"{seed:8d} {m:12d} "
                    f"{fmt(sgalp_objective, width=16, precision=1)} "
                    f"{fmt(cvl_lower_bound, width=16, precision=1)} "
                    f"{policy_cost_str} {best_lb_str} {best_ub_str} {gap_str} {elapsed_time:12.2f}",
                    end=end,
                    flush=True,
                )

            print_progress()

            model_config = base_sgalp_config.with_updates(
                max_random_features=m,
                random_features=base_sgalp_config.random_features.with_updates(random_seed=seed),
            )
            model = sgalp_class(
                mdp=make_inventory_mdp(inventory_config),
                config=model_config,
            )
            solution = model.fit()

            sgalp_objective = float(solution["objective_value"])
            print_progress()

            lb_config = base_lower_bound_config.with_updates(random_seed=seed + 20000)
            cvl_lower_bound = estimate_actual_lower_bound_sgalp(
                model,
                **lb_config.to_kwargs(),
            )
            best_lower_bound = (
                cvl_lower_bound
                if best_lower_bound is None
                else max(best_lower_bound, cvl_lower_bound)
            )
            print_progress()

            if compute_upper_bound:
                ub_config = base_policy_config.with_updates(
                    policy_noise_seed=seed + 510000,
                    simulation_seed=seed + 10000,
                )
                upper_bound = estimate_upper_bound_fast(model, config=ub_config)
                best_upper_bound = (
                    upper_bound
                    if best_upper_bound is None
                    else min(best_upper_bound, upper_bound)
                )
                best_gap = _compute_optimality_gap(best_lower_bound, best_upper_bound)
                print_progress()
                gap = best_gap

            elapsed_time = time.time() - start_time

            results[m][seed] = {
                "model": model,
                "solution": solution,
                "sgalp_objective": sgalp_objective,
                "lower_bound": cvl_lower_bound,
                "upper_bound": upper_bound,
                "best_lower_bound": best_lower_bound,
                "best_upper_bound": best_upper_bound,
                "gap": gap,
                "runtime_sec": elapsed_time,
            }

            print_progress(end="\n")
        print("-" * table_width)

    print("=" * table_width)
    return results


def run_sgalp_stage_trace(
    sgalp_class=None,
    max_random_features=None,
    num_constraints=None,
    num_state_relevance_samples=None,
    num_guiding_states=None,
    bandwidth_choices=None,
    basis_seed=None,
    guiding_violation=None,
    guiding_relax_fraction=None,
    guiding_abs_floor=None,
    guiding_retry_scales=None,
    sgalp_config: SGALPConfig | None = None,
    inventory_config: InventoryMDPConfig | None = None,
):
    """
    Run SGALP stage by stage and store stage-specific solver information.

    Args:
        sgalp_class: Optional SGALP-compatible class to instantiate.
        max_random_features: Largest nonconstant basis size to solve.
        num_constraints: Number of sampled Bellman inequalities per stage.
        num_state_relevance_samples: Number of states used in the ALP objective.
        num_guiding_states: Number of states used for guiding constraints.
        bandwidth_choices: Candidate bandwidths used in feature sampling.
        basis_seed: Seed controlling the sampled random-feature family.
        guiding_violation: Additive guiding-constraint allowance.
        guiding_relax_fraction: Relative guiding-constraint allowance.
        guiding_abs_floor: Minimum positive guiding allowance.
        guiding_retry_scales: Relaxation multipliers tried after LP failure.
        sgalp_config: Optional grouped SGALP settings.
    """

    sgalp_class = SelfGuidedALP if sgalp_class is None else sgalp_class
    inventory_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.inventory if inventory_config is None else inventory_config
    default_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.sgalp
    if max_random_features is None:
        max_random_features = (
            default_config.max_random_features
            if default_config.max_random_features > 0
            else max(CONTINUOUS_MDP_NOTEBOOK_CONFIG.falp_feature_counts)
        )
    num_constraints = default_config.num_constraints if num_constraints is None else num_constraints
    num_state_relevance_samples = (
        default_config.num_state_relevance_samples
        if num_state_relevance_samples is None
        else num_state_relevance_samples
    )
    num_guiding_states = (
        default_config.guiding.num_guiding_states
        if num_guiding_states is None
        else num_guiding_states
    )
    bandwidth_choices = (
        default_config.random_features.bandwidth_choices
        if bandwidth_choices is None
        else bandwidth_choices
    )
    basis_seed = default_config.random_features.random_seed if basis_seed is None else basis_seed
    guiding_violation = (
        default_config.guiding.allowed_violation
        if guiding_violation is None
        else guiding_violation
    )
    guiding_relax_fraction = (
        default_config.guiding.relax_fraction
        if guiding_relax_fraction is None
        else guiding_relax_fraction
    )
    guiding_abs_floor = (
        default_config.guiding.absolute_floor
        if guiding_abs_floor is None
        else guiding_abs_floor
    )
    guiding_retry_scales = (
        default_config.guiding.retry_scales
        if guiding_retry_scales is None
        else guiding_retry_scales
    )
    config = _make_sgalp_config(
        config=sgalp_config,
        max_random_features=max_random_features,
        batch_size=1,
        num_constraints=num_constraints,
        num_state_relevance_samples=num_state_relevance_samples,
        num_guiding_states=num_guiding_states,
        basis_seed=basis_seed,
        bandwidth_choices=bandwidth_choices,
        guiding_violation=guiding_violation,
        guiding_relax_fraction=guiding_relax_fraction,
        guiding_abs_floor=guiding_abs_floor,
        guiding_retry_scales=guiding_retry_scales,
        highs_method="highs-ds",
        primal_feasibility_tolerance=1e-7,
        dual_feasibility_tolerance=1e-7,
    )

    model = sgalp_class(
        mdp=make_inventory_mdp(inventory_config),
        config=config,
    )

    trace = []
    prev_coef = None
    prev_m = None

    for m in model.stage_feature_counts():
        solution = model.fit_stage(
            num_random_features=m,
            prev_coef=prev_coef,
            prev_num_random_features=prev_m,
        )

        model.coef = solution["coef"]
        model.current_num_random_features = m

        num_falp = solution["num_falp_constraints"]
        num_guide = solution["num_guiding_constraints"]

        guiding_states = None
        guiding_duals = None
        state_relevance_uniform = None
        state_relevance_updated = None

        if num_guide > 0:
            guiding_states = model.mdp.sample_state_relevance_states(model.num_guiding_states)
            guiding_states = np.asarray([float(s[0]) for s in guiding_states[:num_guide]], dtype=float)

            all_duals = np.asarray(model.solver_result.ineqlin.marginals, dtype=float)
            all_duals = np.abs(all_duals)
            guiding_duals = all_duals[num_falp:num_falp + num_guide]

            order = np.argsort(guiding_states)
            guiding_states = guiding_states[order]
            guiding_duals = guiding_duals[order]

            s_smooth, dual_smooth = moving_average_smoother(
                guiding_states,
                guiding_duals,
                window_size=81,
            )

            lower_bound = model.mdp.lower_state_bound
            upper_bound = model.mdp.upper_state_bound
            state_relevance_uniform = np.full_like(s_smooth, 1.0 / (upper_bound - lower_bound), dtype=float)

            integral_dual = np.trapezoid(dual_smooth, s_smooth)
            denom = 1.0 + integral_dual
            state_relevance_updated = (state_relevance_uniform + dual_smooth) / denom

            _, state_relevance_updated = moving_average_smoother(
                s_smooth,
                state_relevance_updated,
                window_size=81,
            )

            guiding_states = s_smooth
            guiding_duals = dual_smooth

        trace.append(
            {
                "m": m,
                "solution": solution,
                "coef": solution["coef"],
                "guiding_states": guiding_states,
                "guiding_duals": guiding_duals,
                "uniform_relevance": state_relevance_uniform,
                "updated_relevance": state_relevance_updated,
            }
        )

        prev_coef = solution["coef"]
        prev_m = m

    return model, trace


def run_falp_and_sgalp_comparison(
    sgalp_class=None,
    max_random_features=None,
    num_constraints=None,
    num_state_relevance_samples=None,
    num_guiding_states=None,
    bandwidth_choices=None,
    basis_seed=None,
    guiding_violation=None,
    guiding_relax_fraction=None,
    guiding_abs_floor=None,
    guiding_retry_scales=None,
    falp_config: FALPConfig | None = None,
    sgalp_config: SGALPConfig | None = None,
    inventory_config: InventoryMDPConfig | None = None,
):
    """
    Run SGALP stage by stage and fit matching FALP models for the same m values.

    Args:
        sgalp_class: Optional SGALP-compatible class to instantiate.
        max_random_features: Largest nonconstant basis size to solve.
        num_constraints: Number of sampled Bellman inequalities per stage.
        num_state_relevance_samples: Number of states used in the ALP objective.
        num_guiding_states: Number of states used for guiding constraints.
        bandwidth_choices: Candidate bandwidths used in feature sampling.
        basis_seed: Seed controlling the sampled random-feature family.
        guiding_violation: Additive guiding-constraint allowance.
        guiding_relax_fraction: Relative guiding-constraint allowance.
        guiding_abs_floor: Minimum positive guiding allowance.
        guiding_retry_scales: Relaxation multipliers tried after LP failure.
        falp_config: Optional grouped FALP settings.
        sgalp_config: Optional grouped SGALP settings.
    """

    inventory_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.inventory if inventory_config is None else inventory_config
    default_falp_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.falp
    default_sgalp_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG.sgalp

    max_random_features = (
        max(CONTINUOUS_MDP_NOTEBOOK_CONFIG.falp_feature_counts)
        if max_random_features is None
        else max_random_features
    )
    num_constraints = default_falp_config.num_constraints if num_constraints is None else num_constraints
    num_state_relevance_samples = (
        default_falp_config.num_state_relevance_samples
        if num_state_relevance_samples is None
        else num_state_relevance_samples
    )
    num_guiding_states = default_sgalp_config.guiding.num_guiding_states if num_guiding_states is None else num_guiding_states
    bandwidth_choices = default_falp_config.random_features.bandwidth_choices if bandwidth_choices is None else bandwidth_choices
    basis_seed = default_falp_config.random_features.random_seed if basis_seed is None else basis_seed
    guiding_violation = default_sgalp_config.guiding.allowed_violation if guiding_violation is None else guiding_violation
    guiding_relax_fraction = (
        default_sgalp_config.guiding.relax_fraction
        if guiding_relax_fraction is None
        else guiding_relax_fraction
    )
    guiding_abs_floor = default_sgalp_config.guiding.absolute_floor if guiding_abs_floor is None else guiding_abs_floor
    guiding_retry_scales = (
        default_sgalp_config.guiding.retry_scales
        if guiding_retry_scales is None
        else guiding_retry_scales
    )

    sg_model, sg_trace = run_sgalp_stage_trace(
        sgalp_class=sgalp_class,
        max_random_features=max_random_features,
        num_constraints=num_constraints,
        num_state_relevance_samples=num_state_relevance_samples,
        num_guiding_states=num_guiding_states,
        bandwidth_choices=bandwidth_choices,
        basis_seed=basis_seed,
        guiding_violation=guiding_violation,
        guiding_relax_fraction=guiding_relax_fraction,
        guiding_abs_floor=guiding_abs_floor,
        guiding_retry_scales=guiding_retry_scales,
        sgalp_config=sgalp_config,
        inventory_config=inventory_config,
    )

    base_falp_config = _make_falp_config(
        config=falp_config,
        num_random_features=1,
        num_constraints=num_constraints,
        num_state_relevance_samples=num_state_relevance_samples,
        basis_seed=basis_seed,
        bandwidth_choices=bandwidth_choices,
        solver="auto",
    )

    m_values = [item["m"] for item in sg_trace]
    falp_models = {}

    for m in m_values:
        falp_m = FALP(
            mdp=make_inventory_mdp(inventory_config),
            config=base_falp_config.with_updates(num_random_features=m),
        )
        falp_m.fit()
        falp_models[m] = falp_m

    return {
        "sg_model": sg_model,
        "sg_trace": sg_trace,
        "falp_models": falp_models,
        "m_values": m_values,
    }


def plot_falp_vs_sgalp_vfas_and_relevance(
    comparison_results,
    grid_size=300,
    figsize=(12, 7),
    fontsize=12,
):
    """
    Plot:
    1. FALP vs SGALP value-function approximations for each stage
    2. SGALP updated state-relevance distributions for each stage

    Args:
        comparison_results: Output dictionary from
            `run_falp_and_sgalp_comparison`.
        grid_size: Number of states used in the plotting grid.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
    """

    sg_model = comparison_results["sg_model"]
    sg_trace = comparison_results["sg_trace"]
    falp_models = comparison_results["falp_models"]

    grid = np.linspace(sg_model.mdp.lower_state_bound, sg_model.mdp.upper_state_bound, grid_size)
    n_plots = len(sg_trace)
    ncols = min(2, max(1, n_plots))
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for plot_idx, item in enumerate(sg_trace):
        m = item["m"]
        ax = axes[plot_idx]

        falp_values = evaluate_vfa_on_grid(falp_models[m], grid)
        sgalp_view = SimpleNamespace(
            basis=sg_model.basis,
            coef=item["coef"],
            current_num_random_features=m,
        )
        sgalp_values = evaluate_vfa_on_grid(sgalp_view, grid)

        ax.plot(grid, falp_values, linewidth=2.0, label="FALP")
        ax.plot(grid, sgalp_values, linewidth=2.0, linestyle="--", label="Self-guided ALP")
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"N = {m}", fontsize=fontsize)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=fontsize, frameon=False, loc="best")

    for ax in axes[:n_plots]:
        ax.set_xlabel("State s", fontsize=fontsize)
        ax.set_ylabel(r"$\hat V(s)$", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.suptitle("FALP vs Self-Guided ALP Value Function Approximations", y=0.98, fontsize=fontsize)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for plot_idx, item in enumerate(sg_trace):
        ax = axes[plot_idx]
        m = item["m"]

        if item["updated_relevance"] is None:
            ax.set_title(f"N = {m} (no guiding constraints)", fontsize=fontsize)
            ax.grid(alpha=0.3)

            lower_bound = sg_model.mdp.lower_state_bound
            upper_bound = sg_model.mdp.upper_state_bound
            s_uniform = np.linspace(lower_bound, upper_bound, 300)
            uniform_relevance = np.full_like(s_uniform, 1.0 / (upper_bound - lower_bound), dtype=float)

            ax.plot(
                s_uniform,
                uniform_relevance,
                linestyle="--",
                linewidth=1.8,
                alpha=0.6,
                label="initial uniform",
            )
            ax.legend(fontsize=fontsize, frameon=False, loc="best")
            continue

        ax.plot(
            item["guiding_states"],
            item["uniform_relevance"],
            linestyle="--",
            linewidth=2.2,
            alpha=0.6,
            label="initial uniform",
        )
        ax.plot(
            item["guiding_states"],
            item["updated_relevance"],
            linewidth=2.0,
            alpha=0.8,
            label="updated relevance",
        )
        ax.set_title(f"N = {m}", fontsize=fontsize)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=fontsize, frameon=False, loc="best")

    for ax in axes[:n_plots]:
        ax.set_xlabel("State s", fontsize=fontsize)
        ax.set_ylabel("State-relevance density", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.suptitle("Initial and Updated State-Relevance Distributions", y=0.98, fontsize=fontsize)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def summarize_falp_vs_sgalp_policy_costs(
    comparison_results,
    policy_config: PolicyEvaluationConfig | None = None,
):
    """
    Estimate policy costs for the matched FALP and SGALP stage snapshots.

    Args:
        comparison_results: Output dictionary from
            `run_falp_and_sgalp_comparison`.
        policy_config: Optional grouped policy-evaluation settings reused for
            both methods.
    """

    policy_config = PolicyEvaluationConfig() if policy_config is None else policy_config

    sg_model = comparison_results["sg_model"]
    sg_trace = comparison_results["sg_trace"]
    falp_models = comparison_results["falp_models"]

    rows = []

    for item in sg_trace:
        m = item["m"]
        sgalp_view = SimpleNamespace(
            mdp=sg_model.mdp,
            basis=sg_model.basis,
            coef=item["coef"],
            current_num_random_features=m,
        )

        falp_policy_cost = estimate_upper_bound_fast(falp_models[m], config=policy_config)
        sgalp_policy_cost = estimate_upper_bound_fast(sgalp_view, config=policy_config)

        rows.append(
            {
                "m": m,
                "falp_policy_cost": falp_policy_cost,
                "sgalp_policy_cost": sgalp_policy_cost,
                "policy_cost_difference": falp_policy_cost - sgalp_policy_cost,
            }
        )

    return {
        "rows": rows,
        "m_values": [row["m"] for row in rows],
        "falp_policy_costs": np.asarray([row["falp_policy_cost"] for row in rows], dtype=float),
        "sgalp_policy_costs": np.asarray([row["sgalp_policy_cost"] for row in rows], dtype=float),
        "policy_cost_differences": np.asarray([row["policy_cost_difference"] for row in rows], dtype=float),
    }


def plot_falp_vs_sgalp_policy_costs(
    policy_cost_summary,
    figsize=(12, 3.5),
    fontsize=12,
):
    """
    Plot matched FALP and SGALP policy costs across stage sizes.

    Args:
        policy_cost_summary: Output dictionary from
            `summarize_falp_vs_sgalp_policy_costs`.
        figsize: Figure size passed to Matplotlib.
        fontsize: Font size used for plot text.
    """

    m_values = policy_cost_summary["m_values"]
    falp_policy_costs = policy_cost_summary["falp_policy_costs"]
    sgalp_policy_costs = policy_cost_summary["sgalp_policy_costs"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        m_values,
        falp_policy_costs,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="#1565c0",
        label="FALP policy cost",
    )
    ax.plot(
        m_values,
        sgalp_policy_costs,
        marker="s",
        linewidth=2.2,
        markersize=7,
        linestyle="--",
        color="#ef6c00",
        label="SGALP policy cost",
    )
    ax.set_xticks(m_values)
    ax.set_xlabel("Number of random features (m)", fontsize=fontsize)
    ax.set_ylabel("Estimated policy cost", fontsize=fontsize)
    ax.set_title("FALP vs SGALP Policy Costs", fontsize=fontsize)
    ax.legend(frameon=False, loc="best", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _make_falp_config(
    config,
    num_random_features,
    num_constraints,
    num_state_relevance_samples,
    basis_seed,
    bandwidth_choices,
    solver,
):
    """
    Return the provided FALP config or build one from legacy scalars.

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


def _make_sgalp_config(
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
    Return the provided SGALP config or build one from legacy scalars.

    Args:
        config: Optional already-built `SGALPConfig`.
        max_random_features: Largest nonconstant basis size to solve.
        batch_size: Number of new features added per stage.
        num_constraints: Number of sampled Bellman inequalities per stage.
        num_state_relevance_samples: Number of states used in the ALP objective.
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
        guiding=SGALPConfig().guiding.with_updates(
            num_guiding_states=num_guiding_states,
            allowed_violation=guiding_violation,
            relax_fraction=guiding_relax_fraction,
            absolute_floor=guiding_abs_floor,
            retry_scales=guiding_retry_scales,
        ),
        solver=SGALPConfig().solver.with_updates(
            method=highs_method,
            primal_feasibility_tolerance=primal_feasibility_tolerance,
            dual_feasibility_tolerance=dual_feasibility_tolerance,
        ),
    )


def _make_lower_bound_config(
    config,
    num_mc_init_states,
    chain_length,
    burn_in,
    proposal_state_std,
    proposal_action_std,
    random_seed,
    noise_batch_size,
    sampler,
    num_walkers,
    initial_state,
):
    """
    Return the provided lower-bound config or build one from legacy scalars.

    Args:
        config: Optional already-built `LowerBoundConfig`.
        num_mc_init_states: Number of initial particles or walkers.
        chain_length: Number of MCMC steps per chain.
        burn_in: Number of initial MCMC steps to discard.
        proposal_state_std: Proposal standard deviation for state moves.
        proposal_action_std: Proposal standard deviation for action moves.
        random_seed: Base seed for the sampler.
        noise_batch_size: Number of demand draws reused in residual estimates.
        sampler: Sampler backend, typically `auto`, `metropolis`, or `emcee`.
        num_walkers: Number of walkers used by the optional `emcee` sampler.
        initial_state: Initial inventory level at which the bound is reported.
    """
    if config is not None:
        return config
    return LowerBoundConfig(
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


def _make_policy_config(
    config,
    state_grid_size,
    policy_noise_batch_size,
    policy_noise_seed,
    num_trajectories,
    horizon,
    simulation_seed,
    initial_state,
):
    """
    Return the provided policy config or build one from legacy scalars.

    Args:
        config: Optional already-built `PolicyEvaluationConfig`.
        state_grid_size: Number of states in the greedy-policy lookup grid.
        policy_noise_batch_size: Number of demand samples used in one-step
            lookahead.
        policy_noise_seed: Seed controlling the one-step lookahead noise batch.
        num_trajectories: Number of Monte Carlo policy-evaluation trajectories.
        horizon: Number of periods simulated in each trajectory.
        simulation_seed: Base seed for simulated demand paths.
        initial_state: Initial inventory level used in policy simulation.
    """
    if config is not None:
        return config
    return PolicyEvaluationConfig(
        state_grid_size=state_grid_size,
        policy_noise_batch_size=policy_noise_batch_size,
        policy_noise_seed=policy_noise_seed,
        num_trajectories=num_trajectories,
        horizon=horizon,
        simulation_seed=simulation_seed,
        initial_state=initial_state,
    )


def _make_psmd_config(config):
    """
    Return the provided PSMD config or fall back to tutorial defaults.

    Args:
        config: Optional already-built `PSMDConfig`.
    """
    if config is not None:
        return config
    return CONTINUOUS_MDP_NOTEBOOK_CONFIG.psmd
