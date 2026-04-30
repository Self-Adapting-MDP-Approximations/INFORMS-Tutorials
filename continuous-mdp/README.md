---
# How the Continuous MDP Code Works
---
This notebook is a guided tour of the continuous-state MDP portion of the tutorial codebase.

It is written for readers who want to understand how the inventory-control example is represented, approximated, fitted, and evaluated without reverse-engineering how the files fit together. The emphasis is on the *roles* of the continuous-MDP files, classes, helper functions, and notebooks.


### Table of Contents

1. [Introduction](#introduction)
2. [Codebase Map](#file-map)
3. [Config File Elements](#config)
4. [Continuous Inventory MDP Layer](#mdp)
5. [Basis Functions](#basis)
6. [Model Classes](#models)
7. [Performance Diagnostics](#evaluation)
8. [Orchestration Helpers](#helpers)
9. [Notebook Workflow](#workflow)
10. [Baseline ALP](#baseline)
11. [Self-Adaptation Improves the Baseline ALP](#self-adapt)



---
<a id="introduction"></a>
## 1. Introduction

For a discounted-cost continuous-state Markov decision process (MDP), the goal is to compute a high-quality control policy. This tutorial studies that goal through approximate linear programming, a general-purpose approach that replaces the unknown value function with a value function approximation (VFA) and optimizes its coefficients by solving a linear optimization model, called an approximate linear program (ALP).

ALP is powerful, but it creates three practical design burdens:
- **Basis-function design**: the user must choose the basis functions used in the VFA.
- **State-relevance weighting**: the user must choose a state-relevance distribution, which determines how the fitted VFA is weighted in the ALP objective.
- **Constraint handling**: the user must decide how to handle the Bellman-type constraints. In the continuous-state inventory setting, there is one constraint for every inventory state and feasible order quantity. Even when the tutorial evaluates actions on a discrete order grid, the continuous state dimension leaves a very large constraint system that cannot be directly enumerated in the way a small finite MDP can.

This tutorial uses an inventory-control problem to illustrate how self-adapting frameworks make ALP more accessible. The inventory problem has a continuous inventory state, a bounded order quantity evaluated on a configurable action grid, stochastic demand, holding, backlog, disposal, and lost-sales costs. These features make it a useful testbed for methods that reduce manual feature engineering, improve constraint handling, and limit repeated trial-and-error tuning.

We organize the material through the **COR cycle**: construct, optimize, and refine.

- **Construct**: instantiate the inventory MDP, choose basis functions, such as linear basis functions, choose a state-relevance distribution, such as the uniform distribution, and form the corresponding ALP model. Because the full ALP has too many constraints to enumerate in the continuous-state setting, we replace it with a finite sampled-constraint approximation. That is, we sample a finite set of state-action pairs and enforce the ALP constraints only at those sampled pairs. This is known as the constraint-sampling approach.

- **Optimize**: solve the resulting finite approximation to obtain an optimized VFA and compute its associated greedy policy.

- **Refine**: evaluate the current approximation using lower-bound estimates and simulated greedy-policy performance, then use this information to improve the next model. Refinement may involve expanding or modifying the basis functions, updating the state-relevance distribution, and/or handling the ALP constraints in a more systematic way.

This notebook focuses on two self-adapting methods within this cycle:

- **Constraint violation learning (CVL)** keeps the chosen basis function class fixed, but makes the optimization step adaptive. Rather than relying only on a user-chosen set of sampled constraints, CVL learns which state-action pairs are most problematic for the current approximation and focuses attention on those regions.

- **Self-guided approximate linear programs** make the basis function design adaptive. These ALPs start with a simple random-feature approximation, solve the resulting ALP, and then use information from that solution to guide the next approximation. Over repeated solves, the method adds features and constraints that help the fitted value function improve in the regions that matter most.

Together, these methods substantially reduce user burden. The user does not need to manage a continuum of Bellman-type constraints heuristically, hand-calibrate a state-relevance distribution to guess which regions of the state space matter most, or test many basis-function classes by trial and error. Instead, the procedure adapts parts of the construct, optimize, and refine stages to the inventory instance, the sampled data, and the progress of the fitted value-function approximation.


---
<a id="file-map"></a>
## 2. Codebase Map

The continuous-MDP tutorial is intentionally small. Most of the logic for the inventory model, approximation architecture, fitted algorithms, and evaluation routines lives in a handful of modules.



```python
import sys
from dataclasses import fields
from pathlib import Path
import inspect

def find_project_root(start_path: Path) -> Path:
    """
    Find the tutorial project root by looking for the shared Python modules.

    Args:
        start_path: Directory from which to begin the upward search.
    """
    for candidate in (start_path, *start_path.parents):
        if (candidate / "helper.py").exists() and (candidate / "config.py").exists():
            return candidate
        continuous_mdp = candidate / "continuous-mdp"
        if (continuous_mdp / "helper.py").exists() and (continuous_mdp / "config.py").exists():
            return continuous_mdp
    raise RuntimeError("Could not locate the tutorial project root.")

PROJECT_ROOT = find_project_root(Path.cwd().resolve())
REPOSITORY_ROOT = PROJECT_ROOT.parent
for import_root in [PROJECT_ROOT, REPOSITORY_ROOT]:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

print('Continuous MDP Python modules:')
for path in sorted(PROJECT_ROOT.glob('*.py')):
    print('  ', path.name)

print('\nself_guided_alp package:')
for path in sorted((PROJECT_ROOT / 'self_guided_alp').glob('*.py')):
    print('  ', f'self_guided_alp/{path.name}')

print('\npsmd package:')
for path in sorted((PROJECT_ROOT / 'psmd').glob('*.py')):
    print('  ', f'psmd/{path.name}')

print('\nNotebooks:')
for path in sorted((PROJECT_ROOT / 'notebooks').glob('*.ipynb')):
    print('  ', f'notebooks/{path.name}')

```

    Continuous MDP Python modules:
       basis.py
       config.py
       helper.py
       mdp.py
       policy.py

    self_guided_alp package:
       self_guided_alp/__init__.py
       self_guided_alp/cvl_lower_bound.py
       self_guided_alp/falp.py
       self_guided_alp/sgalp.py

    psmd package:
       psmd/__init__.py
       psmd/psmd.py

    Notebooks:
       notebooks/how-code-works.ipynb
       notebooks/psmd.ipynb
       notebooks/self-guided-alp.ipynb


#### The roles of the main files are:

| Path | Main role | Why it matters for the continuous MDP tutorial |
| --- | --- | --- |
| `config.py` | grouped parameter objects | keeps continuous-MDP model, solver, sampling, and evaluation settings readable |
| `mdp.py` | discounted continuous inventory model | defines state dynamics, action bounds, costs, demand sampling, and vectorized evaluation routines |
| `basis.py` | value-function basis families | provides polynomial and random Fourier basis functions for VFA |
| `self_guided_alp/falp.py` | FALP solver | solves a constraint-sampled ALP for a fixed number of random features |
| `self_guided_alp/sgalp.py` | SGALP solver | solves a sequence of constraint-sampled ALPs with increasing random features and guiding constraints at sampled states |
| `self_guided_alp/cvl_lower_bound.py` | lower-bound estimation | estimates a lower bound on the optimal policy cost for a fitted VFA using a CVL-based heuristic |
| `policy.py` | greedy-policy simulation | estimates the simulated cost of the policy induced by a fitted value approximation |
| `psmd/psmd.py` | PSMD baseline | provides a compact stochastic-gradient comparator for the same inventory MDP |
| `helper.py` | tutorial orchestration | runs continuous-MDP experiment grids, builds plots, and packages repeated evaluation steps |
| `notebooks/*.ipynb` | explanations and demos | show how the continuous-MDP building blocks are used together |


---
<a id="config"></a>
## 3. Config File Elements

`config.py` is the control panel for the continuous-MDP experiments. The most important edit points are the uppercase constants near the top of the file. The dataclasses below those constants simply package the same values so the MDP, ALP solvers, PSMD routine, lower-bound estimator, and policy simulator all receive consistent settings.

A useful way to read `config.py` is by block:

| Config block | Examples | What it controls |
| --- | --- | --- |
| Shared experiment sizes | `SEEDS`, `NUM_CONSTRAINTS`, `NUM_STATE_RELEVANCE_SAMPLES`, `FEATURE_COUNTS` | The seed grid and sampled-ALP sizes used across the polynomial ALP, FALP, and SGALP experiments. |
| Inventory instance | `LOWER_STATE_BOUND`, `UPPER_STATE_BOUND`, `MAX_ORDER`, costs, demand parameters, `ACTION_STEP` | The actual inventory MDP: state range, order grid, stochastic demand, and cost structure. |
| Policy evaluation | `POLICY_STATE_GRID_SIZE`, `POLICY_NOISE_BATCH_SIZE`, `NUM_POLICY_TRAJECTORIES`, `POLICY_HORIZON`, `INITIAL_STATE` | How greedy policies are computed and simulated. |
| Lower-bound estimation | `LOWER_BOUND_NUM_MC_INIT_STATES`, `LOWER_BOUND_CHAIN_LENGTH`, `LOWER_BOUND_SAMPLER` | How fitted VFAs are evaluated through the CVL-style lower-bound estimator. |
| Polynomial ALP example | `POLYNOMIAL_EXPONENTS`, `POLYNOMIAL_ALP_PROBE_STATES` | The hand-built baseline ALP shown in this notebook. |
| Random-feature ALPs | `RANDOM_FEATURE_BANDWIDTH_CHOICES`, `FALP_FEATURE_COUNTS`, `SGALP_FEATURE_COUNTS` | The random Fourier feature families used by FALP and SGALP. |
| SGALP guiding controls | `NUM_GUIDING_STATES`, `GUIDING_RELAX_FRACTION`, `GUIDING_RETRY_SCALES` | How SGALP carries information from one stage to the next. |
| PSMD controls | `PSMD_NUM_ITERATIONS`, `PSMD_NUM_SAMPLER_PARTICLES`, `PSMD_NUM_NOISE_SAMPLES_PER_ITERATION` | The stochastic-gradient baseline and its state-action sampler. |

Two parameters are especially important for consistency:

- `ACTION_STEP` defines the discrete order quantities used by sampled ALP constraints, PSMD sampler actions, and greedy-policy lookup.
- `NUM_CONSTRAINTS` and `NUM_STATE_RELEVANCE_SAMPLES` set the sampled-ALP scale used by the polynomial ALP example, FALP, and SGALP.

The object `CONTINUOUS_MDP_NOTEBOOK_CONFIG` collects these settings for the notebooks. In normal use, edit the uppercase constants first; the grouped configs update from those constants when the module is imported.



```python
from config import CONTINUOUS_MDP_NOTEBOOK_CONFIG


tutorial_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG

config_summary = {
    'seeds': tutorial_config.seeds,
    'sampled ALP constraints': tutorial_config.falp.num_constraints,
    'state-relevance samples': tutorial_config.falp.num_state_relevance_samples,
    'feature counts': tutorial_config.falp_feature_counts,
    'state bounds': (
        tutorial_config.inventory.lower_state_bound,
        tutorial_config.inventory.upper_state_bound,
    ),
    'max order': tutorial_config.inventory.max_order,
    'action step': tutorial_config.inventory.action_step,
    'demand samples': tutorial_config.inventory.num_noise_samples,
    'policy lookup states': tutorial_config.policy_evaluation.state_grid_size,
    'policy simulation paths': tutorial_config.policy_evaluation.num_trajectories,
    'policy horizon': tutorial_config.policy_evaluation.horizon,
    'lower-bound sampler': tutorial_config.lower_bound.sampler,
    'lower-bound chain length': tutorial_config.lower_bound.chain_length,
}

print('Current shared config values')
print('----------------------------')
name_width = max(len(name) for name in config_summary)
for name, value in config_summary.items():
    print(f'{name:<{name_width}} : {value}')

```

    Current shared config values
    ----------------------------
    seeds                    : (111, 222, 333, 444, 555)
    sampled ALP constraints  : 3000
    state-relevance samples  : 3000
    feature counts           : (0, 1, 2, 3, 4, 5, 6)
    state bounds             : (-4.0, 12.0)
    max order                : 6.0
    action step              : 0.1
    demand samples           : 1000
    policy lookup states     : 2000
    policy simulation paths  : 1000
    policy horizon           : 200
    lower-bound sampler      : metropolis
    lower-bound chain length : 2000


---
<a id="mdp"></a>
## 4. Continuous Inventory MDP Layer

`mdp.py` has two jobs:

1. define a small abstract `MarkovDecisionProcess` interface
2. implement the continuous single-product inventory model used by the tutorial algorithms

That second part is the core of the continuous-MDP example. `SingleProductInventoryMDP` defines:

- the continuous inventory state
- the bounded order-quantity action grid used by the tutorial solvers and policy lookups
- the stochastic demand process
- the one-period holding, shortage, and ordering cost
- the vectorized sampling routines reused by ALP fitting, lower-bound estimation, PSMD updates, and policy simulation



```python
from mdp import make_inventory_mdp

mdp = make_inventory_mdp()
sample_actions = mdp.get_discrete_actions()[:6]
sample_noise = mdp.sample_noise_batch(num_samples=5, random_seed=2026)

mdp_summary = {
    'class': type(mdp).__name__,
    'state bounds': (mdp.lower_state_bound, mdp.upper_state_bound),
    'max order': mdp.max_order,
    'discount factor': mdp.discount,
    'first grid actions': f'{sample_actions} ...',
    'sample demand draws': sample_noise,
}

print('Inventory MDP summary')
print('---------------------')
name_width = max(len(name) for name in mdp_summary)
for name, value in mdp_summary.items():
    print(f'{name:<{name_width}} : {value}')

```

    Inventory MDP summary


    ---------------------
    class               : SingleProductInventoryMDP
    state bounds        : (-4.0, 12.0)
    max order           : 6.0
    discount factor     : 0.95
    first grid actions  : [0.  0.1 0.2 0.3 0.4 0.5] ...
    sample demand draws : [4.48898518 1.12494111 7.09049734 5.95367792 7.04353458]


Two design choices make the rest of the continuous-MDP code simpler:

- The MDP exposes vectorized routines such as `evaluate_state_action_batch(...)`, so the ALP, PSMD, policy, and lower-bound code can all reuse one transition-and-cost interface.
- The MDP keeps a few backward-compatible aliases so the notebooks remain readable while the internal names stay descriptive.


---
<a id="basis"></a>
## 5. Basis Functions

`basis.py` defines the value-function approximation families used for continuous inventory states.

There are two basis classes:

- `RandomFourierBasis1D`: used by FALP and SGALP to approximate a value function over a continuous one-dimensional state space
- `PolynomialBasis1D`: used by the lightweight PSMD baseline and the baseline ALP example

The key conceptual point is that FALP and SGALP use the *same* random-feature family. Their difference is not the basis. Their difference is how they impose sampled Bellman constraints and how SGALP adds staged guiding constraints as the basis grows.



```python
import numpy as np

from basis import PolynomialBasis1D, RandomFourierBasis1D
from config import RandomFeatureConfig


def format_fourier_label(theta, intercept):
    sign = '+' if intercept >= 0 else '-'
    return f'cos({theta:.3f}s {sign} {abs(intercept):.3f})'


def print_matrix(title, row_label, rows, col_labels, values):
    print(title)
    print('-' * len(title))
    print('shape:', values.shape)
    col_width = max(10, *(len(label) for label in col_labels))
    header = f'{row_label:>8}  ' + '  '.join(f'{label:>{col_width}}' for label in col_labels)
    print(header)
    for row_value, row in zip(rows, values):
        row_text = '  '.join(f'{entry:{col_width}.4f}' for entry in row)
        print(f'{row_value:8.2f}  {row_text}')


rf_basis = RandomFourierBasis1D(
    max_random_features=4,
    config=RandomFeatureConfig(bandwidth_choices=(1e-2, 1e-5), random_seed=111),
)
poly_basis = PolynomialBasis1D(exponents=(0, 1, 2))

state_grid = np.array([-2.0, 0.0, 3.0])
rf_values = rf_basis.eval_basis_batch(state_grid, num_random_features=2)
poly_values = poly_basis.eval_basis_batch(state_grid)

intercepts, thetas = rf_basis.params
rf_labels = ['1'] + [
    format_fourier_label(theta, intercept)
    for theta, intercept in zip(thetas[1:3], intercepts[1:3])
]

print_matrix(
    title='Random Fourier basis values',
    row_label='state',
    rows=state_grid,
    col_labels=rf_labels,
    values=rf_values,
)
print()
print_matrix(
    title='Polynomial basis values',
    row_label='state',
    rows=state_grid,
    col_labels=['1', 's', 's^2'],
    values=poly_values,
)

```

    Random Fourier basis values
    ---------------------------
    shape: (3, 3)
       state                     1   cos(0.113s - 2.383)  cos(-0.063s - 1.286)
       -2.00                1.0000               -0.8613                0.3997
        0.00                1.0000               -0.7257                0.2810
        3.00                1.0000               -0.4560                0.0950

    Polynomial basis values
    -----------------------
    shape: (3, 3)
       state           1           s         s^2
       -2.00      1.0000     -2.0000      4.0000
        0.00      1.0000      0.0000      0.0000
        3.00      1.0000      3.0000      9.0000


For non-experts, one helpful interpretation is:

- the basis turns a continuous inventory state `s` into a feature vector
- the model learns coefficients for those features
- the approximate value function is just “basis values times coefficients”

That is why almost every fitted model in the continuous-MDP tutorial exposes the same trio of ingredients: `mdp`, `basis`, and `coef`. Once those three pieces exist, the evaluation code can build greedy policies, simulate trajectories, and estimate lower bounds in a consistent way.


---
<a id="models"></a>
## 6. Model Classes

The continuous-MDP tutorial has three main model classes.

| Class | File | Main idea |
| --- | --- | --- |
| `FALP` | `self_guided_alp/falp.py` | solve one sampled approximate LP at one random-feature basis size |
| `SelfGuidedALP` | `self_guided_alp/sgalp.py` | solve a sequence of sampled ALPs while guiding each new approximation with the previous stage |
| `PSMD` | `psmd/psmd.py` | run projected stochastic updates on a compact value-function surrogate |

All three classes operate on the same continuous inventory MDP, but they fit the value-function approximation in different ways.



```python
import inspect

from psmd.psmd import PSMD
from self_guided_alp.cvl_lower_bound import SimpleLNSLowerBound
from self_guided_alp.falp import FALP
from self_guided_alp.sgalp import SelfGuidedALP


def format_default(parameter):
    if parameter.default is inspect.Parameter.empty:
        return 'required'
    return repr(parameter.default)


def print_constructor_summary(model_class):
    parameters = [
        parameter
        for name, parameter in inspect.signature(model_class.__init__).parameters.items()
        if name != 'self'
    ]
    name_width = max(9, *(len(parameter.name) for parameter in parameters))
    default_width = max(7, *(len(format_default(parameter)) for parameter in parameters))

    print(model_class.__name__)
    print('-' * len(model_class.__name__))
    print(f'{"parameter":<{name_width}}  {"default":<{default_width}}')
    print(f'{"-" * name_width}  {"-" * default_width}')
    for parameter in parameters:
        print(f'{parameter.name:<{name_width}}  {format_default(parameter):<{default_width}}')
    print()


for model_class in [FALP, SelfGuidedALP, PSMD, SimpleLNSLowerBound]:
    print_constructor_summary(model_class)

```

    FALP
    ----
    parameter                    default
    ---------------------------  ---------------
    mdp                          required
    config                       None
    num_random_features          1
    num_constraints              40
    num_state_relevance_samples  200
    basis_seed                   111
    bandwidth_choices            (0.001, 0.0001)
    solver                       'auto'

    SelfGuidedALP
    -------------
    parameter                     default
    ----------------------------  ---------------------
    mdp                           required
    config                        None
    max_random_features           10
    batch_size                    1
    num_constraints               40
    num_state_relevance_samples   200
    num_guiding_states            100
    basis_seed                    111
    bandwidth_choices             (0.001, 0.0001)
    guiding_violation             0.0
    guiding_relax_fraction        0.02
    guiding_abs_floor             1e-06
    guiding_retry_scales          (1.0, 2.0, 5.0, 10.0)
    highs_method                  'highs-ds'
    primal_feasibility_tolerance  1e-07
    dual_feasibility_tolerance    1e-07

    PSMD
    ----
    parameter             default
    --------------------  --------
    mdp                   required
    config                None
    num_iterations        1000
    H                     10
    N                     50
    eval_interval         50
    step_size             0.2
    step_size_power       0.5
    sampler_steps         20
    proposal_state_std    0.8
    proposal_action_std   0.8
    sampling_temperature  25.0
    refresh_fraction      0.1
    coefficient_clip      500.0
    random_seed           777
    initial_state         5.0

    SimpleLNSLowerBound
    -------------------
    parameter            default
    -------------------  --------
    mdp                  required
    basis                required
    coef                 required
    num_random_features  required
    num_mc_init_states   64
    chain_length         800
    burn_in              400
    proposal_state_std   0.8
    proposal_action_std  0.8
    random_seed          333
    noise_batch_size     1000
    sampler              'auto'
    num_walkers          32
    initial_state        5.0



A simple way to separate their responsibilities is:

- `FALP` and `SelfGuidedALP` are *fitters* for sampled ALP-style approximations of the continuous-state value function.
- `SimpleLNSLowerBound` is an *evaluator* that takes a fitted approximation and estimates a lower bound from sampled Bellman residuals.
- `PSMD` is both a fitter and a tutorial baseline, but it still relies on the same continuous inventory MDP, lower-bound estimator, and policy-evaluation components.


---
<a id="evaluation"></a>
## 7. Performance Diagnostics

After fitting a value-function approximation, the notebooks report diagnostics that answer different questions. They should not be read as interchangeable numbers.

| Diagnostic | What it answers | Where it comes from |
| --- | --- | --- |
| ALP objective | How large is the fitted VFA on the sampled state-relevance distribution? | The optimization objective inside the sampled ALP. |
| CVL lower-bound estimate | What lower-bound estimate is implied by the fitted VFA and sampled Bellman residuals? | `self_guided_alp/cvl_lower_bound.py`. |
| Policy cost | What cost is obtained by the greedy policy induced by the fitted VFA? | `policy.py`, using a state grid, the `ACTION_STEP` action grid, and Monte Carlo simulation. |
| Optimality-gap estimate | How far is the simulated policy cost from the lower-bound estimate? | `(policy cost - lower bound) / policy cost`, when both quantities are available. |
| Best lower/upper bounds | What are the best estimates seen so far across feature counts or iterations? | The grid runners track max lower bound and min policy cost for each seed. |

The distinction matters. A sampled ALP objective can be useful for understanding the solve, but it is not automatically a valid lower bound because the ALP constraints are sampled. The policy cost is the deployable-policy diagnostic: it estimates how expensive the induced ordering policy is from the configured initial state. The lower-bound estimate is what makes an optimality-gap estimate possible for FALP, SGALP, and PSMD-style fitted approximations.

The project keeps this evaluation protocol shared across notebooks:

- FALP and SGALP use the same lower-bound and policy-evaluation routines.
- PSMD exposes a small fitted-model view so it can use the same policy simulator and lower-bound estimator.
- Shared settings in `config.py` keep the initial state, simulation horizon, noise batches, and sampler parameters aligned across methods.


---
<a id="helpers"></a>
## 8. Orchestration Helpers

`helper.py` is the continuous-MDP notebook support layer. It does not define the inventory problem or the approximation theory. Instead, it packages repeated experiment logic into reusable helpers with shorter, more readable calls.

This lets the notebooks stay focused on the tutorial narrative: choose settings, run continuous-MDP experiments, plot lower bounds and policy costs, and interpret the results.



```python
helper_groups = {
    'small utilities': [
        'apply_tutorial_plot_style',
        'evaluate_vfa_on_grid',
        'estimate_cvl_lower_bound',
    ],
    'experiment runners': [
        'run_falp_grid',
        'run_sgalp_grid',
        'run_psmd_seed_grid',
        'run_sgalp_stage_trace',
        'run_falp_and_sgalp_comparison',
    ],
    'plot helpers': [
        'plot_value_function_curves',
        'plot_bound_boxplots',
        'plot_psmd_iteration_diagnostics',
        'plot_psmd_acceptance_and_value',
        'plot_psmd_sampling_snapshots',
        'plot_falp_vs_sgalp_vfas_and_relevance',
        'plot_falp_vs_sgalp_policy_costs',
    ],
}

for group_index, (group_name, names) in enumerate(helper_groups.items()):
    if group_index:
        print()
    print(group_name)
    print('-' * len(group_name))
    for name in names:
        print(' ', name)

```

    small utilities
    ---------------
      apply_tutorial_plot_style
      evaluate_vfa_on_grid
      estimate_cvl_lower_bound

    experiment runners
    ------------------
      run_falp_grid
      run_sgalp_grid
      run_psmd_seed_grid
      run_sgalp_stage_trace
      run_falp_and_sgalp_comparison

    plot helpers
    ------------
      plot_value_function_curves
      plot_bound_boxplots
      plot_psmd_iteration_diagnostics
      plot_psmd_acceptance_and_value
      plot_psmd_sampling_snapshots
      plot_falp_vs_sgalp_vfas_and_relevance
      plot_falp_vs_sgalp_policy_costs


A few design choices are worth noticing here:

- the helper layer uses grouped config objects so continuous-MDP settings are visible at the notebook level
- FALP, SGALP, and PSMD plotting goes through shared plotting style defaults
- the FALP-versus-SGALP comparison logic is centralized instead of being duplicated inside the notebook
- boxplots use a shared style with visible `+` outliers and a black min-to-max line
- PSMD seed-grid helpers reuse the same inventory model and evaluation routines as the ALP examples


---
<a id="workflow"></a>
## 9. Notebook Workflow

The public continuous-MDP notebooks are not meant to reimplement the algorithms. Their job is to choose settings, call the shared helpers, and explain the outputs.

In practice, the workflow is:

1. find the continuous-MDP project root and import shared modules
2. build the inventory MDP and shared evaluation config bundles
3. choose experiment grids such as random-feature counts, SGALP stages, or PSMD seeds
4. run a helper like `run_falp_grid(...)`, `run_falp_and_sgalp_comparison(...)`, or `run_psmd_seed_grid(...)`
5. visualize fitted value functions, lower bounds, policy costs, and gaps
6. interpret the diagnostics as approximations for the continuous inventory-control problem



```python
from pathlib import Path


def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / 'helper.py').exists() and (candidate / 'config.py').exists():
            return candidate
        continuous_mdp = candidate / 'continuous-mdp'
        if (continuous_mdp / 'helper.py').exists() and (continuous_mdp / 'config.py').exists():
            return continuous_mdp
    raise RuntimeError('Could not locate the continuous-MDP project root.')


project_root = globals().get('PROJECT_ROOT', find_project_root(Path.cwd().resolve()))
notebook_paths = sorted((project_root / 'notebooks').glob('*.ipynb'))

print('Continuous-MDP notebook files')
print('-----------------------------')
for path in notebook_paths:
    print(f'notebooks/{path.name}')

```

    Continuous-MDP notebook files
    -----------------------------
    notebooks/how-code-works.ipynb
    notebooks/psmd.ipynb
    notebooks/self-guided-alp.ipynb


The three continuous-MDP notebooks have different roles:

- `self-guided-alp.ipynb`: main FALP and SGALP tutorial for continuous inventory control
- `psmd.ipynb`: PSMD baseline tutorial on the same continuous inventory MDP
- `how-code-works.ipynb`: codebase tour and orientation notebook for the continuous-MDP folder


---
<a id="baseline"></a>
## 10. Baseline ALP

This example ties the code back to the COR cycle using the simplest VFA architecture in the folder: the polynomial basis also used by the PSMD baseline. The point is not to build the strongest approximation. The point is to make one complete ALP solve visible from construction to policy evaluation.

- **Construct**: choose the inventory MDP, the polynomial VFA $V_\beta(s)=\beta_0+\beta_1s+\beta_2s^2$, a state-relevance sample, and a finite sample of Bellman constraints.
- **Optimize**: solve the finite sampled ALP with `scipy.optimize.linprog`.
- **Refine/evaluate**: convert the fitted VFA into a greedy policy and simulate that policy from the initial state. This example reports the diagnostic, but it does not feed the result back into a new construction step.

The important code parameters are:

| Concept | Math symbol | Code name | Value in the example | How it is used |
| --- | --- | --- | --- | --- |
| Polynomial basis | $\phi(s)$ | `PolynomialBasis1D(exponents=(0, 1, 2))` | `[1, s, s^2]` | Defines the VFA $V_\beta(s)=\phi(s)^\top\beta$. |
| Demand samples per Bellman expectation | $L$ | `tutorial_config.inventory.num_noise_samples` | from `config.py` | `make_inventory_mdp(...)` stores this fixed demand batch in `mdp.list_demand_obs`; `get_batch_next_state(...)` and `get_expected_cost(...)` use it inside each sampled constraint. |
| Constraint samples | $N$ | `tutorial_config.polynomial_alp.num_constraints` | from `config.py` | `sample_constraint_state_actions(num_constraints)` draws the state-action pairs $(s_i,a_i)$ used as sampled Bellman constraints. |
| State-relevance samples | $M$ | `tutorial_config.polynomial_alp.num_state_relevance_samples` | from `config.py` | `sample_state_relevance_states(...)` builds the empirical objective coefficient `c` from uniformly sampled states over the inventory interval. |
| Action grid spacing | none | `tutorial_config.inventory.action_step` | from `config.py` | Defines the discrete order quantities used by sampled ALP constraints, PSMD sampler actions, and greedy-policy lookup. |
| Reproducibility seed | none | `tutorial_config.polynomial_alp.seeds` | from `config.py` | Controls the sampled constraints, state-relevance states, and default demand batch. |
| Policy lookup grid | none | `tutorial_config.polynomial_alp.policy_grid_size` | from `config.py` | Builds a grid of inventory states where greedy actions are precomputed. |
| Demand samples for one-step lookahead | none | `tutorial_config.policy_evaluation.policy_noise_batch_size` | from `config.py` | Approximates the expectation in the greedy policy decision rule. |
| Policy simulation paths | none | `tutorial_config.policy_evaluation.num_trajectories` | from `config.py` | Number of Monte Carlo trajectories used to estimate policy cost. |
| Simulation horizon | none | `tutorial_config.policy_evaluation.horizon` | from `config.py` | Number of periods simulated in each trajectory. |
| Initial state | $s_0$ | `tutorial_config.policy_evaluation.initial_state` | from `config.py` | Starting inventory level for the reported policy-cost estimate. |

In this example, the basis vector is $\phi(s)=(1,s,s^2)^\top$, so the fitted value function is $V_\beta(s)=\phi(s)^\top\beta=\beta_0+\beta_1s+\beta_2s^2$. The ideal ALP objective averages $V_\beta(s)$ under a state-relevance distribution. In the code, this expectation is replaced by an empirical average over sampled states $\bar s_1,\ldots,\bar s_M$:
$$
    \mathbb{E}_{\nu}[V_\beta(s)] \;\approx\; \frac{1}{M}\sum_{m=1}^M V_\beta(\bar s_m) = \frac{1}{M}\sum_{m=1}^M \phi(\bar s_m)^\top\beta.
$$
This empirical average is the vector `c` in the code. The ALP constraints are also sampled. For each sampled state-action pair $(s_i,a_i)$, the Bellman inequality is approximated by a finite demand average. If $S'_{i\ell}$ is the next state obtained from demand sample $W_{i\ell}$, then the sampled constraint is
$$
    \phi(s_i)^\top\beta \le \frac{1}{L}\sum_{\ell=1}^L \left[ c(s_i,a_i,W_{i\ell}) + \gamma \phi(S'_{i\ell})^\top\beta \right], \qquad i=1,\ldots,N.
$$
The code stores the left-hand-side coefficients in `A` and the right-hand-side costs in `b`, so the constraint passed to `linprog` is
$$
    \left(\phi(s_i) - \gamma\frac{1}{L}\sum_{\ell=1}^L \phi(S'_{i\ell}) \right)^\top\beta \le \frac{1}{L}\sum_{\ell=1}^L c(s_i,a_i,W_{i\ell}), \qquad i=1,\ldots,N.
$$
Putting these pieces together, `linprog` solves `maximize c @ beta` subject to `A @ beta <= b`. Because SciPy minimizes by default, the code calls `linprog(c=-c, A_ub=A, b_ub=b, ...)`. After solving for $\beta$, the fitted VFA is wrapped in `alp_model`. The policy helpers approximate the greedy decision rule with a finite demand sample $W_1,\ldots,W_K$, where `K = policy_noise_batch_size`:
$$
    \pi_\beta(s)\in \arg\min_a
    \frac{1}{K}\sum_{k=1}^K
    \left[
    c(s,a,W_k)+\gamma V_\beta(S'_k)
    \right].
$$
Here $S'_k$ is the next inventory state generated from current state $s$, action $a$, and sampled demand $W_k$. The reported ALP objective is therefore a construction/optimization diagnostic, while the simulated policy cost is the refine-stage diagnostic for the actual greedy policy.



```python
from config import CONTINUOUS_MDP_NOTEBOOK_CONFIG
from helper import run_polynomial_sampled_alp_example


tutorial_config = CONTINUOUS_MDP_NOTEBOOK_CONFIG
alp_results = run_polynomial_sampled_alp_example(
    example_config=tutorial_config.polynomial_alp,
    inventory_config=tutorial_config.inventory,
    policy_config=tutorial_config.policy_evaluation,
)

```


    =========================================================================================================================
        seed          ALP obj      policy cost       diff %  bind constr    min slack   time (sec)
    -------------------------------------------------------------------------------------------------------------------------
         111           2304.3           4891.1         52.9            3       0.0000         7.67
    -------------------------------------------------------------------------------------------------------------------------
         222           2250.4           7691.1         70.7            3       0.0000         7.77
    -------------------------------------------------------------------------------------------------------------------------
         333           2304.7           4709.8         51.1            3       0.0000         7.82
    -------------------------------------------------------------------------------------------------------------------------
         444           2314.4           4503.9         48.6            3       0.0000         7.74
    -------------------------------------------------------------------------------------------------------------------------
         555           2277.2           5949.6         61.7            3       0.0000         7.78
    -------------------------------------------------------------------------------------------------------------------------
     AVERAGE           2290.2           5549.1         57.0                                   7.76
    =========================================================================================================================

    Shared ALP example settings
    ---------------------------
    sampled constraints       : 3000
    demand samples/constraint: 1000
    state-relevance states   : 3000
    basis                    : [1, s, s^2]
    policy lookup states     : 2000
    lookahead noise draws    : 1000
    simulation paths         : 1000
    simulation horizon       : 200
    initial state            : 6.0

    Greedy policy sample actions by seed
    ------------------------------------
        seed      -4.0       0.0       4.0       8.0      12.0
    ----------------------------------------------------------
         111       6.0       6.0       6.0       4.3       0.3
         222       6.0       6.0       5.4       1.4       0.0
         333       6.0       6.0       6.0       4.7       0.7
         444       6.0       6.0       6.0       5.1       1.1
         555       6.0       6.0       6.0       3.0       0.0


---
<a id="self-adapt"></a>
## 11. Self-Adaptation Improves the Baseline ALP

The sampled-ALP implementation above illustrates the **construct** and **optimize** steps of the COR cycle, but it does not include a genuine **refine** step. In particular, the reported `ALP obj` values are not guaranteed to be valid lower bounds on the optimal cost, because the ALP is solved only over sampled constraints rather than over the full Bellman-inequality constraint set. Consequently, the values reported in the `diff %` column, which measure the percentage difference between the simulated policy cost and the sampled-ALP objective, should not be interpreted as certified optimality gaps. For example, a large average `diff %` does not, by itself, tell us whether the difference is large because the baseline ALP policy is poor or because the sampled objective is a weak, uncertified lower-bound proxy. This ambiguity is precisely what prevents the baseline implementation from supporting a principled refinement step. The remaining notebooks show how the COR cycle can be strengthened through self-adapting methods that improve both the optimization of the ALP and the construction of the approximation architecture.

- In [psmd.ipynb](https://github.com/Self-Adapting-MDP-Approximations/INFORMS-Tutorial/blob/main/continuous-mdp/notebooks/psmd.ipynb), we present **constraint violation learning**. This method reformulates the ALP as a regularized saddle-point problem and uses a primal-dual first-order algorithm to learn the constraint-violation landscape during the solve. Instead of relying on uniform constraint sampling, as in the baseline ALP, the method endogenously concentrates computational effort on difficult state-action regions where Bellman violations are most informative. As shown in the notebook, this more principled treatment of ALP constraints substantially improves the upper bound relative to the baseline ALP by producing stronger policies with lower simulated costs. At the same time, it reports lower-bound estimates and corresponding optimality-gap estimates, which the sampled baseline ALP does not provide. In this sense, constraint violation learning refines both the construction and optimization steps of the baseline ALP.

- In [self-guided-alp.ipynb](https://github.com/Self-Adapting-MDP-Approximations/INFORMS-Tutorial/blob/main/continuous-mdp/notebooks/self-guided-alp.ipynb), we present **self-guided approximate linear programs**. This method automates the construction of basis functions through random-feature sampling and refines the approximation across iterations using guiding constraints. In contrast to the baseline ALP, which relies on a fixed, hand-specified set of basis functions, the self-guided approach expands and steers the approximation architecture using information revealed during computation. As shown in the notebook, this produces stronger lower bounds and lower policy costs than the baseline ALP; in the reported experiments, it also improves on constraint violation learning with fixed bases. Most of the improvement comes from using random features to substantially strengthen the lower-bound estimates, together with a constraint-violation-learning-based heuristic that turns the constraint-sampled self-guided ALP models into reported optimality-gap estimates. Thus, self-guided ALP tightens the optimality gap by refining all three stages of the COR cycle: construction, optimization, and refinement.

---
