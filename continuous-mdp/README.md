---
# How the Continuous MDP Code Works
---
This notebook is a guided tour of the continuous state-action MDP portion of the tutorial codebase.

It is written for readers who want to understand how the inventory-control example is represented, approximated, fitted, and evaluated without reverse-engineering how the files fit together. The emphasis is on the *roles* of the continuous-MDP files, classes, helper functions, and notebooks.


### Table of Contents

1. [Introduction](#introduction)
2. [Codebase Map](#file-map)
3. [Configuration Objects](#config)
4. [Continuous Inventory MDP Layer](#mdp)
5. [Basis Functions](#basis)
6. [Model Classes](#models)
7. [Evaluation Helpers](#evaluation)
8. [Orchestration Helpers](#helpers)
9. [Notebook Workflow](#workflow)
10. [Baseline ALP](#baseline)
11. [How Self-Adaptation Improves the Baseline ALP](#self-adapt)



---
<a id="introduction"></a>
## 1. Introduction

For a discounted-cost continuous state-action Markov decision process (MDP), the goal is to compute a high-quality control policy. This tutorial studies that goal through approximate linear programming, a general-purpose approach that replaces the unknown value function with a value function approximation (VFA) and optimizes its coefficients by solving a linear optimization model, called an approximate linear program (ALP).

ALP is powerful, but it creates three practical design burdens:
- **Basis-function design**: the user must choose the basis functions used in the VFA.
- **State-relevance weighting**: the user must choose a state-relevance distribution, which determines how the fitted VFA is weighted in the ALP objective.
- **Constraint handling**: the user must decide how to handle the Bellman-type constraints. In continuous state-action spaces, there is one constraint for every feasible state-action pair. Thus, the ALP is a linear optimization problem with infinitely many constraints, rather than a finite model that can be directly passed to existing solvers.

This tutorial uses an inventory-control problem to illustrate how self-adapting frameworks make ALP more accessible. The inventory problem has a continuous inventory state, a continuous order quantity, stochastic demand, holding, backlog, disposal, and lost-sales costs. These features make it a useful testbed for methods that reduce manual feature engineering, improve constraint handling, and limit repeated trial-and-error tuning.

We organize the material through the **COR cycle**: construct, optimize, and refine.

- **Construct**: instantiate the inventory MDP, choose basis functions, such as linear basis functions, choose a state-relevance distribution, such as the uniform distribution, and form the corresponding ALP model. Because the full ALP has infinitely many constraints in continuous state-action spaces, we replace it with a finite sampled-constraint approximation. That is, we sample a finite set of state-action pairs and enforce the ALP constraints only at those sampled pairs. This is known as the constraint-sampling approach.

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

print('\nShared repository Python modules:')
for path in sorted(REPOSITORY_ROOT.glob('*.py')):
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
       config.py
       helper.py
       mdp.py
       policy.py
    
    Shared repository Python modules:
       basis.py
    
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
| `../basis.py` | shared value-function basis families | provides polynomial and random Fourier basis functions for VFA |
| `self_guided_alp/falp.py` | FALP solver | solves a constraint-sampled ALP for a fixed number of random features |
| `self_guided_alp/sgalp.py` | SGALP solver | solves a sequence of constraint-sampled ALPs with increasing random features and guiding constraints that inform the state-relevance distribution |
| `self_guided_alp/cvl_lower_bound.py` | lower-bound estimation | estimates a lower bound on the optimal policy cost for a fitted VFA using a CVL-based heuristic |
| `policy.py` | greedy-policy simulation | estimates the simulated cost of the policy induced by a fitted value approximation |
| `psmd/psmd.py` | PSMD baseline | provides a compact stochastic-gradient comparator for the same inventory MDP |
| `helper.py` | tutorial orchestration | runs continuous-MDP experiment grids, builds plots, and packages repeated evaluation steps |
| `notebooks/*.ipynb` | explanations and demos | show how the continuous-MDP building blocks are used together |


---
<a id="config"></a>
## 3. Configuration Objects

The continuous-MDP code uses grouped config objects to keep modeling, approximation, solver, and evaluation choices separate.

Instead of passing many unrelated scalars through constructors, the project uses dataclasses to bundle related choices together. This is especially helpful in the continuous inventory example because the same experiment combines demand sampling, random-feature sampling, LP solver tolerances, lower-bound sampling, and policy simulation.



```python
from dataclasses import fields

from config import (
    FALPConfig,
    GuidingConstraintConfig,
    HiGHSSolverConfig,
    InventoryMDPConfig,
    LowerBoundConfig,
    PolicyEvaluationConfig,
    PSMDConfig,
    RandomFeatureConfig,
    SGALPConfig,
)

config_groups = {
    'model': [InventoryMDPConfig],
    'basis and solver': [RandomFeatureConfig, HiGHSSolverConfig, GuidingConstraintConfig],
    'algorithms': [FALPConfig, SGALPConfig, PSMDConfig],
    'evaluation': [LowerBoundConfig, PolicyEvaluationConfig],
}

for group_index, (group_name, config_classes) in enumerate(config_groups.items()):
    if group_index:
        print()
    print(group_name.upper())
    print('=' * len(group_name))
    for config_class in config_classes:
        print(f'\n{config_class.__name__}')
        print('-' * len(config_class.__name__))
        for config_field in fields(config_class):
            print(f'  {config_field.name}')

```

    MODEL
    =====
    
    InventoryMDPConfig
    ------------------
      mdp_name
      discount
      random_seed
      lower_state_bound
      upper_state_bound
      max_order
      purchase_cost
      holding_cost
      backlog_cost
      disposal_cost
      lost_sale_cost
      demand_mean
      demand_std
      demand_min
      demand_max
      num_noise_samples
      action_step
    
    BASIS AND SOLVER
    ================
    
    RandomFeatureConfig
    -------------------
      bandwidth_choices
      random_seed
    
    HiGHSSolverConfig
    -----------------
      method
      primal_feasibility_tolerance
      dual_feasibility_tolerance
    
    GuidingConstraintConfig
    -----------------------
      num_guiding_states
      allowed_violation
      relax_fraction
      absolute_floor
      retry_scales
    
    ALGORITHMS
    ==========
    
    FALPConfig
    ----------
      num_random_features
      num_constraints
      num_state_relevance_samples
      random_features
      solver
    
    SGALPConfig
    -----------
      max_random_features
      batch_size
      num_constraints
      num_state_relevance_samples
      random_features
      guiding
      solver
    
    PSMDConfig
    ----------
      num_iterations
      H
      N
      eval_interval
      step_size
      step_size_power
      sampler_steps
      proposal_state_std
      proposal_action_std
      sampling_temperature
      refresh_fraction
      coefficient_clip
      random_seed
      initial_state
      snapshot_iterations
      snapshot_sample_size
      snapshot_sampler_steps
      snapshot_refresh_fraction
      lower_bound
      policy_evaluation
    
    EVALUATION
    ==========
    
    LowerBoundConfig
    ----------------
      num_mc_init_states
      chain_length
      burn_in
      proposal_state_std
      proposal_action_std
      random_seed
      noise_batch_size
      sampler
      num_walkers
      initial_state
    
    PolicyEvaluationConfig
    ----------------------
      state_grid_size
      policy_noise_batch_size
      policy_noise_seed
      num_trajectories
      horizon
      simulation_seed
      initial_state


A useful way to read these config classes is by purpose:

- `InventoryMDPConfig` stores the continuous inventory-model data: bounds, costs, discount factor, and demand distribution.
- `RandomFeatureConfig` controls how the random Fourier basis for continuous states is sampled.
- `HiGHSSolverConfig` stores numerical settings for the sampled linear-program solver.
- `GuidingConstraintConfig` stores SGALP-only settings for staged continuous-state guiding constraints.
- `FALPConfig`, `SGALPConfig`, and `PSMDConfig` store algorithm-level settings.
- `LowerBoundConfig` and `PolicyEvaluationConfig` store evaluation settings shared across the continuous-MDP notebooks.

The helper function `make_shared_evaluation_configs(...)` is especially important because it keeps the lower-bound and policy-cost settings aligned across FALP, SGALP, and PSMD. That makes the reported comparisons reflect the algorithms rather than different evaluation choices.



```python
from dataclasses import fields

from helper import make_shared_evaluation_configs


def print_config(title, config):
    print(title)
    print('-' * len(title))
    name_width = max(len(config_field.name) for config_field in fields(config))
    for config_field in fields(config):
        value = getattr(config, config_field.name)
        print(f'{config_field.name:<{name_width}} : {value}')


shared_lower_bound_config, shared_policy_config = make_shared_evaluation_configs(initial_state=0.0)

print_config('Lower-bound evaluation config', shared_lower_bound_config)
print()
print_config('Policy-evaluation config', shared_policy_config)

```

    Lower-bound evaluation config
    -----------------------------
    num_mc_init_states  : 32
    chain_length        : 2000
    burn_in             : 500
    proposal_state_std  : 0.8
    proposal_action_std : 0.8
    random_seed         : 333
    noise_batch_size    : 1000
    sampler             : metropolis
    num_walkers         : 32
    initial_state       : 0.0
    
    Policy-evaluation config
    ------------------------
    state_grid_size         : 801
    policy_noise_batch_size : 1024
    policy_noise_seed       : 123456
    num_trajectories        : 2000
    horizon                 : 200
    simulation_seed         : 2026
    initial_state           : 0.0


---
<a id="mdp"></a>
## 4. Continuous Inventory MDP Layer

`mdp.py` has two jobs:

1. define a small abstract `MarkovDecisionProcess` interface
2. implement the continuous single-product inventory model used by the tutorial algorithms

That second part is the core of the continuous-MDP example. `SingleProductInventoryMDP` defines:

- the continuous inventory state
- the continuous order-quantity action, with feasible action bounds depending on the current state
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
    state bounds        : (-10.0, 10.0)
    max order           : 10.0
    discount factor     : 0.95
    first grid actions  : [0. 1. 2. 3. 4. 5.] ...
    sample demand draws : [4.13656296 2.21425206 5.62314134 4.97353024 7.89941546]


Two design choices make the rest of the continuous-MDP code simpler:

- The MDP exposes vectorized routines such as `evaluate_state_action_batch(...)`, so the ALP, PSMD, policy, and lower-bound code can all reuse one transition-and-cost interface.
- The MDP keeps a few backward-compatible aliases so the notebooks remain readable while the internal names stay descriptive.


---
<a id="basis"></a>
## 5. Basis Functions

`../basis.py` defines the value-function approximation families used for continuous inventory states.

There are two basis classes:

- `RandomFourierBasis1D`: used by FALP and SGALP to approximate a value function over a continuous one-dimensional state space
- `PolynomialBasis1D`: used by the lightweight PSMD baseline

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
## 7. Evaluation Helpers

After fitting a continuous-MDP approximation, the project reports two complementary diagnostics:

- a **lower bound** from `self_guided_alp/cvl_lower_bound.py`
- a **policy cost** from `policy.py`

The lower bound is a performance certificate based on sampled Bellman residuals. The policy cost is a simulation-based estimate of what the induced greedy ordering policy actually costs from the chosen initial inventory state.



```python
import inspect

from policy import build_greedy_policy_lookup, estimate_upper_bound_fast
from self_guided_alp.cvl_lower_bound import (
    estimate_actual_lower_bound_falp,
    estimate_actual_lower_bound_sgalp,
)


def format_default(parameter):
    if parameter.default is inspect.Parameter.empty:
        return 'required'
    return repr(parameter.default)


def print_helper_summary(helper):
    parameters = list(inspect.signature(helper).parameters.values())
    name_width = max(9, *(len(parameter.name) for parameter in parameters))
    default_width = max(7, *(len(format_default(parameter)) for parameter in parameters))

    print(helper.__name__)
    print('-' * len(helper.__name__))
    print(f'{"parameter":<{name_width}}  {"default":<{default_width}}')
    print(f'{"-" * name_width}  {"-" * default_width}')
    for parameter in parameters:
        print(f'{parameter.name:<{name_width}}  {format_default(parameter):<{default_width}}')
    print()


for helper in [
    build_greedy_policy_lookup,
    estimate_upper_bound_fast,
    estimate_actual_lower_bound_falp,
    estimate_actual_lower_bound_sgalp,
]:
    print_helper_summary(helper)

```

    build_greedy_policy_lookup
    --------------------------
    parameter  default 
    ---------  --------
    model      required
    config     None    
    
    estimate_upper_bound_fast
    -------------------------
    parameter  default 
    ---------  --------
    model      required
    config     None    
    return_se  False   
    
    estimate_actual_lower_bound_falp
    --------------------------------
    parameter            default 
    -------------------  --------
    falp_model           required
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
    return_stats         False   
    
    estimate_actual_lower_bound_sgalp
    ---------------------------------
    parameter            default 
    -------------------  --------
    sgalp_model          required
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
    return_stats         False   
    


The evaluation story is intentionally shared across the continuous-MDP notebooks:

- FALP and SGALP both call the same lower-bound and policy-evaluation logic.
- PSMD also uses the same `estimate_upper_bound_fast(...)` helper through a small fitted-model view.
- The notebooks usually create `shared_lower_bound_config` and `shared_policy_config` once and reuse them throughout so comparisons stay apples-to-apples.

This shared evaluation layer matters because the algorithms are approximate and simulation-based. Keeping the evaluation protocol fixed makes it easier to interpret differences between value-function fits.


---
<a id="helpers"></a>
## 8. Orchestration Helpers

`helper.py` is the continuous-MDP notebook support layer. It does not define the inventory problem or the approximation theory. Instead, it packages repeated experiment logic into reusable helpers with shorter, more readable calls.

This lets the notebooks stay focused on the tutorial narrative: choose settings, run continuous-MDP experiments, plot lower bounds and policy costs, and interpret the results.



```python
helper_groups = {
    'small utilities': [
        'apply_tutorial_plot_style',
        'make_shared_evaluation_configs',
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
      make_shared_evaluation_configs
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
4. run a helper like `run_falp_grid(...)`, `run_falp_sgalp_comparison(...)`, or `run_psmd_seed_grid(...)`
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
- **Refine**: convert the fitted VFA into a greedy policy and simulate that policy from the initial state.

The important code parameters are:

| Concept | Math symbol | Code name | Value in the example | How it is used |
| --- | --- | --- | --- | --- |
| Polynomial basis | $\phi(s)$ | `PolynomialBasis1D(exponents=(0, 1, 2))` | `[1, s, s^2]` | Defines the VFA $V_\beta(s)=\phi(s)^\top\beta$. |
| Demand samples per Bellman expectation | $L$ | `InventoryMDPConfig(num_noise_samples=3000)` | `3000` | `make_inventory_mdp(...)` stores this fixed demand batch in `mdp.list_demand_obs`; `get_batch_next_state(...)` and `get_expected_cost(...)` use it inside each sampled constraint. |
| Constraint samples | $N$ | `num_constraints` | `3000` | `sample_constraint_state_actions(num_constraints)` draws the state-action pairs $(s_i,a_i)$ used as sampled Bellman constraints. |
| State-relevance samples | $M$ | `num_state_relevance_samples` | `3000` random states plus 3 boundary/reference states | `sample_state_relevance_states(...)` builds the empirical objective coefficient `c`. The extra states are the lower bound, zero, and upper bound. |
| Action grid spacing | none | `InventoryMDPConfig(action_step=1.0)` | `1.0` | Defines the discrete order quantities used when constructing the greedy policy. |
| Reproducibility seed | none | `InventoryMDPConfig(random_seed=12345)` | `12345` | Controls the sampled constraints, state-relevance states, and default demand batch. |
| Policy lookup grid | none | `PolicyEvaluationConfig(state_grid_size=121)` | `121` | Builds a grid of inventory states where greedy actions are precomputed. |
| Demand samples for one-step lookahead | none | `policy_noise_batch_size` | `1000` | Approximates the expectation in the greedy policy decision rule. |
| Policy simulation paths | none | `num_trajectories` | `2000` | Number of Monte Carlo trajectories used to estimate policy cost. |
| Simulation horizon | none | `horizon` | `200` | Number of periods simulated in each trajectory. |
| Initial state | $s_0$ | `initial_state` | `5.0` | Starting inventory level for the reported policy-cost estimate. |

In this example, the basis vector is $\phi(s)=(1,s,s^2)^\top$, so the fitted value function is $V_\beta(s)=\phi(s)^\top\beta=\beta_0+\beta_1s+\beta_2s^2$. The ideal ALP objective averages $V_\beta(s)$ under a state-relevance distribution. In the code, this expectation is replaced by an empirical average over sampled states $\bar s_1,\ldots,\bar s_M$:
$$
    \mathbb{E}_{\chi}[V_\beta(s)] \;\approx\; \frac{1}{M}\sum_{m=1}^M V_\beta(\bar s_m) = \frac{1}{M}\sum_{m=1}^M \phi(\bar s_m)^\top\beta.
$$
This empirical average is the vector `c` in the code. The ALP constraints are also sampled. For each sampled state-action pair $(s_i,a_i)$, the continuous ALP constraint is approximated by a finite demand average. If $S'_{i\ell}$ is the next state obtained from demand sample $W_{i\ell}$, then the sampled constraint is
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
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

from basis import PolynomialBasis1D
from config import InventoryMDPConfig, PolicyEvaluationConfig
from mdp import make_inventory_mdp
from policy import build_greedy_policy_lookup, estimate_upper_bound_fast

np.set_printoptions(precision=3, suppress=True)

# Example parameters collected in one place.
DEMAND_SAMPLES_PER_CONSTRAINT = 2000
ACTION_STEP = 1.0
MDP_RANDOM_SEED = 111
POLYNOMIAL_EXPONENTS = (0, 1, 2)
NUM_CONSTRAINTS = 1000
NUM_STATE_RELEVANCE_SAMPLES = 1000
POLICY_GRID_SIZE = 121
POLICY_NOISE_BATCH_SIZE = 2000
POLICY_NOISE_SEED = 222
NUM_POLICY_TRAJECTORIES = 2000
POLICY_HORIZON = 200
POLICY_SIMULATION_SEED = 333
INITIAL_STATE = 5.0
PROBE_STATES = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])

"""
------------------------------------------------------------------------
                                Construct
------------------------------------------------------------------------
Choose the MDP, the polynomial VFA, and the two sample sets.
The fixed demand batch approximates expectations inside each constraint.
"""
mdp = make_inventory_mdp(
    InventoryMDPConfig(
        num_noise_samples=DEMAND_SAMPLES_PER_CONSTRAINT,
        action_step=ACTION_STEP,
        random_seed=MDP_RANDOM_SEED,
    )
)
basis = PolynomialBasis1D(exponents=POLYNOMIAL_EXPONENTS)

state_samples, action_samples = mdp.sample_constraint_state_actions(NUM_CONSTRAINTS)
relevance_states = mdp.sample_state_relevance_states(NUM_STATE_RELEVANCE_SAMPLES)

# Construct A beta <= b, where each row is one sampled Bellman inequality.
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

# Construct objective coefficient c as the average basis vector over relevance states.
c = np.mean([basis.eval_basis(state) for state in relevance_states], axis=0)

"""
------------------------------------------------------------------------
                                Optimize
------------------------------------------------------------------------
linprog minimizes, so maximizing c^T beta becomes minimizing -c^T beta.
"""
result = linprog(c=-c, A_ub=A, b_ub=b, bounds=[(None, None)] * len(c), method='highs')
if not result.success:
    raise RuntimeError(result.message)

coef = np.asarray(result.x, dtype=float)
alp_objective = float(c @ coef)
sampled_slacks = b - A @ coef
min_sampled_slack = 0.0 if abs(sampled_slacks.min()) < 1e-8 else float(sampled_slacks.min())

## Wrap the fitted VFA and evaluate its induced greedy policy.
alp_model = SimpleNamespace(mdp=mdp, basis=basis, coef=coef, num_random_features=len(POLYNOMIAL_EXPONENTS) - 1)
policy_config = PolicyEvaluationConfig(
    state_grid_size=POLICY_GRID_SIZE,
    policy_noise_batch_size=POLICY_NOISE_BATCH_SIZE,
    policy_noise_seed=POLICY_NOISE_SEED,
    num_trajectories=NUM_POLICY_TRAJECTORIES,
    horizon=POLICY_HORIZON,
    simulation_seed=POLICY_SIMULATION_SEED,
    initial_state=INITIAL_STATE,
)
policy_cost, policy_se = estimate_upper_bound_fast(alp_model, config=policy_config, return_se=True)
state_grid, policy_actions = build_greedy_policy_lookup(alp_model, config=policy_config)
probe_actions = [policy_actions[np.abs(state_grid - state).argmin()] for state in PROBE_STATES]

def print_metric(label, value):
    print(f'{label:<27}: {value}')


print('-' * 100)
print('Polynomial sampled ALP result')
print('-' * 100)
print_metric('status', result.message)
print_metric('sampled constraints', NUM_CONSTRAINTS)
print_metric('demand samples/constraint', mdp.num_noise_samples)
print_metric('state-relevance states', len(relevance_states))
print_metric('basis', '[1, s, s^2]')
print_metric('coefficients', coef)
print_metric('ALP objective', f'{alp_objective:,.2f}')
print_metric('min sampled slack', f'{min_sampled_slack:.4f}')
print_metric('binding constraints', int((sampled_slacks <= 1e-6).sum()))
print()
print('Greedy policy from fitted VFA')
print('-' * 100)
print_metric('initial state', policy_config.initial_state)
print_metric('policy lookup states', policy_config.state_grid_size)
print_metric('lookahead noise draws', policy_config.policy_noise_batch_size)
print_metric('simulation paths', policy_config.num_trajectories)
print_metric('simulation horizon', policy_config.horizon)
print_metric('simulated policy cost', f'{policy_cost:,.2f} ± {1.96 * policy_se:,.2f} (95% MC error)')
print('sample actions')
for state, action in zip(PROBE_STATES, probe_actions):
    print(f'  state {state:5.1f} -> order {action:4.1f}')
print('-' * 100)

```

    ----------------------------------------------------------------------------------------------------
    Polynomial sampled ALP result
    ----------------------------------------------------------------------------------------------------
    status                     : Optimization terminated successfully. (HiGHS Status 7: Optimal)
    sampled constraints        : 1000
    demand samples/constraint  : 2000
    state-relevance states     : 1003
    basis                      : [1, s, s^2]
    coefficients               : [2160.203  -20.001    0.   ]
    ALP objective              : 2,162.27
    min sampled slack          : 0.0000
    binding constraints        : 3
    
    Greedy policy from fitted VFA
    ----------------------------------------------------------------------------------------------------
    initial state              : 5.0
    policy lookup states       : 121
    lookahead noise draws      : 2000
    simulation paths           : 2000
    simulation horizon         : 200
    simulated policy cost      : 2,047.30 ± 5.51 (95% MC error)
    sample actions
      state  -8.0 -> order 10.0
      state  -4.0 -> order 10.0
      state   0.0 -> order  6.0
      state   4.0 -> order  2.0
      state   8.0 -> order  0.0
    ----------------------------------------------------------------------------------------------------


---
<a id="self-adapt"></a>
## 11. How Self-Adaptation Improves the Baseline ALP

The simple sampled-ALP implementation above illustrates the construct and optimize steps, but it does not include a genuine refinement step. The remaining notebooks show how the COR cycle can be strengthened with self-adapting methods.

- In `continuous-mdp/notebooks/psmd.ipynb`, we present constraint violation learning. This method reformulates the ALP as a regularized saddle-point problem and uses a primal-dual first-order algorithm to learn the violation landscape during the solve. In this way, it replaces the user's choice of a constraint-sampling distribution with an endogenous distribution that focuses on difficult state-action regions.

- In `continuous-mdp/notebooks/self-guided-alp.ipynb`, we present self-guided approximate linear programs. This method automates the design of basis functions through random-feature sampling and refines the approximation across iterations using guiding constraints. In this way, it addresses the construct and refine stages that the baseline sampled-ALP implementation leaves open.

---
