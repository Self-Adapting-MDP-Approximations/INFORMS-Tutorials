# INFORMS Tutorials in Operations Research

This repository contains the code and tutorial material for **Self-Adapting MDP Approximations**, developed as part of an INFORMS Tutorials in Operations Research project. The repository is intended for researchers, PhD students, and practitioners interested in reinforcement learning, approximate dynamic programming, adaptive approximation methods, and Markov decision processes.

The goal is to connect theory and computation through readable implementations, tutorial notebooks, and supporting references. The material is designed to help readers move from conceptual understanding to reproducible experimentation, even when they are not specialists in the underlying algorithms.

## What This Repository Contains

This repository includes research code and tutorial material for self-adapting approximation methods, organized around two broad classes of Markov decision processes.

### [Weakly Coupled Markov Decision Processes](https://github.com/Self-Adapting-MDP-Approximations/INFORMS-Tutorial/tree/main/weakly-coupled-mdp)

This part focuses on approximation frameworks for large-scale weakly coupled MDPs, where many local decision processes interact through shared resource constraints or aggregate coupling constraints.

- **Self-adapting network relaxations for weakly coupled Markov decision processes**  
  [https://doi.org/10.1287/mnsc.2022.01108](https://doi.org/10.1287/mnsc.2022.01108)

- **Delayed Allocation in Marginalized Flow Models for Weakly Coupled Markov Decision Processes**  
  [http://dx.doi.org/10.2139/ssrn.6127326](http://dx.doi.org/10.2139/ssrn.6127326)

- **Applications:** dynamic assortment optimization.

### [General Continuous State-Action Markov Decision Processes](https://github.com/Self-Adapting-MDP-Approximations/INFORMS-Tutorial/tree/main/continuous-mdp)

This part focuses on approximate linear programming and adaptive value-function approximation methods for MDPs with continuous state and action spaces.

- **Constraint-violation learning, an approximate linear programming method**  
  [https://doi.org/10.1287/mnsc.2019.3289](https://doi.org/10.1287/mnsc.2019.3289)

- **Self-Guided Approximate Linear Programs: Randomized Multi-Shot Approximation of Discounted Cost Markov Decision Processes**  
  [https://doi.org/10.1287/mnsc.2020.00038](https://doi.org/10.1287/mnsc.2020.00038)

- **Applications:** inventory control.

## How to Use This Repository

Readers can use this repository to:

- Study the tutorial notebooks.
- Inspect implementation details behind the methods.
- Reproduce computational examples.
- Modify examples for new applications.
- Build extensions for related MDP models.

The repository is intended as a public-facing scholarly resource rather than a classroom platform. It provides a place to study the methods, examine the code, ask questions, and engage with the research community through discussion and future extensions.

## Repository Layout

- `continuous-mdp/`: code and notebooks for continuous state-action MDP examples.
- `continuous-mdp/self_guided_alp/`: Python package for FALP, SGALP, and lower-bound routines.
- `continuous-mdp/psmd/`: Python package for the PSMD baseline.
- `continuous-mdp/notebooks/`: tutorial notebooks for the continuous MDP material.
- `weakly-coupled-mdp/`: code and notebooks for weakly coupled MDP examples.
- `weakly-coupled-mdp/notebooks/`: tutorial notebooks for the weakly coupled MDP material.

## Authors and Maintainers

This repository is maintained by:

- [Andre Cire](https://utsc.utoronto.ca/mgmt/andre-cire)
- [Selva Nadarajah](https://selvan.people.uic.edu/)
- [Parshan Pakiman](https://parshanpakiman.github.io/)
- [Negar Soheili](https://www.negar-soheili.com/)

## License

This repository is released under the MIT License.
