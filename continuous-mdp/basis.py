# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Shared random-feature utilities for the tutorial value-function models.

Both FALP and SGALP use the same one-dimensional random Fourier features.
Keeping that logic in one place makes the tutorial easier to maintain and
helps readers see that the two algorithms differ in constraint handling,
not in how the basis family is defined.
"""

from __future__ import annotations

import numpy as np

from config import RandomFeatureConfig


class RandomFourierBasis1D:
    """
    Random Fourier basis for a one-dimensional state.

    The first basis function is always the constant basis. The remaining
    functions are sampled once and reused. FALP uses the full sequence,
    while SGALP can use prefixes of the sequence to obtain nested models.
    """

    def __init__(self, max_random_features: int, config: RandomFeatureConfig | None = None):
        """
        Build a reusable sequence of one-dimensional Fourier features.

        Args:
            max_random_features: Largest number of nonconstant features that
                may be requested later.
            config: Random-feature settings controlling bandwidths and seeds.
        """
        self.max_random_features = int(max_random_features)
        self.config = RandomFeatureConfig() if config is None else config
        self.params = self._sample_feature_sequence()

    def _sample_feature_sequence(self):
        rng = np.random.RandomState(self.config.random_seed)

        intercepts = [0.0]
        thetas = [0.0]
        for _ in range(self.max_random_features):
            bandwidth = rng.choice(self.config.bandwidth_choices)
            intercepts.append(rng.uniform(-np.pi, np.pi))
            thetas.append(rng.normal(loc=0.0, scale=np.sqrt(2.0 * bandwidth)))

        return (
            np.asarray(intercepts, dtype=float),
            np.asarray(thetas, dtype=float),
        )

    def _resolve_num_random_features(self, num_random_features: int | None):
        """
        Translate an optional basis-size request into a valid feature count.

        Args:
            num_random_features: Requested number of nonconstant basis terms.
        """
        if num_random_features is None:
            return self.max_random_features
        if num_random_features < 0 or num_random_features > self.max_random_features:
            raise ValueError(
                f"Requested {num_random_features} random features, "
                f"but this basis only stores up to {self.max_random_features}."
            )
        return int(num_random_features)

    def eval_basis(self, state, num_random_features: int | None = None):
        """
        Evaluate the basis vector at one state.

        Args:
            state: One-dimensional state value.
            num_random_features: Number of nonconstant features to keep.
        """
        s = float(np.asarray(state, dtype=float)[0])
        intercepts, thetas = self.params
        use_count = self._resolve_num_random_features(num_random_features) + 1
        return np.cos(thetas[:use_count] * s + intercepts[:use_count])

    def eval_basis_batch(self, states, num_random_features: int | None = None):
        """
        Evaluate the basis matrix on many states at once.

        Args:
            states: Collection of one-dimensional state values.
            num_random_features: Number of nonconstant features to keep.
        """
        state_array = np.asarray(states, dtype=float).reshape(-1)
        intercepts, thetas = self.params
        use_count = self._resolve_num_random_features(num_random_features) + 1
        return np.cos(np.outer(state_array, thetas[:use_count]) + intercepts[:use_count])

    def expected_basis(self, state_list, num_random_features: int | None = None):
        """
        Average the basis vector across a batch of states.

        Args:
            state_list: Collection of one-dimensional state values.
            num_random_features: Number of nonconstant features to keep.
        """
        values = self.eval_basis_batch(state_list, num_random_features=num_random_features)
        return np.mean(values, axis=0)

    def get_vfa(self, state, coef, num_random_features: int | None = None):
        """
        Evaluate the value-function approximation at one state.

        Args:
            state: One-dimensional state value.
            coef: Coefficient vector multiplying the basis values.
            num_random_features: Number of nonconstant features to keep.
        """
        phi = self.eval_basis(state, num_random_features=num_random_features)
        return float(phi @ np.asarray(coef, dtype=float))


class PolynomialBasis1D:
    """
    Deterministic polynomial basis for a one-dimensional state.

    For the PSMD notebook we use the basis [1, s, s^2], but the class accepts
    an arbitrary ordered tuple of exponents so the interface matches the shared
    basis utilities used elsewhere in the tutorial.
    """

    def __init__(self, exponents: tuple[int, ...] = (0, 1, 2)):
        """
        Build a deterministic polynomial basis.

        Args:
            exponents: Ordered exponents used in the basis `[s^p]`.
        """
        self.exponents = tuple(int(exponent) for exponent in exponents)
        if len(self.exponents) == 0:
            raise ValueError("PolynomialBasis1D requires at least one exponent.")
        self.max_random_features = len(self.exponents) - 1

    def _resolve_num_random_features(self, num_random_features: int | None):
        """
        Translate an optional basis-size request into a valid feature count.

        Args:
            num_random_features: Requested number of nonconstant basis terms.
        """
        if num_random_features is None:
            return self.max_random_features
        if num_random_features < 0 or num_random_features > self.max_random_features:
            raise ValueError(
                f"Requested {num_random_features} nonconstant basis functions, "
                f"but this polynomial basis only stores up to {self.max_random_features}."
            )
        return int(num_random_features)

    def eval_basis(self, state, num_random_features: int | None = None):
        """
        Evaluate the basis vector at one state.

        Args:
            state: One-dimensional state value.
            num_random_features: Number of nonconstant basis terms to keep.
        """
        s = float(np.asarray(state, dtype=float)[0])
        use_count = self._resolve_num_random_features(num_random_features) + 1
        return np.asarray([s**power for power in self.exponents[:use_count]], dtype=float)

    def eval_basis_batch(self, states, num_random_features: int | None = None):
        """
        Evaluate the basis matrix on many states at once.

        Args:
            states: Collection of one-dimensional state values.
            num_random_features: Number of nonconstant basis terms to keep.
        """
        state_array = np.asarray(states, dtype=float).reshape(-1)
        use_count = self._resolve_num_random_features(num_random_features) + 1
        return np.column_stack([state_array**power for power in self.exponents[:use_count]])

    def expected_basis(self, state_list, num_random_features: int | None = None):
        """
        Average the basis vector across a batch of states.

        Args:
            state_list: Collection of one-dimensional state values.
            num_random_features: Number of nonconstant basis terms to keep.
        """
        values = self.eval_basis_batch(state_list, num_random_features=num_random_features)
        return np.mean(values, axis=0)

    def get_vfa(self, state, coef, num_random_features: int | None = None):
        """
        Evaluate the value-function approximation at one state.

        Args:
            state: One-dimensional state value.
            coef: Coefficient vector multiplying the basis values.
            num_random_features: Number of nonconstant basis terms to keep.
        """
        phi = self.eval_basis(state, num_random_features=num_random_features)
        return float(phi @ np.asarray(coef, dtype=float))
