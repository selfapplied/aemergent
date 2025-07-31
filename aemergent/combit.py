from .operators import convolve_op
from typing import Iterable, Callable, Union
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import comb

# Configure JAX for CPU to avoid Metal complex number issues
jax.config.update("jax_platform_name", "cpu")

# Core constants
ZETA_NEG_HALF = -0.20788622497737067

# Type aliases
Operator = Callable[[jnp.ndarray], jnp.ndarray]
Template = Union[jnp.ndarray, np.ndarray]
Indices = Union[jnp.ndarray, np.ndarray, list[int]]
Values = Union[jnp.ndarray, np.ndarray, list[float]]


class Comfit:
    basis: np.ndarray
    preds: np.ndarray
    error: float
    coeffs: np.ndarray
    energy: float

    def __init__(self, X, Y, dim):
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        self.basis = np.vander(X, N=dim, increasing=True)
        self.coeffs = np.linalg.lstsq(self.basis, Y, rcond=None)[0]
        self.preds = self.basis @ self.coeffs
        self.error = float(np.mean(np.abs(self.preds - Y)))
        self.energy = float(np.mean(np.abs(self.preds)))

    def __call__(self, X):
        X = np.atleast_1d(X)
        basis = np.vander(X, N=len(self.coeffs), increasing=True)
        return basis @ self.coeffs


def pascal_zeta(dim: int, order: int) -> jnp.ndarray:
    n, k = np.ogrid[:order, :dim]
    pascal = jnp.array(comb(n, k)) * abs(ZETA_NEG_HALF)
    return jnp.tril(pascal)


def template_op(template: Template, indices: Indices) -> Operator:
    """Create template operator for Combit template rendering"""
    return lambda values: render_template_ca(template, values, indices)


def create_template_operator(template: Template, indices: Indices) -> Operator:
    return template_op(template, indices)


def padded(state, Y):
    signal = np.array(Y)
    signal = signal[:len(state)]
    return np.pad(signal, (0, len(state) - len(signal)))


def compose_capabilities(*capabilities):
    """Compose capabilities as actual convolution through capability manifold"""
    def combined(state, vals, idxs):
        # Create a kernel from all capabilities
        capability_kernel = jnp.array(
            [cap(state, vals, idxs) for cap in capabilities])

        # Actual convolution: capabilities as kernel
        return jnp.convolve(state, capability_kernel, mode="same")
    return combined


def render_template_ca(template, values, indices):
    """Fast CA-based template rendering using vectorized operations.
    
    Args:
        template: Base array to use as template
        values: Array of values to place
        indices: Array of indices where to place values
    
    Returns:
        New array with values distributed using CA rules
    """
    result = jnp.array(template)

    # Create a mask for target positions
    mask = jnp.zeros(len(result), dtype=bool)
    mask = mask.at[indices].set(True)

    # Create value array that matches template size
    value_array = jnp.zeros(len(result))

    # Use CA rule: distribute values to masked positions
    # This is more efficient than individual assignments
    for i, idx in enumerate(indices):
        if i < len(values):
            value_array = value_array.at[idx].set(values[i])

    # Apply CA rule: where mask is True, use value_array; otherwise keep template
    result = jnp.where(mask, value_array, result)

    return result


class Combit:
    """Combit: A combinatorial base for structured signal transformation and functional encoding.

    This class implements a hybrid system that combines Pascal triangle structures with cyclotomic field 
    transformations to project input signals into a combinatorial-semantic space. It is designed as a 
    foundation for symbolic encoding, polynomial approximation, and phase-aligned signal processing.

    Core Concepts:
    - Pascal Matrix: Encodes binomial coefficients up to a given depth, weighted by ζ(-1/2), 
      creating a structured basis akin to polynomial expansions.
    - Cyclotomic Roots: Introduce a complex-valued symmetry space, allowing for phase rotations and 
      alignment of attractors in frequency-like dimensions.
    - Combit Transform: Applies a two-stage transform (Pascal basis followed by cyclotomic rotation) 
      to embed signals in a harmonically aware, structure-preserving latent space.
    - Function Approximation: Uses permuted Pascal bases to optimally align structural features 
      of functions (e.g., inflection points) with combinatorial coefficients.
    """
    def __init__(self, names: Iterable[str], state):
        self.attractors = {name: i for i, name in enumerate(names)}
        self.state = jnp.array(state)

    def __repr__(self):
        return f"[Combit order={self.order}, energy={self.energy:.4f}]"

    @property
    def order(self):
        return np.size(self.state)
    
    @property
    def dim(self):
        "Step yelling at me, this is a F_2n generated field"
        return len(self.attractors)

    @property
    def roots(self):
        return np.exp(2j * np.pi * np.arange(self.order) / self.order)

    def bitmask(self, names) -> np.ndarray:
        if isinstance(names, str):
            names = [names]
        idx = np.array([self.attractors[name] for name in names])
        bitmasks = np.zeros(self.dim, dtype=bool)
        bitmasks[idx] = True
        return bitmasks

    def transform(self, *names) -> 'Combit':
        bitmask = self.bitmask(list(names))
        pascal_tx = jnp.dot(pascal_zeta(
            self.dim, self.order), bitmask[:self.dim])
        state = jnp.convolve(
            pascal_tx, bitmask[:self.dim], mode="same") * abs(ZETA_NEG_HALF)
        return Combit(names, state[:len(names)])

    def __getitem__(self, names) -> jnp.ndarray:
        bitmasks = self.bitmask(names)
        return self.state @ bitmasks

    def __setitem__(self, names, signal):
        bitmask = self.bitmask(names)
        insides = self.state[bitmask]

        # Use convolve operator for pattern matching
        conv_op = convolve_op(signal)
        convolved = conv_op(insides)

        # Use operator-based template rendering - no more shape mismatching!
        selected_indices = jnp.where(bitmask)[0]
        template_op = create_template_operator(self.state, selected_indices)
        self.state = template_op(convolved)
        return self

    @property
    def energy(self) -> float:
        # Energy should be a scalar measure of the system state
        return float(jnp.mean(jnp.abs(self.state)))

    @property
    def capacity(self) -> jnp.ndarray:
        anext_mat = self.Anext(self.order + 1)  # Already 1D
        pascal_mat = pascal_zeta(self.dim, self.order)[0]  # Get first row
        # Both are now 1D arrays for convolution
        return jnp.convolve(anext_mat, pascal_mat, mode="same")

    def Anext(self, p: int):
        # Return first row to match expected length
        return pascal_zeta(self.dim, p)[0]

    def permute(self, energy):
        # Create a permutation based on the energy values
        # Pad energy array to match state dimension if needed
        if len(energy) < self.order:
            energy = jnp.concatenate([energy, jnp.zeros(self.order - len(energy))])
        elif len(energy) > self.order:
            energy = energy[:self.order]
        
        # Get permutation indices (sorted by energy)
        indices = jnp.argsort(energy)
        # Apply permutation by reordering state elements
        self.state = self.state[indices]

    def observe(self, X, Y, lr=1e-3):
        signal = padded(self.state, Y)
        self.state = self.state * (1 - lr) + signal * lr
        self.permute(self.state)

    def __call__(self, X):
        X = np.atleast_1d(X)
        basis = np.vander(X, N=self.order, increasing=True)
        return basis @ self.state


__all__ = ["Combit", "Comfit"]