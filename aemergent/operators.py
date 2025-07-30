#!/usr/bin/env python3
"""
Kernel Operators

Small, focused operators for signal processing and transformations.
Each operator does one thing well.
"""

import numpy as np
import jax.numpy as jnp
from typing import Callable, Union, List, Dict
from functools import partial

# Core constants
ZETA_NEG_HALF = -0.20788622497737067



# Type aliases
Operator = Callable[[jnp.ndarray], jnp.ndarray]
Values = Union[jnp.ndarray, np.ndarray, List[float]]


def convolve_op(kernel: Values) -> Operator:
    """Create convolution operator with given kernel."""
    return lambda values: jnp.convolve(values, jnp.array(kernel), mode="same")


def clip_op(threshold: float) -> Operator:
    """Clip values above threshold."""
    return lambda values: jnp.clip(values, -threshold, threshold)


def scale_op(factor: float) -> Operator:
    """Scale values by factor."""
    return lambda values: values * factor


def shift_op(offset: float) -> Operator:
    """Shift values by offset."""
    return lambda values: values + offset


def abs_op() -> Operator:
    """Absolute value operator."""
    return lambda values: jnp.abs(values)


def square_op() -> Operator:
    """Square values operator."""
    return lambda values: values ** 2


def normalize_op() -> Operator:
    """Normalize to unit range."""
    return lambda values: values / (jnp.max(jnp.abs(values)) + 1e-8)


def threshold_op(level: float) -> Operator:
    """Threshold values above level."""
    return lambda values: jnp.maximum(values - level, jnp.zeros_like(values))


def smooth_op(window: int = 3) -> Operator:
    """Smooth with moving average."""
    kernel = jnp.ones(window) / window
    return convolve_op(kernel)


def edge_op() -> Operator:
    """Edge detection operator."""
    kernel = jnp.array([-1, 0, 1])
    return convolve_op(kernel)


def blur_op() -> Operator:
    """Gaussian-like blur operator."""
    kernel = jnp.array([0.25, 0.5, 0.25])
    return convolve_op(kernel)


def sharpen_op() -> Operator:
    """Sharpening operator."""
    kernel = jnp.array([-0.5, 2.0, -0.5])
    return convolve_op(kernel)


def compose(*ops: Operator) -> Operator:
    """Compose multiple operators."""
    def composed(values):
        result = values
        for op in ops:
            result = op(result)
        return result
    return composed


def multiply(op1: Operator, op2: Operator) -> Operator:
    """Multiply two operators (convolution of outputs)."""
    return lambda values: jnp.convolve(op1(values), op2(values), mode="same")


def add(op1: Operator, op2: Operator) -> Operator:
    """Add two operators."""
    return lambda values: op1(values) + op2(values)


def chain(*ops: Operator) -> Operator:
    """Chain operators in sequence."""
    return compose(*ops)


# Substrate operators
def lookup_op(substrate: Dict) -> Callable:
    """Create lookup operator for substrate connections"""
    return lambda key: substrate.get(key, 0.0)

def store_op(substrate: Dict) -> Callable:
    """Create store operator for substrate connections"""
    def store(key, value):
        substrate[key] = value
        return value
    return store

def domain_op(field_names: List[str]) -> Callable:
    """Create domain operator for automata values"""
    return lambda combit: {name: combit[name] for name in field_names}

def connect_op(connection_map: Dict[str, str]) -> Callable:
    """Create connection operator mapping domains to substrate"""
    return lambda domain: {connection_map.get(k, k): v for k, v in domain.items()}

def update_op(substrate: Dict) -> Callable:
    """Create update operator for substrate"""
    def update(domain):
        substrate.update(domain)
        return domain
    return update

# Fractal chiseling operators
def chisel_op(depth: int) -> Callable:
    """Create chisel operator that carves at given depth"""
    def chisel(values):
        # Chisel creates nested patterns based on depth
        return values * (0.5 ** depth)
    return chisel

def fractal_compose(base_op: Callable, chisel_op: Callable) -> Callable:
    """Compose base operator with recursive chiseling"""
    def fractal(values):
        # Apply base transformation
        base_result = base_op(values)
        # Apply chiseling recursively
        chiseled = chisel_op(base_result)
        return chiseled
    return fractal

def pascal_chisel_op(n: int, k: int) -> Callable:
    """Create Pascal triangle-based chiseling operator"""
    from scipy.special import comb
    coeff = comb(n, k) * abs(ZETA_NEG_HALF)
    
    def chisel(values):
        # Use Pascal coefficient to determine chiseling intensity
        return values * coeff
    return chisel

def shaper_op(pattern: str) -> Callable:
    """Create shaper operator for specific patterns"""
    # Pattern operators as pure functions
    def pascal_op(values):
        pattern_array = jnp.array([1, 2, 1])
        return values * pattern_array[:len(values)]
    
    def sierpinski_op(values):
        pattern_array = jnp.array([1, 0, 1])
        return values * pattern_array[:len(values)]
    
    def fractal_op(values):
        pattern_array = jnp.array([1, 0.5, 0.25])
        return values * pattern_array[:len(values)]
    
    def identity_op(values):
        return values
    
    # Pattern mapping as operator composition
    pattern_ops = {
        "pascal": pascal_op,
        "sierpinski": sierpinski_op,
        "fractal": fractal_op
    }
    
    return pattern_ops.get(pattern, identity_op)

 