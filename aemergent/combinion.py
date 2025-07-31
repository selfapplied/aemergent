#!/usr/bin/env python3
"""
Combinion - Quaternion-Dimensional Array Type System

A revolutionary array type system that separates:
- Pascal Order: Fixed combinatorial depth (mathematical invariant)
- Quaternion Dimensions: Geometric spatial encoding via QNode rotations
- F_2^n Fields: Finite field algebra for type safety

Architecture:
- CombitArray: Arrays with quaternion-encoded dimensions
- QDim: Quaternion dimensions that rotate in 4D space
- Pascal order stays pure and combinatorial
- Type checking becomes geometric collision detection

Example:
    >>> from aemergent.combinion import CombitArray, QDim
    >>> # Create array with quaternion dimensions
    >>> arr = CombitArray(data, pascal_order=3, qdim=QDim(1, 0, 0, 0))
    >>> # Rotate dimensions in quaternion space
    >>> rotated = arr.rotate_dims('x', np.pi/4)
"""

import numpy as np
import jax.numpy as jnp
from typing import Union, Tuple, Optional, Any
from dataclasses import dataclass
from .gravtree import QNode, quat_mult_vec


@dataclass
class QDim:
    """
    Quaternion Dimension Encoder
    
    Encodes array dimensions as quaternion rotations in 4D space.
    This separates geometric representation from combinatorial structure.
    """
    qnode: QNode
    field_base: int = 2  # F_2^n field characteristic
    
    def __init__(self, w: float, x: float, y: float, z: float, field_base: int = 2):
        self.qnode = QNode(w, x, y, z)
        self.field_base = field_base
    
    @property
    def quat(self) -> np.ndarray:
        """Get raw quaternion values"""
        return self.qnode.quat
    
    @property
    def geometric_dims(self) -> Tuple[int, ...]:
        """Convert quaternion to actual array dimensions"""
        # Map quaternion components to dimension sizes
        # w: primary dimension, x,y,z: secondary dimensions
        w, x, y, z = np.abs(self.qnode.quat)
        
        # Scale to reasonable dimension sizes (1-16 typical range)
        primary = max(1, int(w * 16))
        secondary = max(1, int(x * 8))
        tertiary = max(1, int(y * 4))
        quaternary = max(1, int(z * 2))
        
        return (primary, secondary, tertiary, quaternary)
    
    def rotate(self, axis: str, angle: float) -> 'QDim':
        """Rotate quaternion dimensions"""
        new_qnode = QNode(*self.qnode.quat)
        new_qnode.rotate(axis, angle)
        return QDim(0, 0, 0, 0, self.field_base)._replace_qnode(new_qnode)
    
    def _replace_qnode(self, qnode: QNode) -> 'QDim':
        """Internal method to replace qnode"""
        result = QDim(0, 0, 0, 0, self.field_base)
        result.qnode = qnode
        return result
    
    def field_reduce(self, value: Union[int, float]) -> int:
        """Reduce value to F_2^n field"""
        if self.field_base == 2:
            return int(value) % 2
        return int(value) % (2 ** self.field_base)
    
    def __mul__(self, other: 'QDim') -> 'QDim':
        """Quaternion multiplication for dimension composition"""
        result_quat = quat_mult_vec(self.qnode.quat, other.qnode.quat)
        new_qnode = QNode(*result_quat)
        return QDim(0, 0, 0, 0, max(self.field_base, other.field_base))._replace_qnode(new_qnode)


class CombitArray:
    """
    Quaternion-Dimensional Array with Pascal Order Separation
    
    Revolutionary array type that keeps:
    - Pascal order: Mathematical combinatorial depth (invariant)
    - Quaternion dims: Geometric spatial encoding (rotatable)
    - Field base: Algebraic structure for type safety
    """
    
    def __init__(self, 
                 data: Union[np.ndarray, jnp.ndarray, list], 
                 pascal_order: int,
                 qdim: QDim):
        self.data = jnp.array(data)
        self.pascal_order = pascal_order  # Fixed combinatorial invariant
        self.qdim = qdim                  # Rotatable quaternion dimensions
        
        # Validate dimensional consistency
        self._validate_dimensions()
    
    def _validate_dimensions(self):
        """Ensure array shape matches quaternion dimensions"""
        expected_dims = self.qdim.geometric_dims
        actual_shape = self.data.shape
        
        # Pad or reshape data to match quaternion dimensions
        if len(actual_shape) != len(expected_dims):
            # Flatten and reshape to quaternion dimensions
            flat_data = self.data.flatten()
            target_size = np.prod(expected_dims)
            
            if len(flat_data) < target_size:
                # Pad with zeros
                padded = jnp.zeros(target_size)
                padded = padded.at[:len(flat_data)].set(flat_data)
                self.data = padded.reshape(expected_dims)
            else:
                # Truncate
                self.data = flat_data[:target_size].reshape(expected_dims)
        
    @property
    def qshape(self) -> Tuple[int, ...]:
        """Quaternion-encoded shape"""
        return self.qdim.geometric_dims
    
    @property
    def field_type(self) -> str:
        """F_2^n field type"""
        return f"F_{2**self.qdim.field_base}"
    
    def rotate_dims(self, axis: str, angle: float) -> 'CombitArray':
        """Rotate array dimensions in quaternion space"""
        new_qdim = self.qdim.rotate(axis, angle)
        
        # Create new array with rotated dimensions
        result = CombitArray(self.data, self.pascal_order, new_qdim)
        return result
    
    def pascal_transform(self, target_order: Optional[int] = None) -> 'CombitArray':
        """Transform using Pascal triangle while preserving quaternion dims"""
        from .combit import pascal_zeta
        
        if target_order is None:
            target_order = self.pascal_order + 1
            
        # Apply Pascal transformation to flattened data
        flat_data = self.data.flatten()
        dim = len(flat_data)
        
        # Generate Pascal matrix
        pascal_matrix = pascal_zeta(dim, target_order)
        
        # Apply transformation
        transformed = pascal_matrix @ flat_data
        
        # Reshape back to quaternion dimensions
        result = CombitArray(transformed, target_order, self.qdim)
        return result
    
    def geometric_convolve(self, kernel: Union[np.ndarray, 'CombitArray']) -> 'CombitArray':
        """Convolution that respects quaternion geometry"""
        if isinstance(kernel, CombitArray):
            kernel_data = kernel.data
        else:
            kernel_data = jnp.array(kernel)
        
        # Perform convolution along quaternion-defined dimensions
        # Use the primary dimension for convolution
        convolved = jnp.convolve(self.data.flatten(), kernel_data.flatten(), mode='same')
        
        result = CombitArray(convolved, self.pascal_order, self.qdim)
        return result
    
    def field_reduce(self) -> 'CombitArray':
        """Reduce all values to F_2^n field"""
        reduced_data = jnp.array([
            self.qdim.field_reduce(x) for x in self.data.flatten()
        ]).reshape(self.data.shape)
        
        return CombitArray(reduced_data, self.pascal_order, self.qdim)
    
    def compose(self, other: 'CombitArray') -> 'CombitArray':
        """Compose two CombitArrays via quaternion multiplication"""
        # Multiply quaternion dimensions
        new_qdim = self.qdim * other.qdim
        
        # Combine Pascal orders (take maximum for stability)
        new_order = max(self.pascal_order, other.pascal_order)
        
        # Element-wise composition of data
        # Broadcast to compatible shapes
        self_flat = self.data.flatten()
        other_flat = other.data.flatten()
        
        min_len = min(len(self_flat), len(other_flat))
        composed_data = self_flat[:min_len] * other_flat[:min_len]
        
        return CombitArray(composed_data, new_order, new_qdim)
    
    def __repr__(self) -> str:
        return (f"CombitArray(shape={self.qshape}, "
                f"pascal_order={self.pascal_order}, "
                f"field={self.field_type}, "
                f"quat={self.qdim.quat})")
    
    def __getitem__(self, key):
        """Get item while preserving type structure"""
        sliced_data = self.data[key]
        # Note: slicing might change dimensions, so we create new qdim
        # This is where geometric collision detection would happen
        return CombitArray(sliced_data, self.pascal_order, self.qdim)
    
    def __setitem__(self, key, value):
        """Set item with type checking"""
        if isinstance(value, CombitArray):
            # Type check: ensure compatible Pascal orders and field types
            if value.pascal_order != self.pascal_order:
                raise TypeError(f"Pascal order mismatch: {value.pascal_order} != {self.pascal_order}")
            if value.qdim.field_base != self.qdim.field_base:
                raise TypeError(f"Field type mismatch: {value.field_type} != {self.field_type}")
            
            self.data = self.data.at[key].set(value.data)
        else:
            # Convert scalar to field type
            field_value = self.qdim.field_reduce(value)
            self.data = self.data.at[key].set(field_value)


# Factory functions for common quaternion dimension patterns
def identity_qdim(field_base: int = 2) -> QDim:
    """Identity quaternion dimension (1, 0, 0, 0)"""
    return QDim(1, 0, 0, 0, field_base)

def rotation_qdim(axis: str, angle: float, field_base: int = 2) -> QDim:
    """Create rotated quaternion dimension"""
    qdim = identity_qdim(field_base)
    return qdim.rotate(axis, angle)

def balanced_qdim(field_base: int = 2) -> QDim:
    """Balanced quaternion dimension (0.5, 0.5, 0.5, 0.5)"""
    return QDim(0.5, 0.5, 0.5, 0.5, field_base)


# Type-safe array creation functions
def zeros_qarray(pascal_order: int, qdim: QDim) -> CombitArray:
    """Create zeros array with quaternion dimensions"""
    shape = qdim.geometric_dims
    data = jnp.zeros(shape)
    return CombitArray(data, pascal_order, qdim)

def ones_qarray(pascal_order: int, qdim: QDim) -> CombitArray:
    """Create ones array with quaternion dimensions"""
    shape = qdim.geometric_dims
    data = jnp.ones(shape)
    return CombitArray(data, pascal_order, qdim)

def random_qarray(pascal_order: int, qdim: QDim, key: Optional[Any] = None) -> CombitArray:
    """Create random array with quaternion dimensions"""
    if key is None:
        import jax.random as jrandom
        key = jrandom.PRNGKey(42)
    
    shape = qdim.geometric_dims
    data = jax.random.normal(key, shape)
    return CombitArray(data, pascal_order, qdim)


__all__ = [
    "CombitArray", "QDim", 
    "identity_qdim", "rotation_qdim", "balanced_qdim",
    "zeros_qarray", "ones_qarray", "random_qarray"
]