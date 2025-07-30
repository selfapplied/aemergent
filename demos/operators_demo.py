#!/usr/bin/env python3
"""
Operators Demo - Signal Processing and Mathematical Operators

Demonstrates small, focused operators for signal processing.
"""

import numpy as np
from src.operators import convolve_op, compose, scale_op, shift_op

def demo_operators():
    """Demo mathematical operators"""
    print("⚡ OPERATORS DEMO - SIGNAL PROCESSING")
    print("=" * 60)
    
    # Create test signal
    signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    print(f"Original signal: {signal}")
    
    # Test convolution operator
    print("\n🔍 CONVOLUTION OPERATOR:")
    kernel = np.array([0.25, 0.5, 0.25])
    conv_op = convolve_op(kernel)
    convolved = conv_op(signal)
    print(f"Kernel: {kernel}")
    print(f"Convolved: {convolved}")
    
    # Test composition
    print("\n🔄 OPERATOR COMPOSITION:")
    scale_shift = compose(scale_op(2.0), shift_op(1.0))
    transformed = scale_shift(signal)
    print(f"Scale(2) + Shift(1): {transformed}")
    
    # Test chaining
    print("\n⛓️ OPERATOR CHAINING:")
    chain_op = compose(scale_op(0.5), shift_op(-1), convolve_op(kernel))
    chained = chain_op(signal)
    print(f"Chain result: {chained}")

if __name__ == "__main__":
    demo_operators() 
