#!/usr/bin/env python3
"""
Combit Demo - Combinatorial Mathematics and Template Rendering

Demonstrates combinatorial operations and template capabilities.
"""

import numpy as np
from src.combit import Combit

def demo_combit():
    """Demo Combit combinatorial mathematics"""
    print("🔢 COMBIT DEMO - COMBINATORIAL MATHEMATICS")
    print("=" * 60)
    
    # Create Combit instance
    names = ['math', 'template', 'render']
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cb = Combit(names, state)
    
    # Test basic properties
    print("📊 COMBIT PROPERTIES:")
    print(f"Order: {cb.order}")
    print(f"Dimension: {cb.dim}")
    print(f"Energy: {cb.energy:.4f}")
    print(f"State: {cb.state}")
    
    # Test basic functionality
    print("\n🎯 BASIC FUNCTIONALITY:")
    print(f"Attractors: {cb.attractors}")
    print(f"Roots: {cb.roots}")
    
    # Test function call
    print("\n📈 FUNCTION APPROXIMATION:")
    X = np.array([1.0, 2.0, 3.0])
    result = cb(X)
    print(f"Input: {X}")
    print(f"Output: {result}")
    
    # Test capacity
    print("\n⚡ CAPACITY:")
    capacity = cb.capacity
    print(f"Capacity: {capacity}")

if __name__ == "__main__":
    demo_combit() 
