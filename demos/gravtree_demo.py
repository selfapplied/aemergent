#!/usr/bin/env python3
"""
GravTree Demo - Quantum Quadtree Text Processing

Demonstrates geometric convolution and pattern detection in text space.
"""

from src.gravtree import GravTree

def demo_gravtree():
    """Demo GravTree quantum quadtree text processing"""
    print("🌳 GRAVTREE DEMO - QUANTUM QUADTREE TEXT PROCESSING")
    print("=" * 60)
    
    # Create GravTree instance
    gt = GravTree()
    
    # Test text processing
    text = "Hello, world! This is a test of quantum quadtree processing."
    print(f"Input text: {text}")
    
    # Detect patterns
    patterns = gt.detect_patterns(text)
    print(f"Found {len(patterns)} patterns")
    
    # Show pattern details
    for i, pattern in enumerate(patterns[:5]):  # Show first 5 patterns
        print(f"Pattern {i+1}: {pattern}")
    
    # Test geometric convolution
    print("\n🔍 GEOMETRIC CONVOLUTION:")
    conv_result = gt.geometric_convolution(text)
    print(f"Convolution result: {conv_result[:100]}...")
    
    # Test quaternion transformations
    print("\n🌀 QUATERNION TRANSFORMATIONS:")
    quat_result = gt.quaternion_transform(text)
    print(f"Quaternion result: {quat_result[:100]}...")

if __name__ == "__main__":
    demo_gravtree() 
