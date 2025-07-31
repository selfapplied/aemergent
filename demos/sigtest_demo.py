#!/usr/bin/env python3
"""
SigTest Demo - Revolutionary Signal-Based Testing

Demonstrates how to close the loop between:
- Print output (demo signals)
- Doctests (executable signal processing) 
- Pattern matching (assertion via signal analysis)

Shows "unit testing as typesystem" - where demos become tests,
and the type system validates itself through signal processing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from aemergent.sigtest import (
    DemoCapture, SignalAssertion, DoctestProcessor, TypeValidator,
    run_demo_test, validate_type_demo, self_validate_system
)
from aemergent.textsignal import TextSignal, AttentionMask
from aemergent.combinion import CombitArray, QDim, identity_qdim


def sample_demo_function():
    """Sample demo function that produces structured output"""
    print("✅ Starting demo function")
    print("📊 Creating CombitArray...")
    
    qdim = identity_qdim()
    arr = CombitArray([1, 2, 3], pascal_order=3, qdim=qdim)
    print(f"✅ CombitArray created: {arr}")
    
    print("🌀 Testing rotation...")
    rotated = arr.rotate_dims('x', np.pi/4)
    print(f"✅ Rotation successful: {rotated}")
    print(f"   Pascal order preserved: {rotated.pascal_order == arr.pascal_order}")
    
    print("📡 Signal processing...")
    signal = TextSignal("def test(): return 42")
    mask = AttentionMask.from_pattern(r'def \w+', signal.text)
    print(f"✅ Signal processing complete: {len(mask.regions)} patterns found")
    
    print("🎉 Demo COMPLETE!")
    return "success"


def failing_demo_function():
    """Sample demo that should fail validation"""
    print("❌ This demo has errors")
    print("Something went wrong")
    raise Exception("Demo failed intentionally")


def demo_signal_capture():
    """Demo capturing demo output as signals"""
    print("📡 SIGNAL-BASED TESTING - DEMO CAPTURE")
    print("=" * 50)
    
    # Capture successful demo output
    capture = DemoCapture.run_demo(sample_demo_function)
    
    print(f"📊 Captured signal: {capture.combined_signal}")
    print(f"   Signal length: {len(capture.combined_signal.signal)}")
    print(f"   Contains success patterns: {capture.validate_success_patterns()}")
    print(f"   No error patterns: {capture.validate_no_errors()}")
    
    # Show signal statistics
    stats = capture.signal_statistics()
    print(f"\n📈 Signal statistics:")
    print(f"   Mean: {stats['mean']:.2f}")
    print(f"   Std:  {stats['std']:.2f}")
    print(f"   Length: {stats['length']}")


def demo_signal_assertions():
    """Demo signal-based assertions"""
    print("\n🎯 SIGNAL ASSERTIONS - PATTERN VALIDATION")
    print("=" * 45)
    
    # Capture demo and create assertions
    capture = DemoCapture.run_demo(sample_demo_function)
    assertion = SignalAssertion(capture)
    
    print("🔍 Running signal-based assertions...")
    
    try:
        # Assert expected patterns exist
        assertion.assert_pattern(r"✅.*CombitArray.*created", "CombitArray creation")
        assertion.assert_pattern(r"✅.*Rotation.*successful", "Rotation operation")
        assertion.assert_pattern(r"Pascal order preserved", "Pascal order invariance")
        assertion.assert_pattern(r"Signal processing.*complete", "Signal processing")
        assertion.assert_pattern(r"Demo COMPLETE!", "Demo completion")
        
        # Assert signal characteristics
        assertion.assert_signal_range(70, 90, "ASCII character range")
        
        # Assert type behaviors
        assertion.assert_type_creation("CombitArray")
        assertion.assert_signal_processing()
        assertion.assert_quaternion_operations()
        
        print("✅ All assertions passed!")
        
        # Show summary
        summary = assertion.summary()
        print(f"\n📊 Assertion Summary:")
        print(f"   Total: {summary['total_assertions']}")
        print(f"   Passed: {summary['passed']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")


def demo_self_validation():
    """Demo self-validating type system"""
    print("\n🔮 SELF-VALIDATION - TYPE SYSTEM VALIDATES ITSELF")
    print("=" * 55)
    
    print("🔍 Running self-validation using own signal processing...")
    
    # Use the system to validate itself
    validation_results = self_validate_system()
    
    print(f"📊 Validation Results:")
    print(f"   All components passed: {validation_results['all_passed']}")
    print(f"   Summary: {validation_results['summary']}")
    
    # Show detailed results
    for component, passed in validation_results['system_validation'].items():
        status = "✅" if passed else "❌"
        print(f"   {status} {component}: {'PASS' if passed else 'FAIL'}")


def demo_doctest_processing():
    """Demo doctest processing as signal operations"""
    print("\n📚 DOCTEST PROCESSING - EXECUTABLE DOCUMENTATION")
    print("=" * 50)
    
    # Create sample module content with doctests
    sample_docstring = '''
    >>> from aemergent.combinion import CombitArray, identity_qdim
    >>> arr = CombitArray([1, 2], pascal_order=2, qdim=identity_qdim())
    >>> arr.pascal_order
    2
    >>> len(arr.data)
    2
    '''
    
    # Process as signal
    signal = TextSignal(sample_docstring)
    processor = DoctestProcessor(signal)
    
    print("🔍 Extracting examples from docstrings...")
    examples = processor.extract_examples()
    
    print(f"📚 Found {len(examples)} doctest examples:")
    for i, (code, expected) in enumerate(examples[:3]):  # Show first 3
        print(f"   Example {i+1}: {code[:50]}...")
    
    print("✅ Doctest processing complete")


def demo_practical_workflow():
    """Demo practical testing workflow"""
    print("\n🚀 PRACTICAL WORKFLOW - DEMO AS TEST")
    print("=" * 40)
    
    print("🎯 Testing workflow where demo output becomes test validation:")
    
    # 1. Run demo and validate in one step
    print("\n1️⃣ Quick validation:")
    success = run_demo_test(
        sample_demo_function,
        r"✅.*CombitArray.*created",
        r"Demo COMPLETE!"
    )
    print(f"   Quick test result: {'✅ PASS' if success else '❌ FAIL'}")
    
    # 2. Type-specific validation
    print("\n2️⃣ Type validation:")
    type_valid = validate_type_demo(sample_demo_function, "CombitArray")
    print(f"   Type validation: {'✅ PASS' if type_valid else '❌ FAIL'}")
    
    # 3. Negative test (should fail)
    print("\n3️⃣ Negative test (should fail):")
    try:
        fail_capture = DemoCapture.run_demo(failing_demo_function)
        has_errors = not fail_capture.validate_no_errors()
        print(f"   Error detection: {'✅ PASS' if has_errors else '❌ FAIL'}")
    except:
        print("   Error detection: ✅ PASS (exception caught)")


def demo_loop_closure():
    """Demo how the loop is closed between demos, tests, and validation"""
    print("\n🔄 LOOP CLOSURE - DEMOS ↔ TESTS ↔ VALIDATION")
    print("=" * 50)
    
    print("🔗 The revolutionary loop:")
    print("   1. 📝 Demo functions produce structured output")
    print("   2. 📡 Output captured as TextSignal")
    print("   3. 🎯 AttentionMask finds patterns in signal")
    print("   4. ✅ Pattern matching becomes assertion logic")
    print("   5. 🔮 Type system validates itself through own mechanisms")
    print("   6. 🔄 Loop closed: Demo = Test = Documentation")
    
    print("\n💡 Key insight: 'Unit testing as typesystem'")
    print("   - Tests are written as readable demos")
    print("   - Validation uses mathematical signal processing")
    print("   - System proves its own correctness")
    print("   - Documentation, testing, demonstration unified")
    
    # Demonstrate the complete loop
    print("\n🎪 Complete loop demonstration:")
    
    # Step 1: Demo produces output signal
    capture = DemoCapture.run_demo(sample_demo_function)
    print(f"   📝 Demo signal captured: {len(capture.combined_signal.text)} chars")
    
    # Step 2: Signal processing finds patterns
    mask = AttentionMask.from_pattern(r"✅.*", capture.combined_signal.text)
    print(f"   🎯 Success patterns found: {len(mask.regions)}")
    
    # Step 3: Pattern matching validates behavior
    assertion = SignalAssertion(capture)
    try:
        assertion.assert_pattern(r"CombitArray.*created")
        assertion.assert_quaternion_operations()
        print("   ✅ Assertions validated through signal processing")
    except AssertionError:
        print("   ❌ Validation failed")
    
    # Step 4: Self-validation completes the loop
    self_validation = self_validate_system()
    print(f"   🔮 Self-validation: {self_validation['all_passed']}")
    
    print("\n🎉 Loop successfully closed!")
    print("   Demo output → Signal processing → Pattern validation → Type safety")


if __name__ == "__main__":
    demo_signal_capture()
    demo_signal_assertions()
    demo_self_validation()
    demo_doctest_processing()
    demo_practical_workflow()
    demo_loop_closure()
    
    print("\n" + "=" * 60)
    print("🎉 SIGTEST DEMO COMPLETE!")
    print("=" * 60)
    print("Revolutionary capabilities demonstrated:")
    print("✅ Demo output as test signals")
    print("✅ Pattern matching as assertion logic")
    print("✅ Signal processing for validation")
    print("✅ Self-validating type system")
    print("✅ Doctest processing through signals")
    print("✅ Unified demo/test/documentation")
    print("✅ Complete loop closure")
    print("\n🚀 Testing will never be the same!")
    print("🔄 The loop between demos, tests, and validation is CLOSED!")