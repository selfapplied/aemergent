#!/usr/bin/env python3
"""
SigTest - Signal-Based Testing Framework

Revolutionary testing system that closes the loop between:
- Print output (demo signals)
- Doctests (executable signal processing)
- Pattern matching (assertion via signal analysis)

The type system validates itself through its own signal processing mechanisms.
Demo output becomes test signals, pattern matching becomes assertions.

Architecture:
- DemoCapture: Capture demo output as TextSignal
- SignalAssertion: Pattern-based assertions using AttentionMasks
- DoctestProcessor: Execute doctests as signal operations
- TypeValidator: Self-validating type system through signal analysis

Example:
    >>> from aemergent.sigtest import DemoCapture, SignalAssertion
    >>> # Capture demo output as signal
    >>> capture = DemoCapture.run_demo(demo_function)
    >>> # Assert patterns in the signal
    >>> assert capture.contains_pattern("✅.*success")
    >>> # Validate type behavior through signal analysis
    >>> capture.validate_type_behavior(CombitArray)
"""

import io
import sys
import re
import numpy as np
import jax.numpy as jnp
from typing import Callable, List, Dict, Optional, Any, Union, Tuple
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from .textsignal import TextSignal, AttentionMask
from .combinion import CombitArray, QDim


@dataclass
class SignalPattern:
    """Pattern for signal-based assertions"""
    name: str
    pattern: str
    expected_count: Optional[int] = None
    intensity_threshold: float = 0.5
    description: str = ""


class DemoCapture:
    """
    Capture demo output as TextSignal for analysis
    
    Converts print output into signal data for pattern matching
    and type validation through signal processing.
    """
    
    def __init__(self, stdout_signal: TextSignal, stderr_signal: Optional[TextSignal] = None):
        self.stdout_signal = stdout_signal
        self.stderr_signal = stderr_signal
        self.combined_signal = self._combine_signals()
        
    def _combine_signals(self) -> TextSignal:
        """Combine stdout and stderr into single signal"""
        combined_text = self.stdout_signal.text
        if self.stderr_signal and self.stderr_signal.text:
            combined_text += "\n--- STDERR ---\n" + self.stderr_signal.text
        return TextSignal(combined_text)
    
    @classmethod
    def run_demo(cls, demo_func: Callable) -> 'DemoCapture':
        """Run demo function and capture output as signals"""
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                demo_func()
            except Exception as e:
                print(f"Demo failed: {e}", file=sys.stderr)
        
        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        
        stdout_signal = TextSignal(stdout_text) if stdout_text else TextSignal("")
        stderr_signal = TextSignal(stderr_text) if stderr_text else None
        
        return cls(stdout_signal, stderr_signal)
    
    def contains_pattern(self, pattern: str, min_count: int = 1) -> bool:
        """Check if signal contains pattern (signal-based assertion)"""
        mask = AttentionMask.from_pattern(pattern, self.combined_signal.text)
        return len(mask.regions) >= min_count
    
    def extract_values(self, pattern: str) -> List[str]:
        """Extract values matching pattern from signal"""
        matches = re.findall(pattern, self.combined_signal.text)
        return matches
    
    def signal_statistics(self) -> Dict[str, float]:
        """Get signal statistics for numerical validation"""
        signal = self.combined_signal.signal
        return {
            'mean': float(jnp.mean(signal)),
            'std': float(jnp.std(signal)),
            'min': float(jnp.min(signal)),
            'max': float(jnp.max(signal)),
            'length': len(signal)
        }
    
    def validate_success_patterns(self) -> bool:
        """Validate that demo shows success patterns"""
        success_patterns = [
            r'✅.*success',
            r'✅.*works',
            r'✅.*created',
            r'✅.*operational',
            r'COMPLETE!',
            r'DEMO COMPLETE'
        ]
        
        for pattern in success_patterns:
            if self.contains_pattern(pattern):
                return True
        return False
    
    def validate_no_errors(self) -> bool:
        """Validate that no error patterns exist in signal"""
        error_patterns = [
            r'❌.*Error',
            r'Exception:',
            r'Traceback',
            r'FAILED'
        ]
        
        for pattern in error_patterns:
            if self.contains_pattern(pattern):
                return False
        return True


class SignalAssertion:
    """
    Signal-based assertion system using pattern matching
    
    Transforms traditional assertions into signal processing operations
    on demo output, creating self-validating type demonstrations.
    """
    
    def __init__(self, capture: DemoCapture):
        self.capture = capture
        self.assertion_results: List[Tuple[str, bool, str]] = []
    
    def assert_pattern(self, pattern: str, description: str = "", min_count: int = 1) -> 'SignalAssertion':
        """Assert that pattern exists in signal with minimum count"""
        result = self.capture.contains_pattern(pattern, min_count)
        self.assertion_results.append((f"Pattern '{pattern}'", result, description))
        
        if not result:
            raise AssertionError(f"Pattern '{pattern}' not found (min_count={min_count}): {description}")
        
        return self
    
    def assert_signal_range(self, min_val: float, max_val: float, description: str = "") -> 'SignalAssertion':
        """Assert signal values are within range"""
        stats = self.capture.signal_statistics()
        result = min_val <= stats['mean'] <= max_val
        self.assertion_results.append((f"Signal range [{min_val}, {max_val}]", result, description))
        
        if not result:
            raise AssertionError(f"Signal mean {stats['mean']:.2f} not in range [{min_val}, {max_val}]: {description}")
        
        return self
    
    def assert_type_creation(self, type_name: str) -> 'SignalAssertion':
        """Assert that type was successfully created"""
        patterns = [
            f"✅.*{type_name}.*creat",
            f"{type_name}.*success",
            f"✅.*{type_name}"
        ]
        
        found = any(self.capture.contains_pattern(p) for p in patterns)
        self.assertion_results.append((f"Type creation: {type_name}", found, f"Validates {type_name} instantiation"))
        
        if not found:
            raise AssertionError(f"Type creation for {type_name} not validated in output")
        
        return self
    
    def assert_signal_processing(self) -> 'SignalAssertion':
        """Assert that signal processing operations occurred"""
        patterns = [
            r'Signal.*stats',
            r'mean.*std',
            r'convol.*filter',
            r'After.*filtering',
            r'After.*rotation'
        ]
        
        found = any(self.capture.contains_pattern(p) for p in patterns)
        self.assertion_results.append(("Signal processing", found, "Validates signal operations"))
        
        if not found:
            raise AssertionError("No evidence of signal processing in output")
        
        return self
    
    def assert_quaternion_operations(self) -> 'SignalAssertion':
        """Assert quaternion operations were performed"""
        patterns = [
            r'quaternion',
            r'rotation',
            r'quat.*\[.*\]',
            r'Pascal order preserved'
        ]
        
        found = any(self.capture.contains_pattern(p) for p in patterns)
        self.assertion_results.append(("Quaternion operations", found, "Validates quaternion math"))
        
        if not found:
            raise AssertionError("No evidence of quaternion operations in output")
        
        return self
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of all assertions"""
        total = len(self.assertion_results)
        passed = sum(1 for _, result, _ in self.assertion_results if result)
        
        return {
            'total_assertions': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total if total > 0 else 0.0,
            'details': self.assertion_results
        }


class DoctestProcessor:
    """
    Process doctests as signal operations
    
    Converts doctest examples into signal processing operations,
    creating executable documentation through signal analysis.
    """
    
    def __init__(self, module_signal: TextSignal):
        self.module_signal = module_signal
        self.doctest_results: List[Dict[str, Any]] = []
    
    @classmethod
    def from_module(cls, module) -> 'DoctestProcessor':
        """Create processor from module docstrings"""
        docstrings = cls._extract_docstrings(module)
        combined_text = "\n".join(docstrings)
        signal = TextSignal(combined_text)
        return cls(signal)
    
    @staticmethod
    def _extract_docstrings(module) -> List[str]:
        """Extract all docstrings from module"""
        docstrings = []
        
        # Module docstring
        if hasattr(module, '__doc__') and module.__doc__:
            docstrings.append(module.__doc__)
        
        # Class and function docstrings
        for name in dir(module):
            obj = getattr(module, name)
            if hasattr(obj, '__doc__') and obj.__doc__:
                docstrings.append(obj.__doc__)
        
        return docstrings
    
    def extract_examples(self) -> List[Tuple[str, str]]:
        """Extract code examples from docstrings using signal processing"""
        # Pattern for doctest examples
        example_pattern = r'>>> (.+?)(?=\n\s*(?:>>>|\.\.\.|[A-Za-z]|\Z))'
        expected_pattern = r'>>> .+?\n(.+?)(?=\n\s*(?:>>>|\.\.\.|[A-Za-z]|\Z))'
        
        code_mask = AttentionMask.from_pattern(example_pattern, self.module_signal.text)
        expected_mask = AttentionMask.from_pattern(expected_pattern, self.module_signal.text)
        
        examples = []
        for region in code_mask.regions:
            code = self.module_signal.text[region[0]:region[1]]
            # Extract actual code after >>>
            code = re.sub(r'>>> ', '', code)
            examples.append((code.strip(), ""))  # Simplified for now
        
        return examples
    
    def validate_examples(self) -> bool:
        """Validate that examples execute correctly"""
        examples = self.extract_examples()
        
        for code, expected in examples:
            try:
                # Create namespace for execution
                namespace = {}
                exec(f"from aemergent.textsignal import *", namespace)
                exec(f"from aemergent.combinion import *", namespace)
                
                # Execute example
                result = eval(code, namespace)
                
                self.doctest_results.append({
                    'code': code,
                    'expected': expected,
                    'result': str(result),
                    'success': True
                })
                
            except Exception as e:
                self.doctest_results.append({
                    'code': code,
                    'expected': expected,
                    'result': str(e),
                    'success': False
                })
        
        return all(r['success'] for r in self.doctest_results)


class TypeValidator:
    """
    Self-validating type system through signal analysis
    
    Uses the type system's own signal processing capabilities
    to validate its behavior and correctness.
    """
    
    def __init__(self):
        self.validation_results: Dict[str, bool] = {}
    
    def validate_combit_array(self) -> bool:
        """Validate CombitArray behavior through signal analysis"""
        try:
            from .combinion import CombitArray, QDim, identity_qdim
            
            # Create test instance and capture its string representation
            qdim = identity_qdim()
            arr = CombitArray([1, 2, 3], pascal_order=2, qdim=qdim)
            
            # Convert representation to signal for analysis
            repr_signal = TextSignal(str(arr))
            
            # Validate expected patterns in representation
            patterns = [
                r'CombitArray',
                r'pascal_order=2',
                r'shape=\(',
                r'quat=\['
            ]
            
            for pattern in patterns:
                mask = AttentionMask.from_pattern(pattern, repr_signal.text)
                if len(mask.regions) == 0:
                    self.validation_results['combit_array'] = False
                    return False
            
            # Test rotation preserves Pascal order
            rotated = arr.rotate_dims('x', np.pi/4)
            if rotated.pascal_order != arr.pascal_order:
                self.validation_results['combit_array'] = False
                return False
            
            self.validation_results['combit_array'] = True
            return True
            
        except Exception:
            self.validation_results['combit_array'] = False
            return False
    
    def validate_text_signal(self) -> bool:
        """Validate TextSignal behavior through self-analysis"""
        try:
            from .textsignal import TextSignal, AttentionMask
            
            # Create test signal
            test_text = "def test(): pass\nclass Test: pass"
            signal = TextSignal(test_text)
            
            # Validate signal properties
            if len(signal.signal) != len(test_text):
                self.validation_results['text_signal'] = False
                return False
            
            # Test mask creation
            mask = AttentionMask.from_pattern(r'def \w+', test_text)
            if len(mask.regions) == 0:
                self.validation_results['text_signal'] = False
                return False
            
            # Test signal processing
            filtered = signal.filter_signal()
            if len(filtered.signal) != len(signal.signal):
                self.validation_results['text_signal'] = False
                return False
            
            self.validation_results['text_signal'] = True
            return True
            
        except Exception:
            self.validation_results['text_signal'] = False
            return False
    
    def validate_all(self) -> Dict[str, bool]:
        """Validate all components of the type system"""
        validations = [
            ('combit_array', self.validate_combit_array),
            ('text_signal', self.validate_text_signal),
        ]
        
        for name, validator in validations:
            validator()
        
        return self.validation_results


# Convenience functions for signal-based testing
def run_demo_test(demo_func: Callable, *patterns: str) -> bool:
    """Run demo and test for patterns"""
    capture = DemoCapture.run_demo(demo_func)
    assertion = SignalAssertion(capture)
    
    try:
        for pattern in patterns:
            assertion.assert_pattern(pattern)
        return True
    except AssertionError:
        return False

def validate_type_demo(demo_func: Callable, type_name: str) -> bool:
    """Validate type demonstration through signal analysis"""
    capture = DemoCapture.run_demo(demo_func)
    assertion = SignalAssertion(capture)
    
    try:
        assertion.assert_type_creation(type_name)
        assertion.assert_signal_processing()
        return assertion.capture.validate_success_patterns()
    except AssertionError:
        return False

def self_validate_system() -> Dict[str, Any]:
    """Self-validate the entire system using its own mechanisms"""
    validator = TypeValidator()
    results = validator.validate_all()
    
    return {
        'system_validation': results,
        'all_passed': all(results.values()),
        'summary': f"Validated {len(results)} components, {sum(results.values())} passed"
    }


__all__ = [
    "DemoCapture", "SignalAssertion", "DoctestProcessor", "TypeValidator",
    "SignalPattern", "run_demo_test", "validate_type_demo", "self_validate_system"
]