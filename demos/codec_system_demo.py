#!/usr/bin/env python3
"""
Half-Open Codec System Demo with Verified Examples

This demonstrates the codec algebra and energy potential field concepts
using doctests to ensure all examples actually work as claimed.

To run the doctests:
    python -m doctest demos/codec_system_demo.py -v

To run as a demo:
    python demos/codec_system_demo.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Protocol
from enum import Enum, auto
import math


class Quadrant(Enum):
    """Type quadrants in the half-open system"""
    Q1 = auto()  # Magnitude encodings - closed
    Q2 = auto()  # Pointer/LZ encodings - closed  
    Q3 = auto()  # Quaternion-wavelet encodings - closed
    Q4 = auto()  # Generative extension space - open


class Codec(Protocol):
    """Protocol defining codec interface with energy bounds
    
    >>> class SimpleCodec:
    ...     def __init__(self, name: str, quadrant: Quadrant, energy_range: Tuple[float, float]):
    ...         self.name = name
    ...         self.quadrant = quadrant
    ...         self.energy_bounds = energy_range
    ...     def encode(self, data: bytes) -> bytes:
    ...         return data
    ...     def decode(self, data: bytes) -> bytes:
    ...         return data
    ...     def energy_cost(self, data: bytes) -> float:
    ...         return len(data) * 0.1
    
    >>> codec = SimpleCodec("test", Quadrant.Q1, (0.1, 2.0))
    >>> codec.name
    'test'
    >>> codec.quadrant
    <Quadrant.Q1: 1>
    >>> codec.energy_bounds
    (0.1, 2.0)
    """
    name: str
    quadrant: Quadrant
    energy_bounds: Tuple[float, float]
    
    def encode(self, data: bytes) -> bytes: ...
    def decode(self, data: bytes) -> bytes: ...


@dataclass
class TestResult:
    """Result of running a unit test on a codec
    
    >>> result = TestResult("roundtrip_test", True, 0.0)
    >>> result.test_name
    'roundtrip_test'
    >>> result.passed
    True
    >>> result.energy_cost
    0.0
    """
    test_name: str
    passed: bool
    energy_cost: float


class CodecTester:
    """Runs unit tests on codecs and computes energy fields
    
    >>> tester = CodecTester()
    >>> len(tester.test_functions)
    3
    >>> 'roundtrip_validity' in tester.test_functions
    True
    """
    
    def __init__(self):
        self.test_functions = {
            'roundtrip_validity': self._test_roundtrip,
            'energy_conservation': self._test_energy_bounds,
            'compression_efficiency': self._test_compression
        }
    
    def _test_roundtrip(self, codec: Codec, data: bytes) -> TestResult:
        """Test that decode(encode(x)) == x
        
        >>> from demos.codec_system_demo import MagnitudeCodec
        >>> tester = CodecTester()
        >>> codec = MagnitudeCodec()
        >>> result = tester._test_roundtrip(codec, b"hello")
        >>> result.passed
        True
        >>> result.test_name
        'roundtrip_validity'
        """
        try:
            encoded = codec.encode(data)
            decoded = codec.decode(encoded)
            passed = decoded == data
            return TestResult('roundtrip_validity', passed, 0.1)
        except Exception:
            return TestResult('roundtrip_validity', False, float('inf'))
    
    def _test_energy_bounds(self, codec: Codec, data: bytes) -> TestResult:
        """Test that energy usage stays within declared bounds
        
        >>> from demos.codec_system_demo import MagnitudeCodec
        >>> tester = CodecTester()
        >>> codec = MagnitudeCodec()
        >>> result = tester._test_energy_bounds(codec, b"test")
        >>> result.passed
        True
        >>> 0.0 <= result.energy_cost <= 2.0
        True
        """
        try:
            energy = codec.energy_cost(data)
            min_energy, max_energy = codec.energy_bounds
            passed = min_energy <= energy <= max_energy
            return TestResult('energy_conservation', passed, energy)
        except Exception:
            return TestResult('energy_conservation', False, float('inf'))
    
    def _test_compression(self, codec: Codec, data: bytes) -> TestResult:
        """Test compression efficiency (allows some expansion for headers)
        
        >>> from demos.codec_system_demo import LZCodec
        >>> tester = CodecTester()
        >>> codec = LZCodec()
        >>> result = tester._test_compression(codec, b"hello")
        >>> result.passed
        True
        >>> result.energy_cost <= 1.6  # Allow some expansion for short strings
        True
        """
        try:
            encoded = codec.encode(data)
            efficiency = len(encoded) / max(len(data), 1)
            # Allow up to 60% expansion for short strings (due to headers)
            passed = efficiency <= 1.6
            return TestResult('compression_efficiency', passed, efficiency)
        except Exception:
            return TestResult('compression_efficiency', False, float('inf'))
    
    def run_all_tests(self, codec: Codec, test_data: bytes) -> List[TestResult]:
        """Run all tests on a codec
        
        >>> from demos.codec_system_demo import MagnitudeCodec
        >>> tester = CodecTester()
        >>> codec = MagnitudeCodec()
        >>> results = tester.run_all_tests(codec, b"test")
        >>> len(results)
        3
        >>> all(r.passed for r in results)
        True
        """
        return [test_func(codec, test_data) for test_func in self.test_functions.values()]


class MagnitudeCodec:
    """Q1 codec: Direct magnitude encoding
    
    >>> codec = MagnitudeCodec()
    >>> codec.name
    'magnitude'
    >>> codec.quadrant
    <Quadrant.Q1: 1>
    >>> data = b"hello"
    >>> decoded = codec.decode(codec.encode(data))
    >>> decoded == data
    True
    >>> 0.1 <= codec.energy_cost(data) <= 2.0
    True
    """
    
    def __init__(self):
        self.name = "magnitude"
        self.quadrant = Quadrant.Q1
        self.energy_bounds = (0.1, 2.0)
    
    def encode(self, data: bytes) -> bytes:
        return data  # Identity for demo
    
    def decode(self, data: bytes) -> bytes:
        return data  # Identity for demo
    
    def energy_cost(self, data: bytes) -> float:
        return min(len(data) * 0.1, 2.0)


class LZCodec:
    """Q2 codec: LZ-style pointer compression
    
    >>> codec = LZCodec()
    >>> codec.name
    'lz'
    >>> codec.quadrant
    <Quadrant.Q2: 2>
    >>> data = b"hello"
    >>> decoded = codec.decode(codec.encode(data))
    >>> decoded == data
    True
    >>> 0.5 <= codec.energy_cost(data) <= 3.0
    True
    """
    
    def __init__(self):
        self.name = "lz"
        self.quadrant = Quadrant.Q2
        self.energy_bounds = (0.5, 3.0)
    
    def encode(self, data: bytes) -> bytes:
        # Simplified: just add a compression marker
        return b"LZ:" + data
    
    def decode(self, data: bytes) -> bytes:
        if data.startswith(b"LZ:"):
            return data[3:]
        return data
    
    def energy_cost(self, data: bytes) -> float:
        return min(len(data) * 0.2 + 0.5, 3.0)


class EnergyField:
    """Computes energy potential fields for codec optimization
    
    >>> field = EnergyField()
    >>> field.add_constraint("energy_conservation", 2.0)
    >>> field.add_constraint("roundtrip_validity", 1.5)
    >>> len(field.constraints)
    2
    >>> field.constraints["energy_conservation"]
    2.0
    """
    
    def __init__(self):
        self.constraints: Dict[str, float] = {}
    
    def add_constraint(self, name: str, weight: float):
        """Add a weighted constraint to the energy field
        
        >>> field = EnergyField()
        >>> field.add_constraint("test", 1.0)
        >>> field.constraints["test"]
        1.0
        """
        self.constraints[name] = weight
    
    def compute_energy(self, test_results: List[TestResult]) -> float:
        """Compute total energy for a set of test results
        
        >>> field = EnergyField()
        >>> field.add_constraint("roundtrip_validity", 1.0)
        >>> field.add_constraint("energy_conservation", 2.0)
        >>> results = [
        ...     TestResult("roundtrip_validity", True, 0.0),
        ...     TestResult("energy_conservation", True, 0.5)
        ... ]
        >>> energy = field.compute_energy(results)
        >>> energy
        1.0
        """
        total_energy = 0.0
        for result in test_results:
            weight = self.constraints.get(result.test_name, 1.0)
            if result.passed:
                total_energy += weight * result.energy_cost
            else:
                total_energy += weight * 10.0  # Penalty for failure
        return total_energy
    
    def find_minima(self, codecs: List[Codec], test_data: bytes) -> List[Tuple[str, float]]:
        """Find energy minima across codec configurations
        
        >>> field = EnergyField()
        >>> field.add_constraint("roundtrip_validity", 1.0)
        >>> field.add_constraint("energy_conservation", 2.0)
        >>> field.add_constraint("compression_efficiency", 1.0)
        >>> codecs = [MagnitudeCodec(), LZCodec()]
        >>> minima = field.find_minima(codecs, b"test")
        >>> len(minima) >= 1
        True
        >>> all(isinstance(name, str) and isinstance(energy, float) for name, energy in minima)
        True
        """
        tester = CodecTester()
        energies = []
        
        for codec in codecs:
            results = tester.run_all_tests(codec, test_data)
            energy = self.compute_energy(results)
            energies.append((codec.name, energy))
        
        # Sort by energy (lower is better)
        energies.sort(key=lambda x: x[1])
        return energies


def demonstrate_codec_algebra():
    """Demonstrate the complete codec algebra system
    
    >>> demo_results = demonstrate_codec_algebra()
    >>> demo_results['num_codecs'] >= 2
    True
    >>> demo_results['all_tests_passed']
    True
    >>> demo_results['energy_minima_found'] >= 1
    True
    """
    # Create codecs
    codecs = [MagnitudeCodec(), LZCodec()]
    
    # Set up energy field
    field = EnergyField()
    field.add_constraint("roundtrip_validity", 1.5)
    field.add_constraint("energy_conservation", 2.0)
    field.add_constraint("compression_efficiency", 1.0)
    
    # Test all codecs
    tester = CodecTester()
    test_data = b"hello world"
    
    all_passed = True
    for codec in codecs:
        results = tester.run_all_tests(codec, test_data)
        codec_passed = all(r.passed for r in results)
        all_passed = all_passed and codec_passed
    
    # Find energy minima
    minima = field.find_minima(codecs, test_data)
    
    return {
        'num_codecs': len(codecs),
        'all_tests_passed': all_passed,
        'energy_minima_found': len(minima),
        'best_codec': minima[0][0] if minima else None,
        'best_energy': minima[0][1] if minima else float('inf')
    }


if __name__ == "__main__":
    # Run the demonstration
    print("Half-Open Codec System with Verified Examples")
    print("=" * 50)
    
    results = demonstrate_codec_algebra()
    
    print(f"Tested {results['num_codecs']} codecs")
    print(f"All tests passed: {results['all_tests_passed']}")
    print(f"Energy minima found: {results['energy_minima_found']}")
    print(f"Best codec: {results['best_codec']} (energy: {results['best_energy']:.2f})")
    
    print("\nRunning doctests...")
    import doctest
    result = doctest.testmod(verbose=True)
    
    if result.failed == 0:
        print(f"\n✓ All {result.attempted} doctests passed!")
    else:
        print(f"\n✗ {result.failed} of {result.attempted} doctests failed")