#!/usr/bin/env python3
"""
Half-Open Codec System - Core Interfaces

This module defines the core protocols and types for the half-open codec system
where Q1-Q3 quadrants contain closed codec types and Q4 enables generative extension.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Set, List, Optional, Callable, TypeVar, Generic, Union, Tuple, Protocol
from enum import Enum, auto
from pathlib import Path

# Type variables for generic codec operations
T = TypeVar('T')
U = TypeVar('U')

class Quadrant(Enum):
    """Type quadrants in the half-open system"""
    Q1 = auto()  # Magnitude encodings - closed
    Q2 = auto()  # Pointer/LZ encodings - closed  
    Q3 = auto()  # Quaternion-wavelet encodings - closed
    Q4 = auto()  # Generative extension space - open

class Codec(Protocol):
    """Protocol defining codec interface with energy bounds"""
    name: str
    quadrant: Quadrant
    energy_bounds: Tuple[float, float]
    
    def encode(self, data: bytes) -> bytes:
        """Encode input data"""
        ...
    
    def decode(self, data: bytes) -> bytes:
        """Decode input data"""
        ...
    
    def energy_cost(self, data: bytes) -> float:
        """Calculate energy cost for processing data"""
        ...

@dataclass
class TestResult:
    """Result of running a unit test on a codec"""
    test_name: str
    passed: bool
    energy_cost: float

class CodecRegistry:
    """Registry for codec discovery and validation"""
    
    def __init__(self):
        self.codecs: Dict[str, Codec] = {}
        self.quadrant_codecs: Dict[Quadrant, Set[str]] = {
            q: set() for q in Quadrant
        }
    
    def register(self, codec: Codec) -> None:
        """Register a codec in the system"""
        self.codecs[codec.name] = codec
        self.quadrant_codecs[codec.quadrant].add(codec.name)
    
    def get_codec(self, name: str) -> Optional[Codec]:
        """Retrieve a codec by name"""
        return self.codecs.get(name)
    
    def get_quadrant_codecs(self, quadrant: Quadrant) -> List[Codec]:
        """Get all codecs in a specific quadrant"""
        names = self.quadrant_codecs[quadrant]
        return [self.codecs[name] for name in names if name in self.codecs]
    
    def validate_completeness(self, quadrant: Quadrant) -> bool:
        """Check if a quadrant has complete codec coverage"""
        # Q1-Q3 require at least one codec, Q4 is always incomplete (generative)
        if quadrant == Quadrant.Q4:
            return False
        return len(self.quadrant_codecs[quadrant]) > 0

class EnergyConstraint:
    """Represents an energy constraint in the codec system"""
    
    def __init__(self, name: str, weight: float, constraint_func: Callable[[TestResult], bool]):
        self.name = name
        self.weight = weight
        self.constraint_func = constraint_func
    
    def evaluate(self, result: TestResult) -> float:
        """Evaluate constraint and return energy contribution"""
        if self.constraint_func(result):
            return self.weight * result.energy_cost
        else:
            return self.weight * float('inf')  # Constraint violation

class CodecValidator:
    """Validates codecs against energy constraints"""
    
    def __init__(self):
        self.constraints: List[EnergyConstraint] = []
    
    def add_constraint(self, constraint: EnergyConstraint) -> None:
        """Add an energy constraint"""
        self.constraints.append(constraint)
    
    def validate_codec(self, codec: Codec, test_data: bytes) -> List[TestResult]:
        """Run validation tests on a codec"""
        results = []
        
        # Basic roundtrip test
        try:
            encoded = codec.encode(test_data)
            decoded = codec.decode(encoded)
            roundtrip_passed = decoded == test_data
            results.append(TestResult("roundtrip", roundtrip_passed, 0.1))
        except Exception:
            results.append(TestResult("roundtrip", False, float('inf')))
        
        # Energy bounds test
        try:
            energy = codec.energy_cost(test_data)
            min_energy, max_energy = codec.energy_bounds
            bounds_passed = min_energy <= energy <= max_energy
            results.append(TestResult("energy_bounds", bounds_passed, energy))
        except Exception:
            results.append(TestResult("energy_bounds", False, float('inf')))
        
        return results
    
    def compute_total_energy(self, results: List[TestResult]) -> float:
        """Compute total energy across all constraints"""
        total = 0.0
        for result in results:
            for constraint in self.constraints:
                total += constraint.evaluate(result)
        return total

# Global registry instance
REGISTRY = CodecRegistry()

def register_codec(codec: Codec) -> None:
    """Convenience function to register a codec"""
    REGISTRY.register(codec)