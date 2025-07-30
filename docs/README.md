# Aemergent

A computational metaphysics framework with quantum quadtree text processing.

## What is Aemergent?

Aemergent is a framework for exploring computational metaphysics through:
- **GravTree**: Quantum quadtree text processing with geometric convolution
- **Combit**: Combinatorial mathematics and template rendering
- **Operators**: Mathematical operators for signal processing
- **Pascell**: Cellular automata for Pascal's Triangle with prime number generation

## Installation

```bash
uv add aemergent
```

## Quick Start

```python
from aemergent.src.gravtree import GravTree
from aemergent.src.pascell import generate_ca_pascal, prime_generator_ca

# Create a quantum quadtree text processor
gt = GravTree()

# Process text with geometric convolution
text = "Hello, world!"
patterns = gt.detect_patterns(text)
print(f"Found {len(patterns)} patterns")

# Generate Pascal's Triangle with cellular automata
ca_triangle = generate_ca_pascal(20, use_mod=True, mod=10)

# Find prime numbers using palindromic patterns
primes, palindromes = prime_generator_ca(100)
print(f"Found {len(primes)} primes: {primes}")
```

## Components

### GravTree
Quantum quadtree implementation that combines:
- **Quad tree structure** for hierarchical geometric operations
- **Quantum quaternions** for spatial transformations
- **Geometric convolution** for pattern detection in text space

### Combit
Combinatorial mathematics with template rendering and capability composition.

### Operators
Mathematical operators for signal processing and transformations.

### Pascell
Cellular automata for Pascal's Triangle with:
- **CA evolution rules** for building triangles from (0,0)
- **Modular arithmetic** with overflow evolution
- **Prime number generation** using palindromic patterns
- **Prime mask analysis** for pattern identification

## Development

```bash
# Install development dependencies
uv add --dev aemergent

# Run tests
pytest

# Run demos
python demos/gravtree_demo.py
```

## License

MIT License 