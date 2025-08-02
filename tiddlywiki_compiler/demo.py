#!/usr/bin/env python3
"""
Demo script for TiddlyWiki Compiler
Shows how to compile files into a TiddlyWiki with sample content
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from main import TiddlyWikiCompiler


def create_sample_content():
    """Create sample content for demonstration"""
    sample_dir = Path("sample_content")
    sample_dir.mkdir(exist_ok=True)
    
    # Copy the sample blockprimes.md file
    blockprimes_source = Path("sample_blockprimes.md")
    if blockprimes_source.exists():
        import shutil
        shutil.copy(blockprimes_source, sample_dir / "blockprimes.md")
    
    # Create additional sample files
    (sample_dir / "introduction.md").write_text("""# Welcome to the Mathematical Wiki

This wiki demonstrates the TiddlyWiki compiler's ability to structure and organize mathematical content.

## Features

- **Automatic tagging** based on content patterns
- **Distance-metric compression** using blockprimes concepts  
- **Structured tiddlers** with summaries and key concepts
- **Interactive filtering** by tags and search

## Sample Content

The blockprimes.md file has been automatically broken down into multiple tiddlers, each with:
- Summary sections
- Key concept extraction
- Automatic tagging (proof, demo, question, etc.)
- Follow-up question detection

**Example**: Mathematical proofs are automatically tagged as "proof" and formatted with special styling.

What patterns will you discover in your own content?
""")
    
    (sample_dir / "algorithms.py").write_text("""# Sample Algorithm
def factorial(n):
    \"\"\"
    Compute factorial using recursive algorithm
    Time complexity: O(n)
    Space complexity: O(n) due to recursion
    \"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    \"\"\"
    Compute Fibonacci number
    Demonstrates exponential growth pattern
    \"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
print(f"5! = {factorial(5)}")
print(f"F(10) = {fibonacci(10)}")
""")
    
    (sample_dir / "concepts.txt").write_text("""Key Mathematical Concepts

Prime Numbers: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29...
Fibonacci Sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34...
Factorials: 1, 2, 6, 24, 120, 720...

Distance Metrics:
- Euclidean: d(x,y) = √(Σ(xi-yi)²)
- Manhattan: d(x,y) = Σ|xi-yi|
- Hamming: number of differing positions

Computational Complexity:
- O(1): constant time
- O(log n): logarithmic 
- O(n): linear time
- O(n log n): linearithmic
- O(n²): quadratic time

What relationships exist between these concepts?
""")
    
    # Create a subdirectory with more content
    (sample_dir / "proofs").mkdir(exist_ok=True)
    (sample_dir / "proofs" / "theorem1.md").write_text("""# Fundamental Theorem of Arithmetic

**Theorem**: Every integer greater than 1 is either prime or can be uniquely factored into prime numbers.

**Proof**: 
1. Existence: Assume some integer n > 1 cannot be factored. 
2. Let S be the set of all such integers.
3. By well-ordering, S has a minimum element m.
4. Since m cannot be factored, m must be prime.
5. This contradicts our assumption.

Therefore, every integer can be factored into primes.

**Demo**: 12 = 2² × 3, 15 = 3 × 5, 30 = 2 × 3 × 5

This theorem is fundamental to number theory and cryptography.

How does this relate to the distance metrics in prime space?
""")
    
    return sample_dir


def run_demo():
    """Run the demonstration"""
    print("TiddlyWiki Compiler Demo")
    print("=" * 40)
    
    # Create sample content
    print("Creating sample content...")
    sample_dir = create_sample_content()
    print(f"Sample content created in: {sample_dir}")
    
    # Initialize compiler
    print("\nInitializing TiddlyWiki compiler...")
    compiler = TiddlyWikiCompiler(str(sample_dir), "demo_wiki.html")
    compiler.compression_enabled = True
    
    # Compile the wiki
    print("Compiling wiki...")
    compiler.compile()
    
    print(f"\nDemo complete!")
    print(f"Generated wiki: demo_wiki.html")
    print(f"Processed {len(compiler.tiddlers)} tiddlers")
    
    # Show tiddler breakdown
    print("\nTiddler Breakdown:")
    for tiddler in compiler.tiddlers:
        print(f"  - {tiddler.title} ({len(tiddler.tags)} tags)")
        print(f"    Tags: {', '.join(tiddler.tags)}")
        if hasattr(tiddler, 'fields') and tiddler.fields.get('complexity_level'):
            print(f"    Complexity: {tiddler.fields['complexity_level']}")
        print()
    
    # Show tag statistics
    all_tags = []
    for tiddler in compiler.tiddlers:
        all_tags.extend(tiddler.tags)
    
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    print("Tag Statistics:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tag}: {count}")
    
    print(f"\nOpen demo_wiki.html in your browser to explore the compiled wiki!")


if __name__ == "__main__":
    run_demo()