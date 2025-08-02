# TiddlyWiki Compiler

A powerful tool for compiling files from directories into structured TiddlyWiki files with support for various media types, advanced compression, and intelligent content analysis.

## Features

### 🔍 **Smart Content Analysis**
- Automatic pattern detection for mathematical content, proofs, algorithms, and more
- Intelligent tagging based on content structure and patterns
- Follow-up question detection and research direction identification
- Complexity level estimation (basic, intermediate, advanced)

### 📚 **Blockprimes.md Structure Breaking**
- Automatic breakdown of large markdown files into structured tiddlers
- Summary extraction and key concept identification
- Special handling for mathematical content with distance-metric awareness
- Proof, demo, and example pattern recognition

### 🗜️ **Advanced Compression**
- Distance-metric based compression using blockprimes-style computational representations
- Integration with aemergent codec system for optimal compression
- Mathematical pattern recognition for efficient encoding
- Hybrid compression strategies

### 📁 **Multi-Format Support**
- Text files (markdown, txt, python, json, yaml)
- Images (png, jpg, gif, etc.) with base64 encoding
- Videos and audio files
- Binary file support with MIME type detection

### 🏷️ **Intelligent Tagging System**
- Automatic tag generation based on content patterns
- Special tags for mathematical concepts, proofs, algorithms
- Security and cryptography pattern detection
- Follow-up questions and research directions

## Installation

```bash
# Clone or copy the tiddlywiki_compiler directory
cd tiddlywiki_compiler

# Optional: Install aemergent for advanced compression
pip install -e ../aemergent  # if aemergent is available
```

## Quick Start

### wikic Command (Recommended)

```bash
# Simple zip-like syntax
./wikic output.html directory

# Examples
./wikic docs.html .                    # Current directory -> docs.html
./wikic research.html papers/          # papers/ -> research.html  
./wikic wikihtml .                     # Current directory -> wikihtml.html

# With options
./wikic project.html src/ -v           # Verbose output
./wikic archive.html data/ --no-compression  # Disable compression
```

### Python API

```python
from main import TiddlyWikiCompiler

# Compile a directory into a TiddlyWiki
compiler = TiddlyWikiCompiler("source_directory", "output_wiki.html")
compiler.compile()
```

### Legacy Command Line

```bash
python3 main.py source_directory -o output_wiki.html
python3 wikic.py output.html directory  # Same as ./wikic
```

### Demo

```bash
python3 demo.py
```

This creates sample content and generates `demo_wiki.html` to showcase the system's capabilities.

## Project Structure

```
tiddlywiki_compiler/
├── main.py              # Main compiler class and CLI
├── compression.py       # Advanced compression system
├── wiki_template.py     # TiddlyWiki HTML template generator
├── demo.py             # Demonstration script
├── sample_blockprimes.md # Sample mathematical content
└── README.md           # This file
```

## Real TiddlyWiki Integration

The compiler now uses the **actual TiddlyWiki** from npm to generate fully functional wiki files instead of simplified templates.

### Features of Real TiddlyWiki Output:
- **Complete TiddlyWiki functionality** - all standard features work
- **KaTeX plugin** - Mathematical formula rendering
- **CodeMirror plugin** - Advanced code editing 
- **Highlight.js plugin** - Syntax highlighting for code
- **Full search capabilities** - Built-in TiddlyWiki search
- **Tag system** - Complete tagging and filtering
- **Responsive design** - Works on all devices
- **No server required** - Standalone HTML files

### File Size Comparison:
- **Basic template**: ~25KB (limited functionality)
- **Real TiddlyWiki**: ~3-4MB (full functionality)
- The size increase includes the complete TiddlyWiki core with all features

## Advanced Features

### Distance-Metric Compression

The compiler includes a novel compression system inspired by blockprimes computational distance metrics:

```python
from compression import DistanceMetricCompressor

compressor = DistanceMetricCompressor()
compressed = compressor.encode_via_distances(text)
```

This system:
- Recognizes mathematical sequences (primes, fibonacci, factorials)
- Detects computational complexity patterns
- Encodes content based on proximity to mathematical concepts
- Provides superior compression for mathematical/technical content

### Blockprimes Content Parsing

For `blockprimes.md` style content, the compiler automatically:

1. **Splits by sections** (# and ## headers)
2. **Extracts summaries** from first paragraphs
3. **Identifies key concepts** (theorems, algorithms, proofs)
4. **Detects patterns**:
   - **Proof**: `**Proof**`, `**Theorem**`, `**Lemma**`
   - **Demo**: `**Demo**`, `**Example**`, `**Illustration**`
   - **Algorithm**: `**Algorithm**`, code blocks
   - **Security**: RSA, cryptography patterns
   - **Questions**: `What...?`, `How...?`, `**Research Direction**`

### Custom Tagging

The system recognizes and tags:

- **Mathematics**: formulas, equations, mathematical symbols
- **Proofs**: formal proof structures
- **Demos**: examples and illustrations  
- **Code**: algorithms and programming content
- **Questions**: follow-up questions and research directions
- **Security**: cryptographic concepts
- **Complexity**: computational complexity analysis

## Example Output

Given a `blockprimes.md` file, the compiler generates multiple tiddlers:

```
blockprimes: Introduction
├── Tags: mathematics, blockprimes, number-theory, basic
├── Summary: "Block primes represent a novel approach..."
├── Key Concepts: prime, factorization, coordinates, space
└── Content: Full section with enhanced formatting

blockprimes: Distance Metrics in Prime Space  
├── Tags: proof, demo, distance-metric, mathematics, intermediate
├── Summary: "For numbers m = ∏pᵢ^aᵢ and n = ∏pᵢ^bᵢ..."
├── Key Concepts: Euclidean, Manhattan, distance, metric
└── Content: Mathematical formulas and examples
```

## TiddlyWiki Features

The generated HTML includes:
- **Interactive sidebar** with search and tag filtering
- **Responsive design** optimized for mathematical content
- **Tag-based navigation** with color-coded categories
- **Full-text search** across all tiddlers
- **Mathematical formatting** support
- **Statistics dashboard** showing content breakdown

## Configuration

### Compression Options

```python
compiler = TiddlyWikiCompiler("source", "output.html")

# Disable compression
compiler.compression_enabled = False

# Use specific compression type
from compression import create_compressor
compressor = create_compressor("distance")  # or "aemergent", "hybrid"
```

### Custom Pattern Detection

```python
# Add custom patterns to detect_file_patterns method
def detect_custom_patterns(self, content):
    tags = []
    if re.search(r'quantum|qubits', content, re.IGNORECASE):
        tags.append("quantum")
    return tags
```

## Integration with Aemergent

When aemergent is available, the compiler uses:
- `CodecSystem` for advanced mathematical compression
- `Combit` for bit-level optimization
- `Pascell` for pattern-based encoding

This provides superior compression ratios for mathematical and computational content.

## Use Cases

1. **Research Documentation**: Convert mathematical research papers into interactive wikis
2. **Educational Content**: Structure textbook material with automatic concept extraction
3. **Code Documentation**: Combine code, documentation, and examples into searchable wikis
4. **Mathematical Archives**: Organize and compress large collections of mathematical content

## Browser Compatibility

The generated TiddlyWiki files work in all modern browsers:
- Chrome/Chromium
- Firefox
- Safari
- Edge

No server required - everything runs client-side.

## License

This project builds upon TiddlyWiki concepts and integrates with the aemergent codec system. See individual license files for specific components.

## Contributing

Contributions welcome! Areas of interest:
- Additional compression algorithms
- Enhanced mathematical pattern recognition
- New content type support
- UI/UX improvements

## Examples

See `demo.py` for a complete working example that generates sample content and demonstrates all features of the system.