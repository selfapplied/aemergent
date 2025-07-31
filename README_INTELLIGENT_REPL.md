# Intelligent REPL - Aemergent Framework

🌟 **A sophisticated, context-aware command-line interface that combines traditional REPL functionality with advanced pattern recognition, cellular automata, and adaptive behavior.**

## Features

### 🧠 Core Intelligence
- **Context Awareness**: Tracks directory, typing patterns, command rhythm, and emotional sentiment
- **70/30 Rule Healing**: Automatically fixes commands with low confidence scores
- **Command History**: Smart history tracking with pattern-based suggestions
- **Fatigue Detection**: Monitors typing patterns and triggers breaks when fatigue is detected

### ⚡ Cellular Automata Integration
- **Dynamic Command Rendering**: Uses cellular automata patterns to visually enhance commands
- **Pattern Evolution**: Commands influence CA evolution for visual feedback
- **Adaptive Visualization**: Energy-based character highlighting (⚡ high, ✨ medium)

### 🔤 Unicode Symbol System
- **Symbol Mapping**: Extensive Unicode symbol library (mathematical, geometric, technical)
- **Inline Expansion**: Use `:symbol_name:` syntax (e.g., `:lambda:` → λ, `:arrow_right:` → →)
- **Extensible**: Easy to add new symbol mappings

### 📋 Pattern Banks & Palettes
- **Loadable Palettes**: Switch between command pattern sets (default, mathematical, etc.)
- **Smart Suggestions**: Context-aware command completion
- **Category Organization**: Commands grouped by type (file_ops, development, etc.)

### 🎯 Emotional & Gravity Analysis
- **Sentiment Detection**: Analyzes commands for positive/negative emotional content
- **Gravity Assessment**: Measures command importance and potential risk
- **Visual Indicators**: Emoji-based feedback (😃 positive, 😞 negative, 🔥 high gravity)

### 🔧 String Validation & Healing
- **Lambda Notation**: Support for lambda expressions as validation rules
- **Confidence Scoring**: Multi-factor confidence calculation
- **Typo Correction**: Common command typo fixes
- **History-Based Healing**: Suggests completions from command history

## Usage

### Basic Usage

```python
from aemergent.intelligent_repl import IntelligentREPL

# Create and run interactive REPL
repl = IntelligentREPL()
repl.run()
```

### Programmatic Usage

```python
# Process individual commands
repl = IntelligentREPL()
result = repl.process_input("ls -la")
print(result)  # Enhanced command with CA rendering

# Use Unicode symbols
expanded = repl.expand_unicode("calc :pi: * 2")
print(expanded)  # "calc π * 2"

# Check command confidence
confidence = repl.validator.calculate_confidence("lss")  # Low confidence typo
healed = repl.validator.heal_command("lss", confidence)
print(healed)  # "ls"
```

### Special Commands

Start commands with `!` for special functionality:

- `!help` - Show help information
- `!status` - Display system status and metrics
- `!palette <name>` - Switch command palette
- `!symbols` - List available Unicode symbols
- `!history [n]` - Show command history
- `!debug` - Toggle debug mode
- `!ca` - Toggle cellular automata visualization

### Unicode Symbols

Use the `:symbol_name:` syntax for Unicode symbols:

```
echo :lambda: function :arrow_right: output
math :integral: :sum: :infinity:
symbols :heart: :star: :diamond:
```

Available symbols include:
- **Mathematical**: alpha, beta, gamma, lambda, pi, sigma, integral, infinity
- **Arrows**: arrow_left, arrow_right, arrow_up, arrow_down
- **Geometric**: triangle, square, circle, diamond, star
- **Technical**: null, exists, forall, equivalent, not_equal

## Architecture

### Core Components

1. **IntelligentREPL**: Main orchestrator class
2. **TypingMetrics**: Tracks typing patterns for fatigue detection
3. **StringValidator**: 70/30 rule validation and healing
4. **UnicodeMapper**: Symbol name to codepoint mapping
5. **PatternBank**: Command palette management
6. **CommandContext**: Context state tracking

### Integration with Aemergent Framework

- **Combit**: Used for pattern matching and signal transformation
- **Pascell**: Provides cellular automata for command rendering
- **Operators**: Convolution and transformation operations

## Demo

Run the comprehensive demo:

```bash
# Full demo (requires numpy/jax)
python demos/intelligent_repl_demo.py

# Simplified demo (no dependencies)
python demos/simple_repl_demo.py
```

### Demo Output Example

```
🌟 Simple Intelligent REPL Demo
✨ Features: healing, patterns, Unicode, context awareness

📝 Demo 1: ls -la
   😐📝 Result: ⚡l✨s ✨-⚡l✨a
   Context: file (complexity: 20.0%)

📝 Demo 2: echo hello :lambda: world :arrow_right:
   Unicode expanded: echo hello λ world →
   😐📝 Result: e⚡cho he⚡l⚡lo ✨λ ⚡wor⚡ld →

🔧 Command healed (confidence: 70%): lss typo → ls typo
```

## Advanced Features

### Fatigue Detection

The system monitors typing patterns and triggers timeouts when fatigue is detected:

```python
repl.fatigue_threshold = 0.7  # Trigger at 70% fatigue
repl.timeout_duration = 30    # 30-second timeout
```

### Custom Palettes

Create custom command palettes:

```python
repl.pattern_bank.load_palette("custom", {
    'data_science': ['pandas', 'numpy', 'matplotlib'],
    'cloud': ['aws', 'gcp', 'azure', 'kubectl'],
})
repl.pattern_bank.switch_palette("custom")
```

### Lambda Validators

Use lambda expressions for custom validation:

```python
# Validate only numeric input
repl.validator.validate("123", "lambda x: x.isdigit()")  # True

# Custom validation rules
custom_rule = "lambda cmd: len(cmd) > 3 and cmd.startswith('git')"
repl.validator.validate("git status", custom_rule)  # True
```

## Configuration

### Initialization Options

```python
repl = IntelligentREPL(
    max_history=1000,           # Command history size
)

# Configure behavior
repl.fatigue_threshold = 0.6    # Fatigue detection threshold
repl.timeout_duration = 20      # Timeout duration in seconds
repl.debug_mode = True          # Enable debug output
repl.ca_visualization = True    # Enable CA rendering
```

### Adding Custom Symbols

```python
repl.unicode_mapper.symbol_map.update({
    'custom_symbol': 0x1F680,   # 🚀
    'rocket': 0x1F680,
})
```

## Technical Details

### Context Awareness

The system tracks multiple context dimensions:

- **Directory**: Current working directory
- **Command Type**: file, directory, development, system, general
- **Complexity**: Based on word count, pipes, command length
- **Emotional Valence**: -1 (negative) to +1 (positive)
- **Gravity**: 0 (low) to 1 (high importance/risk)

### 70/30 Rule Implementation

Commands with confidence < 70% trigger healing:

1. **Length Factor**: Shorter commands are more reliable
2. **Common Words**: Known shell commands boost confidence
3. **Structure**: Well-formed syntax increases confidence
4. **History**: Similarity to previous commands helps

### Cellular Automata Rendering

- Commands influence CA evolution patterns
- Character intensity based on Unicode value and CA state
- Visual enhancement: ⚡ (high energy), ✨ (medium), normal (low)

## Dependencies

### Full Version
- numpy
- jax/jaxlib
- scipy

### Simplified Version
- Python 3.8+ standard library only

## Future Enhancements

- **Machine Learning**: Advanced pattern recognition
- **Voice Input**: Speech-to-text integration
- **Collaborative**: Multi-user session support
- **Plugin System**: Extensible architecture
- **Visual Interface**: GUI mode with rich graphics

---

**Built with the Aemergent computational metaphysics framework** 🌟