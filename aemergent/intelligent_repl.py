#!/usr/bin/env python3
"""
Intelligent REPL - Context-Aware Command Interface

A sophisticated REPL system that combines:
- Command history and validation
- Unicode symbol mapping
- Cellular automata command rendering
- Context awareness (directory, typing patterns, rhythm)
- Emotion/fatigue detection
- Pattern banks and palettes
- 70/30 rule with command healing

Uses the aemergent framework's Combit, Pascell, and operator systems.
"""

import os
import sys
import time
import re
import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
import jax.numpy as jnp

from aemergent.combit import Combit, Comfit
from aemergent.pascell import CAEngine, CASetup, CARule, CAGame, play_pascal, play_pascal911
from aemergent.operators import convolve_op, scale_op, clip_op


@dataclass
class TypingMetrics:
    """Tracks typing patterns and rhythm for context awareness"""
    keystroke_times: deque = field(default_factory=lambda: deque(maxlen=100))
    pause_durations: deque = field(default_factory=lambda: deque(maxlen=50))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=20))
    rhythm_pattern: np.ndarray = field(default_factory=lambda: np.zeros(8))

    def update(self, timestamp: float):
        """Update typing metrics with new keystroke"""
        if self.keystroke_times:
            pause = timestamp - self.keystroke_times[-1]
            self.pause_durations.append(pause)

            # Calculate velocity (chars per second)
            if len(self.keystroke_times) >= 2:
                recent_duration = self.keystroke_times[-1] - self.keystroke_times[-10] if len(
                    self.keystroke_times) >= 10 else self.keystroke_times[-1] - self.keystroke_times[0]
                velocity = len(self.keystroke_times) / \
                    max(recent_duration, 0.1)
                self.velocity_history.append(velocity)

        self.keystroke_times.append(timestamp)
        self._update_rhythm()

    def _update_rhythm(self):
        """Extract rhythm pattern from recent typing"""
        if len(self.pause_durations) < 8:
            return

        recent_pauses = list(self.pause_durations)[-8:]
        self.rhythm_pattern = np.array(recent_pauses) / np.mean(recent_pauses)

    @property
    def fatigue_score(self) -> float:
        """Estimate fatigue based on typing patterns"""
        if len(self.velocity_history) < 5:
            return 0.0

        recent_velocity = np.mean(list(self.velocity_history)[-5:])
        baseline_velocity = np.mean(list(self.velocity_history))
        velocity_ratio = recent_velocity / max(baseline_velocity, 0.1)

        # Fatigue indicators: slower typing, irregular rhythm
        rhythm_variance = np.var(self.rhythm_pattern) if len(
            self.rhythm_pattern) > 0 else 0
        fatigue = (1 - velocity_ratio) + rhythm_variance * 0.5

        return np.clip(fatigue, 0, 1)


@dataclass
class CommandContext:
    """Tracks command execution context"""
    directory: str = ""
    command_type: str = ""
    complexity: float = 0.0
    emotional_valence: float = 0.0  # -1 to 1 (negative to positive)
    gravity: float = 0.0  # Importance/urgency
    timestamp: float = 0.0


class UnicodeMapper:
    """Maps symbol names to Unicode codepoints"""

    def __init__(self):
        self.symbol_map = {
            # Mathematical symbols
            'alpha': 0x03B1, 'beta': 0x03B2, 'gamma': 0x03B3, 'delta': 0x03B4,
            'epsilon': 0x03B5, 'zeta': 0x03B6, 'eta': 0x03B7, 'theta': 0x03B8,
            'lambda': 0x03BB, 'mu': 0x03BC, 'pi': 0x03C0, 'sigma': 0x03C3,
            'phi': 0x03C6, 'psi': 0x03C8, 'omega': 0x03C9,

            # Arrows and symbols
            'arrow_left': 0x2190, 'arrow_right': 0x2192, 'arrow_up': 0x2191, 'arrow_down': 0x2193,
            'infinity': 0x221E, 'integral': 0x222B, 'partial': 0x2202, 'nabla': 0x2207,
            'sum': 0x2211, 'product': 0x220F, 'union': 0x222A, 'intersection': 0x2229,

            # Geometric shapes
            'triangle': 0x25B3, 'square': 0x25A1, 'circle': 0x25CB, 'diamond': 0x25C7,
            'star': 0x2605, 'heart': 0x2665, 'spade': 0x2660, 'club': 0x2663,

            # Technical symbols
            'null': 0x2205, 'exists': 0x2203, 'forall': 0x2200, 'therefore': 0x2234,
            'because': 0x2235, 'equivalent': 0x2261, 'approx': 0x2248, 'not_equal': 0x2260,
        }

        # Add reverse mapping for exploration
        self.name_map = {v: k for k, v in self.symbol_map.items()}

    def get_symbol(self, name: str) -> str:
        """Get Unicode symbol by name"""
        codepoint = self.symbol_map.get(name.lower())
        return chr(codepoint) if codepoint else name

    def get_codepoint(self, name: str) -> int:
        """Get Unicode codepoint by name"""
        return self.symbol_map.get(name.lower(), ord(name[0]) if name else 0)

    def list_symbols(self) -> List[Tuple[str, str, int]]:
        """List all available symbols"""
        return [(name, chr(code), code) for name, code in self.symbol_map.items()]


class StringValidator:
    """String mask validator with lambda notation support"""

    def __init__(self):
        self.patterns = {
            'command': r'^[a-zA-Z_][a-zA-Z0-9_]*$',
            'path': r'^[/~]?[\w/.-]*$',
            'number': r'^-?\d+\.?\d*$',
            'symbol': r'^[^\w\s]$',
            'lambda_expr': r'^lambda\s+\w+(\s*,\s*\w+)*\s*:\s*.+$',
        }

        # Confidence scoring for 70/30 rule
        self.confidence_factors = {
            'length': 0.1,      # Shorter commands tend to be more reliable
            'common_words': 0.3,  # Common shell commands boost confidence
            'structure': 0.4,    # Well-structured syntax
            'history': 0.2,      # Similarity to previous commands
        }

    def validate(self, text: str, mask: str) -> bool:
        """Validate text against mask pattern"""
        if mask.startswith('lambda'):
            # Parse lambda expression as validation rule
            try:
                # Safe evaluation for simple lambda expressions
                if re.match(r'^lambda\s+\w+\s*:\s*[\w\s<>=!]+$', mask):
                    validator = eval(mask)
                    return validator(text)
                return False
            except:
                return False

        pattern = self.patterns.get(mask, mask)
        return bool(re.match(pattern, text))

    def calculate_confidence(self, command: str, history: List[str] = None) -> float:
        """Calculate command confidence for 70/30 rule"""
        confidence = 0.0

        # Length factor: shorter commands often more reliable
        length_score = max(0, 1 - len(command) / 100)
        confidence += length_score * self.confidence_factors['length']

        # Common words factor
        common_commands = {'ls', 'cd', 'pwd', 'cat', 'echo',
                           'mkdir', 'rm', 'cp', 'mv', 'grep', 'find'}
        first_word = command.split()[0] if command.split() else ""
        if first_word in common_commands:
            confidence += self.confidence_factors['common_words']

        # Structure factor: well-formed syntax
        structure_score = 0.5  # Base score
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\s+\S+)*$', command):
            structure_score = 0.8
        confidence += structure_score * self.confidence_factors['structure']

        # History factor: similarity to previous commands
        if history:
            max_similarity = 0
            for hist_cmd in history[-10:]:  # Check last 10 commands
                similarity = self._calculate_similarity(command, hist_cmd)
                max_similarity = max(max_similarity, similarity)
            confidence += max_similarity * self.confidence_factors['history']

        return min(confidence, 1.0)

    def _calculate_similarity(self, cmd1: str, cmd2: str) -> float:
        """Calculate similarity between two commands"""
        if not cmd1 or not cmd2:
            return 0.0

        # Simple Levenshtein-like similarity
        max_len = max(len(cmd1), len(cmd2))
        min_len = min(len(cmd1), len(cmd2))

        common_chars = sum(1 for c1, c2 in zip(cmd1, cmd2) if c1 == c2)
        return common_chars / max_len if max_len > 0 else 0.0

    def heal_command(self, command: str, confidence: float = 0.7, history: List[str] = None) -> str:
        """Apply 70/30 rule healing to command"""
        if confidence >= 0.7:
            return command

        healed = command

        # Fix common typos
        typo_fixes = {
            'sis': 'sys', 'mkdri': 'mkdir', 'lss': 'ls', 'cdd': 'cd',
            'ecoh': 'echo', 'grpe': 'grep', 'finde': 'find', 'catt': 'cat'
        }

        for typo, fix in typo_fixes.items():
            healed = healed.replace(typo, fix)

        # Add spaces before numbers if missing
        healed = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', healed)

        # Remove double spaces
        healed = re.sub(r'\s+', ' ', healed).strip()

        # Try to match against command history for partial matches
        if history and confidence < 0.5:
            words = healed.split()
            if words:
                first_word = words[0]
                for hist_cmd in reversed(history[-20:]):
                    if hist_cmd.startswith(first_word) and len(hist_cmd.split()) > len(words):
                        # Suggest completion from history
                        return hist_cmd

        return healed


class PatternBank:
    """Manages loadable pattern palettes"""

    def __init__(self):
        self.palettes = {}
        self.current_palette = "default"
        self._load_default_palette()

    def _load_default_palette(self):
        """Load default command patterns"""
        self.palettes["default"] = {
            'file_ops': ['ls', 'cat', 'touch', 'rm', 'cp', 'mv', 'chmod', 'chown'],
            'directory_ops': ['cd', 'mkdir', 'rmdir', 'pwd', 'pushd', 'popd'],
            'system_ops': ['ps', 'top', 'kill', 'which', 'whereis', 'jobs', 'nohup'],
            'network_ops': ['ping', 'curl', 'wget', 'ssh', 'scp', 'rsync'],
            'development': ['git', 'python', 'pip', 'npm', 'make', 'gcc', 'vim'],
            'text_processing': ['grep', 'sed', 'awk', 'sort', 'uniq', 'wc', 'head', 'tail'],
        }

        # Mathematical palette
        self.palettes["mathematical"] = {
            'functions': ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs'],
            'constants': ['pi', 'e', 'phi', 'inf', 'nan'],
            'operations': ['sum', 'prod', 'diff', 'integral', 'derivative'],
        }

    def load_palette(self, name: str, patterns: Dict[str, List[str]]):
        """Load a new pattern palette"""
        self.palettes[name] = patterns

    def switch_palette(self, name: str) -> bool:
        """Switch to a different palette"""
        if name in self.palettes:
            self.current_palette = name
            return True
        return False

    def get_pattern_suggestions(self, partial: str) -> List[str]:
        """Get pattern suggestions for partial command"""
        suggestions = []
        current = self.palettes.get(self.current_palette, {})

        for category, commands in current.items():
            suggestions.extend(
                [cmd for cmd in commands if cmd.startswith(partial)])

        return sorted(suggestions)

    def list_palettes(self) -> List[str]:
        """List available palettes"""
        return list(self.palettes.keys())


class IntelligentREPL:
    """Main intelligent REPL system"""

    def __init__(self, max_history: int = 1000):
        self.history = deque(maxlen=max_history)
        self.typing_metrics = TypingMetrics()
        self.unicode_mapper = UnicodeMapper()
        self.validator = StringValidator()
        self.pattern_bank = PatternBank()

        # Cellular automata for command rendering
        self.ca_engine = CAEngine(CASetup.by_complexity(0.3))
        self.command_combit = Combit(
            ['input', 'process', 'output', 'context'], np.ones(4))

        # Context tracking
        self.current_context = CommandContext()
        self.emotion_map = defaultdict(float)

        # Fatigue monitoring
        self.fatigue_threshold = 0.7
        self.timeout_duration = 30  # seconds
        self.is_timeout_active = False

        self.running = True

        # Enhanced features
        self.debug_mode = False
        self.ca_visualization = True

    def update_context(self, command: str):
        """Update command context awareness"""
        self.current_context.directory = os.getcwd()
        self.current_context.timestamp = time.time()

        # Analyze command complexity
        words = command.split()
        pipes = len(re.findall(r'[|&;<>]', command))
        complexity = len(words) * 0.1 + pipes * 0.2 + len(command) * 0.01
        self.current_context.complexity = min(complexity, 1.0)

        # Enhanced emotion/gravity analysis
        negative_words = ['error', 'fail', 'crash',
                          'bug', 'problem', 'issue', 'wrong', 'broken']
        positive_words = ['success', 'complete', 'done',
                          'good', 'great', 'perfect', 'works', 'fixed']
        urgent_words = ['urgent', 'critical', 'important',
                        'emergency', 'asap', 'now', 'immediately']
        destructive_words = ['rm', 'delete',
                             'kill', 'stop', 'destroy', 'remove']

        valence = 0.0
        gravity = 0.0

        command_lower = command.lower()

        for word in negative_words:
            if word in command_lower:
                valence -= 0.2
                gravity += 0.1

        for word in positive_words:
            if word in command_lower:
                valence += 0.2

        for word in urgent_words:
            if word in command_lower:
                gravity += 0.3

        for word in destructive_words:
            if word in command_lower:
                gravity += 0.4  # Destructive commands are high gravity

        self.current_context.emotional_valence = np.clip(valence, -1, 1)
        self.current_context.gravity = min(gravity, 1.0)

        # Determine command type
        if words:
            first_word = words[0].lower()
            if first_word in ['cd', 'mkdir', 'rmdir', 'pwd']:
                self.current_context.command_type = "directory"
            elif first_word in ['ls', 'cat', 'touch', 'rm', 'cp', 'mv']:
                self.current_context.command_type = "file"
            elif first_word in ['git', 'python', 'pip', 'npm']:
                self.current_context.command_type = "development"
            else:
                self.current_context.command_type = "general"

    def render_command_with_ca(self, command: str) -> str:
        """Use cellular automata to render/enhance command"""
        try:
            # Run CA evolution
            ca_result = self.ca_engine.generate()

            # Use CA pattern to influence command rendering
            if hasattr(ca_result, 'grid') and len(ca_result.grid) > 1:
                # Get the last row of the CA evolution
                last_row = ca_result.grid[-1]
                pattern_length = min(len(command), len(last_row))

                if self.ca_visualization and pattern_length > 0:
                    # Apply pattern as visual enhancement
                    enhanced = ""
                    for i, char in enumerate(command):
                        if i < pattern_length:
                            intensity = int(
                                last_row[i]) if last_row[i] > 0 else 0
                            # Use command character influence for intensity
                            char_influence = (ord(char) % 10) / 10.0
                            combined_intensity = intensity * char_influence

                            if combined_intensity > 0.7:
                                enhanced += f"⚡{char}"  # High energy
                            elif combined_intensity > 0.3:
                                enhanced += f"✨{char}"  # Medium energy
                            else:
                                enhanced += char
                        else:
                            enhanced += char
                    return enhanced

        except Exception as e:
            if self.debug_mode:
                print(f"CA rendering error: {e}")

        return command

    def check_fatigue_and_timeout(self):
        """Check fatigue level and trigger timeout if needed"""
        fatigue = self.typing_metrics.fatigue_score

        if fatigue > self.fatigue_threshold and not self.is_timeout_active:
            print(
                f"\n🌿 Fatigue detected (level: {fatigue:.2f}). Taking a brief pause...")
            print(
                f"⏰ Timeout for {self.timeout_duration} seconds. Time to stretch!")

            self.is_timeout_active = True

            def timeout_thread():
                time.sleep(self.timeout_duration)
                self.is_timeout_active = False
                print("\n✨ Timeout complete. Welcome back!")

            threading.Thread(target=timeout_thread, daemon=True).start()

    def process_input(self, user_input: str) -> str:
        """Process user input with full intelligence pipeline"""
        timestamp = time.time()
        self.typing_metrics.update(timestamp)

        # Check for fatigue
        self.check_fatigue_and_timeout()

        if self.is_timeout_active:
            return "⏸️  Please wait, timeout is active..."

        # Handle special commands
        if user_input.startswith('!'):
            return self.handle_special_command(user_input[1:])

        # Update context
        self.update_context(user_input)

        # Calculate confidence and validate
        history_commands = [entry['command'] for entry in self.history]
        confidence = self.validator.calculate_confidence(
            user_input, history_commands)

        # Apply 70/30 rule healing if confidence is low
        if confidence < 0.7:
            healed = self.validator.heal_command(
                user_input, confidence, history_commands)
            if healed != user_input:
                print(
                    f"🔧 Command healed (confidence: {confidence:.1%}): {user_input} → {healed}")
                user_input = healed
                confidence = self.validator.calculate_confidence(
                    user_input, history_commands)

        # Process with Combit for pattern matching
        try:
            # Update combit state based on command characteristics
            input_energy = len(user_input) / 50.0  # Normalize command length
            context_energy = self.current_context.complexity

            self.command_combit['input'] = input_energy
            self.command_combit['context'] = context_energy
        except Exception as e:
            if self.debug_mode:
                print(f"Combit processing error: {e}")

        # Render with cellular automata
        rendered_command = self.render_command_with_ca(user_input)

        # Add to history
        self.history.append({
            'command': user_input,
            'rendered': rendered_command,
            'context': self.current_context,
            'timestamp': timestamp,
            'fatigue': self.typing_metrics.fatigue_score,
            'confidence': confidence
        })

        return rendered_command

    def handle_special_command(self, command: str) -> str:
        """Handle special system commands starting with !"""
        parts = command.split()
        if not parts:
            return "❓ Empty special command"

        cmd = parts[0].lower()

        if cmd == 'help':
            return self.show_help()
        elif cmd == 'status':
            return self.show_status()
        elif cmd == 'palette':
            if len(parts) > 1:
                palette_name = parts[1]
                if self.pattern_bank.switch_palette(palette_name):
                    return f"✅ Switched to palette: {palette_name}"
                else:
                    return f"❌ Palette '{palette_name}' not found. Available: {', '.join(self.pattern_bank.list_palettes())}"
            else:
                return f"📋 Current palette: {self.pattern_bank.current_palette}. Available: {', '.join(self.pattern_bank.list_palettes())}"
        elif cmd == 'symbols':
            symbols = self.unicode_mapper.list_symbols()
            return f"🔣 Available symbols ({len(symbols)}): " + ", ".join([f"{name}→{symbol}" for name, symbol, _ in symbols[:10]]) + "..."
        elif cmd == 'history':
            count = int(parts[1]) if len(
                parts) > 1 and parts[1].isdigit() else 5
            recent = list(self.history)[-count:]
            return "\n".join([f"{i+1}. {entry['command']} (confidence: {entry.get('confidence', 0):.1%})" for i, entry in enumerate(recent)])
        elif cmd == 'debug':
            self.debug_mode = not self.debug_mode
            return f"🔧 Debug mode: {'ON' if self.debug_mode else 'OFF'}"
        elif cmd == 'ca':
            self.ca_visualization = not self.ca_visualization
            return f"🔮 CA visualization: {'ON' if self.ca_visualization else 'OFF'}"
        else:
            return f"❓ Unknown special command: {cmd}. Try !help"

    def show_help(self) -> str:
        """Show help information"""
        return """
🌟 Intelligent REPL Help:

Special Commands (start with !):
  !help          - Show this help
  !status        - Show system status
  !palette <name> - Switch command palette
  !symbols       - List Unicode symbols
  !history [n]   - Show command history
  !debug         - Toggle debug mode
  !ca            - Toggle CA visualization

Unicode Symbols:
  Use :symbol_name: in commands (e.g., :lambda: :arrow_right:)

Features:
  ⚡ Cellular automata command rendering
  🧠 Context awareness and pattern matching
  😴 Fatigue detection with auto-timeout
  🔧 70/30 rule command healing
  📋 Loadable pattern palettes
        """.strip()

    def show_status(self) -> str:
        """Show current system status"""
        fatigue = self.typing_metrics.fatigue_score
        context = self.current_context

        return f"""
📊 System Status:
  Directory: {context.directory}
  Fatigue: {fatigue:.1%} {'😴' if fatigue > 0.5 else '😊'}
  Commands: {len(self.history)}
  Palette: {self.pattern_bank.current_palette}
  CA Viz: {'ON' if self.ca_visualization else 'OFF'}
  Debug: {'ON' if self.debug_mode else 'OFF'}
  
🎯 Last Context:
  Type: {context.command_type}
  Complexity: {context.complexity:.1%}
  Emotion: {context.emotional_valence:+.1f}
  Gravity: {context.gravity:.1%}
        """.strip()

    def get_suggestions(self, partial: str) -> List[str]:
        """Get intelligent command suggestions"""
        suggestions = []

        # Pattern bank suggestions
        suggestions.extend(self.pattern_bank.get_pattern_suggestions(partial))

        # History-based suggestions
        for entry in reversed(list(self.history)):
            if entry['command'].startswith(partial) and entry['command'] not in suggestions:
                suggestions.append(entry['command'])

        return suggestions[:10]  # Limit to 10 suggestions

    def expand_unicode(self, text: str) -> str:
        """Expand Unicode symbol names in text"""
        def replace_symbol(match):
            symbol_name = match.group(1)
            return self.unicode_mapper.get_symbol(symbol_name)

        # Replace :symbol_name: with actual Unicode
        return re.sub(r':(\w+):', replace_symbol, text)

    def run(self):
        """Main REPL loop"""
        print("🌟 Intelligent REPL v2.0 - Aemergent Framework")
        print(
            "✨ Features: CA rendering, fatigue detection, pattern matching, Unicode symbols")
        print(
            "💡 Try commands with :symbol_name: for Unicode (e.g., :lambda: :arrow_right:)")
        print("🔄 Type 'exit' or Ctrl+C to quit, '!help' for special commands\n")

        while self.running:
            try:
                # Show context info
                fatigue = self.typing_metrics.fatigue_score
                fatigue_indicator = "😴" if fatigue > 0.5 else "😊" if fatigue < 0.2 else "🙂"

                # Show current directory and palette
                current_dir = os.path.basename(os.getcwd()) or "/"
                palette_indicator = f"[{self.pattern_bank.current_palette[0].upper()}]" if self.pattern_bank.current_palette != "default" else ""

                prompt = f"{fatigue_indicator}{palette_indicator} {current_dir} >>> "

                user_input = input(prompt).strip()

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye! Stay curious! ✨")
                    self.running = False
                    break

                if not user_input:
                    continue

                # Expand Unicode symbols
                expanded_input = self.expand_unicode(user_input)

                # Process through intelligence pipeline
                result = self.process_input(expanded_input)

                # Display result with context
                context = self.current_context
                emotion_indicator = "😃" if context.emotional_valence > 0.2 else "😞" if context.emotional_valence < -0.2 else "😐"
                gravity_indicator = "🔥" if context.gravity > 0.5 else "⚡" if context.gravity > 0.2 else "📝"

                print(f"{emotion_indicator}{gravity_indicator} {result}")

                # Show fatigue warning if high
                if fatigue > 0.6:
                    print(
                        f"⚠️  Fatigue level: {fatigue:.1%} - Consider taking a break!")

            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                self.running = False
            except EOFError:
                print("\n👋 End of input. Goodbye!")
                self.running = False
            except Exception as e:
                if self.debug_mode:
                    print(f"❌ Debug Error: {e}")
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"❌ Error: {e}")


if __name__ == "__main__":
    repl = IntelligentREPL()
    repl.run()
