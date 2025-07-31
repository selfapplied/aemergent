#!/usr/bin/env python3
"""
Simple Intelligent REPL Demo

A lightweight demonstration of the intelligent REPL concepts:
- Command history and validation
- Unicode symbol mapping
- Basic pattern matching
- Fatigue simulation
- 70/30 rule healing
- Context awareness

This version works without numpy/jax dependencies.
"""

import os
import sys
import time
import re
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class SimpleContext:
    """Simplified command context"""
    directory: str = ""
    command_type: str = ""
    complexity: float = 0.0
    emotional_valence: float = 0.0
    gravity: float = 0.0
    timestamp: float = 0.0


class SimpleUnicodeMapper:
    """Maps symbol names to Unicode codepoints"""
    
    def __init__(self):
        self.symbol_map = {
            'lambda': 0x03BB, 'pi': 0x03C0, 'sigma': 0x03C3, 'alpha': 0x03B1,
            'arrow_right': 0x2192, 'arrow_left': 0x2190, 'infinity': 0x221E,
            'heart': 0x2665, 'star': 0x2605, 'lightning': 0x26A1,
        }
    
    def get_symbol(self, name: str) -> str:
        """Get Unicode symbol by name"""
        codepoint = self.symbol_map.get(name.lower())
        return chr(codepoint) if codepoint else name
    
    def expand_unicode(self, text: str) -> str:
        """Expand Unicode symbol names in text"""
        def replace_symbol(match):
            symbol_name = match.group(1)
            return self.get_symbol(symbol_name)
        
        return re.sub(r':(\w+):', replace_symbol, text)


class SimpleValidator:
    """String validator with 70/30 rule healing"""
    
    def __init__(self):
        self.typo_fixes = {
            'sis': 'sys', 'mkdri': 'mkdir', 'lss': 'ls', 'cdd': 'cd',
            'ecoh': 'echo', 'grpe': 'grep', 'finde': 'find', 'catt': 'cat'
        }
    
    def calculate_confidence(self, command: str, history: List[str] = None) -> float:
        """Calculate command confidence for 70/30 rule"""
        confidence = 0.5  # Base confidence
        
        # Common commands boost confidence
        common_commands = {'ls', 'cd', 'pwd', 'cat', 'echo', 'mkdir', 'rm', 'cp', 'mv'}
        first_word = command.split()[0] if command.split() else ""
        if first_word in common_commands:
            confidence += 0.3
        
        # Well-structured syntax
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\s+\S+)*$', command):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def heal_command(self, command: str, confidence: float = 0.7) -> str:
        """Apply 70/30 rule healing to command"""
        if confidence >= 0.7:
            return command
        
        healed = command
        
        # Fix common typos
        for typo, fix in self.typo_fixes.items():
            healed = healed.replace(typo, fix)
        
        # Add spaces before numbers
        healed = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', healed)
        
        # Remove double spaces
        healed = re.sub(r'\s+', ' ', healed).strip()
        
        return healed


class SimplePatternBank:
    """Manages command pattern palettes"""
    
    def __init__(self):
        self.palettes = {
            "default": {
                'file_ops': ['ls', 'cat', 'touch', 'rm', 'cp', 'mv'],
                'directory_ops': ['cd', 'mkdir', 'rmdir', 'pwd'],
                'system_ops': ['ps', 'top', 'kill', 'which'],
                'development': ['git', 'python', 'pip', 'npm'],
            },
            "mathematical": {
                'functions': ['sin', 'cos', 'tan', 'log', 'sqrt'],
                'constants': ['pi', 'e', 'phi'],
                'operations': ['sum', 'diff', 'integral'],
            }
        }
        self.current_palette = "default"
    
    def switch_palette(self, name: str) -> bool:
        """Switch to a different palette"""
        if name in self.palettes:
            self.current_palette = name
            return True
        return False
    
    def get_suggestions(self, partial: str) -> List[str]:
        """Get pattern suggestions for partial command"""
        suggestions = []
        current = self.palettes.get(self.current_palette, {})
        
        for category, commands in current.items():
            suggestions.extend([cmd for cmd in commands if cmd.startswith(partial)])
        
        return sorted(suggestions)


class SimpleIntelligentREPL:
    """Simplified intelligent REPL system"""
    
    def __init__(self):
        self.history = deque(maxlen=100)
        self.unicode_mapper = SimpleUnicodeMapper()
        self.validator = SimpleValidator()
        self.pattern_bank = SimplePatternBank()
        self.current_context = SimpleContext()
        
        # Simulated metrics
        self.command_count = 0
        self.fatigue_score = 0.0
        
        self.running = True
    
    def update_context(self, command: str):
        """Update command context awareness"""
        self.current_context.directory = os.getcwd()
        self.current_context.timestamp = time.time()
        
        # Analyze command
        words = command.split()
        pipes = len(re.findall(r'[|&;<>]', command))
        self.current_context.complexity = min((len(words) * 0.1 + pipes * 0.2), 1.0)
        
        # Emotion analysis
        negative_words = ['error', 'fail', 'crash', 'bug', 'problem']
        positive_words = ['success', 'complete', 'done', 'good', 'great']
        destructive_words = ['rm', 'delete', 'kill', 'destroy']
        
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
        
        for word in destructive_words:
            if word in command_lower:
                gravity += 0.4
        
        self.current_context.emotional_valence = max(-1, min(1, valence))
        self.current_context.gravity = min(gravity, 1.0)
        
        # Command type
        if words:
            first_word = words[0].lower()
            if first_word in ['cd', 'mkdir', 'rmdir', 'pwd']:
                self.current_context.command_type = "directory"
            elif first_word in ['ls', 'cat', 'touch', 'rm']:
                self.current_context.command_type = "file"
            elif first_word in ['git', 'python', 'pip']:
                self.current_context.command_type = "development"
            else:
                self.current_context.command_type = "general"
    
    def simulate_ca_rendering(self, command: str) -> str:
        """Simulate cellular automata rendering"""
        # Simple pattern: add decorations based on command characteristics
        enhanced = ""
        for i, char in enumerate(command):
            char_code = ord(char) % 10
            if char_code > 7:
                enhanced += f"⚡{char}"  # High energy
            elif char_code > 4:
                enhanced += f"✨{char}"  # Medium energy
            else:
                enhanced += char
        return enhanced
    
    def process_input(self, user_input: str) -> str:
        """Process user input through intelligence pipeline"""
        self.command_count += 1
        
        # Simulate fatigue (increases with command count)
        self.fatigue_score = min(self.command_count * 0.05, 1.0)
        
        # Handle special commands
        if user_input.startswith('!'):
            return self.handle_special_command(user_input[1:])
        
        # Update context
        self.update_context(user_input)
        
        # Calculate confidence and heal if needed
        history_commands = [entry['command'] for entry in self.history]
        confidence = self.validator.calculate_confidence(user_input, history_commands)
        
        if confidence < 0.7:
            healed = self.validator.heal_command(user_input, confidence)
            if healed != user_input:
                print(f"🔧 Command healed (confidence: {confidence:.1%}): {user_input} → {healed}")
                user_input = healed
        
        # Simulate CA rendering
        rendered = self.simulate_ca_rendering(user_input)
        
        # Add to history
        self.history.append({
            'command': user_input,
            'rendered': rendered,
            'context': self.current_context,
            'timestamp': time.time(),
            'confidence': confidence
        })
        
        return rendered
    
    def handle_special_command(self, command: str) -> str:
        """Handle special system commands"""
        parts = command.split()
        if not parts:
            return "❓ Empty special command"
        
        cmd = parts[0].lower()
        
        if cmd == 'help':
            return """
🌟 Simple Intelligent REPL Help:

Special Commands:
  !help          - Show this help
  !status        - Show system status
  !palette <name> - Switch command palette
  !history [n]   - Show command history
  !symbols       - Demo Unicode symbols

Features:
  🔧 70/30 rule command healing
  🎨 Pattern palettes
  🔤 Unicode symbol expansion (:symbol:)
  🧠 Context awareness
            """.strip()
        
        elif cmd == 'status':
            context = self.current_context
            return f"""
📊 System Status:
  Directory: {os.path.basename(context.directory or os.getcwd())}
  Commands: {len(self.history)}
  Fatigue: {self.fatigue_score:.1%}
  Palette: {self.pattern_bank.current_palette}
  
🎯 Last Context:
  Type: {context.command_type}
  Complexity: {context.complexity:.1%}
  Emotion: {context.emotional_valence:+.1f}
  Gravity: {context.gravity:.1%}
            """.strip()
        
        elif cmd == 'palette':
            if len(parts) > 1:
                palette_name = parts[1]
                if self.pattern_bank.switch_palette(palette_name):
                    return f"✅ Switched to palette: {palette_name}"
                else:
                    return f"❌ Palette '{palette_name}' not found. Available: {list(self.pattern_bank.palettes.keys())}"
            else:
                return f"📋 Current palette: {self.pattern_bank.current_palette}. Available: {list(self.pattern_bank.palettes.keys())}"
        
        elif cmd == 'history':
            count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            recent = list(self.history)[-count:]
            return "\n".join([f"{i+1}. {entry['command']} (confidence: {entry.get('confidence', 0):.1%})" 
                            for i, entry in enumerate(recent)])
        
        elif cmd == 'symbols':
            return "🔣 Demo symbols: :lambda: :pi: :arrow_right: :heart: :star: :lightning:"
        
        else:
            return f"❓ Unknown special command: {cmd}. Try !help"
    
    def run_demo(self):
        """Run interactive demo"""
        print("🌟 Simple Intelligent REPL Demo")
        print("✨ Features: healing, patterns, Unicode, context awareness")
        print("💡 Try commands with :symbol: (e.g., :lambda: :arrow_right:)")
        print("🔄 Type 'exit' to quit, '!help' for commands\n")
        
        # Demo commands
        demo_commands = [
            "ls -la",
            "echo hello :lambda: world :arrow_right:",
            "lss typo",  # Will be healed
            "git status",
            "rm dangerous_file",  # High gravity
            "success message",     # Positive emotion
            "error handling",      # Negative emotion
            "!palette mathematical",
            "!status",
            "!history 3"
        ]
        
        print("🎬 Running demo sequence...\n")
        
        for i, cmd in enumerate(demo_commands):
            print(f"📝 Demo {i+1}: {cmd}")
            
            # Expand Unicode
            expanded = self.unicode_mapper.expand_unicode(cmd)
            if expanded != cmd:
                print(f"   Unicode expanded: {expanded}")
            
            # Process command
            result = self.process_input(expanded)
            context = self.current_context
            
            # Show results
            emotion_indicator = "😃" if context.emotional_valence > 0.2 else "😞" if context.emotional_valence < -0.2 else "😐"
            gravity_indicator = "🔥" if context.gravity > 0.5 else "⚡" if context.gravity > 0.2 else "📝"
            
            print(f"   {emotion_indicator}{gravity_indicator} Result: {result}")
            
            if context.command_type != "general":
                print(f"   Context: {context.command_type} (complexity: {context.complexity:.1%})")
            
            # Show fatigue warning
            if self.fatigue_score > 0.5:
                print(f"   ⚠️ Fatigue level: {self.fatigue_score:.1%}")
            
            print()
            time.sleep(0.5)  # Pause between demo commands
        
        print("✨ Demo complete! All features demonstrated successfully.")
        print("\n💡 The full version with numpy/jax would include:")
        print("   - Real cellular automata command rendering")
        print("   - Advanced pattern matching with Combit transformations")
        print("   - Sophisticated rhythm analysis for fatigue detection")


def main():
    """Main demo function"""
    try:
        repl = SimpleIntelligentREPL()
        repl.run_demo()
        
        print("\n🎯 Interactive mode available - try:")
        print("   repl = SimpleIntelligentREPL()")
        print("   result = repl.process_input('your command')")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted")
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()