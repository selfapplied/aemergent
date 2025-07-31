#!/usr/bin/env python3
"""
Intelligent REPL Demo

Demonstrates the capabilities of the intelligent REPL system including:
- Command history and context awareness
- Cellular automata command rendering
- Fatigue detection and timeout
- Unicode symbol mapping
- Pattern banks and palettes
- 70/30 rule command healing
- Special commands
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aemergent.intelligent_repl import IntelligentREPL


def demo_basic_features():
    """Demo basic REPL features"""
    print("🎬 === Intelligent REPL Demo ===\n")
    
    repl = IntelligentREPL()
    
    print("🔧 Testing basic command processing...")
    
    # Test command processing
    test_commands = [
        "ls -la",
        "cd /tmp",
        "mkdir test_directory", 
        "echo hello :lambda: world :arrow_right:",
        "git status",
        "python -c 'print(\"Hello CA world!\")'",
        "rm dangerous_file",  # High gravity command
        "success message",     # Positive emotion
        "error handling bug",  # Negative emotion
    ]
    
    for cmd in test_commands:
        print(f"\n📝 Processing: {cmd}")
        result = repl.process_input(cmd)
        context = repl.current_context
        
        print(f"   Rendered: {result}")
        print(f"   Context: {context.command_type}, complexity: {context.complexity:.2f}")
        print(f"   Emotion: {context.emotional_valence:+.2f}, gravity: {context.gravity:.2f}")
        
        # Brief pause to simulate typing
        time.sleep(0.1)
    
    return repl


def demo_special_commands(repl):
    """Demo special commands"""
    print("\n\n🌟 === Special Commands Demo ===\n")
    
    special_commands = [
        "!help",
        "!status", 
        "!symbols",
        "!palette mathematical",
        "!palette",
        "!history 3",
        "!debug",
        "!ca",
    ]
    
    for cmd in special_commands:
        print(f"\n🔮 Special command: {cmd}")
        result = repl.process_input(cmd)
        print(f"Result:\n{result}")
        time.sleep(0.5)


def demo_command_healing(repl):
    """Demo command healing with 70/30 rule"""
    print("\n\n🔧 === Command Healing Demo ===\n")
    
    # Test commands with typos and low confidence
    typo_commands = [
        "lss -la",           # Simple typo
        "mkdri newdir",      # Another typo
        "ecoh hello",        # Echo typo
        "gitstatusi",        # Complex typo
        "cdd /home",         # cd typo
        "rm-rf",             # Missing space
    ]
    
    for cmd in typo_commands:
        print(f"\n🩹 Testing healing for: '{cmd}'")
        
        # Calculate confidence manually for demo
        history_commands = [entry['command'] for entry in repl.history]
        confidence = repl.validator.calculate_confidence(cmd, history_commands)
        
        print(f"   Original confidence: {confidence:.1%}")
        
        healed = repl.validator.heal_command(cmd, confidence, history_commands)
        print(f"   Healed command: '{healed}'")
        
        # Process the healed command
        result = repl.process_input(cmd)
        print(f"   Final result: {result}")


def demo_unicode_symbols(repl):
    """Demo Unicode symbol mapping"""
    print("\n\n🔣 === Unicode Symbols Demo ===\n")
    
    unicode_commands = [
        "echo :lambda: function",
        "calc :pi: * 2",
        "show :alpha: :beta: :gamma:",
        "path :arrow_right: :destination:",
        "math :integral: :sum: :infinity:",
        "symbols :heart: :star: :diamond:",
    ]
    
    print("Available symbols:")
    symbols = repl.unicode_mapper.list_symbols()
    for name, symbol, code in symbols[:15]:  # Show first 15
        print(f"  :{name}: → {symbol} (U+{code:04X})")
    
    print("\nTesting Unicode expansion:")
    for cmd in unicode_commands:
        print(f"\n🔤 Command: {cmd}")
        expanded = repl.expand_unicode(cmd)
        print(f"   Expanded: {expanded}")
        result = repl.process_input(cmd)
        print(f"   Processed: {result}")


def demo_pattern_banks(repl):
    """Demo pattern banks and palettes"""
    print("\n\n📋 === Pattern Banks Demo ===\n")
    
    # Show available palettes
    print("Available palettes:")
    for palette in repl.pattern_bank.list_palettes():
        print(f"  - {palette}")
    
    # Test suggestions with different palettes
    test_partials = ["g", "c", "p", "s"]
    
    for palette in ["default", "mathematical"]:
        print(f"\n🎨 Testing palette: {palette}")
        repl.pattern_bank.switch_palette(palette)
        
        for partial in test_partials:
            suggestions = repl.pattern_bank.get_pattern_suggestions(partial)
            if suggestions:
                print(f"   '{partial}*' → {suggestions[:5]}")  # Show first 5


def demo_fatigue_detection():
    """Demo fatigue detection and timeout"""
    print("\n\n😴 === Fatigue Detection Demo ===\n")
    
    # Create a fresh REPL for fatigue testing
    repl = IntelligentREPL()
    repl.fatigue_threshold = 0.3  # Lower threshold for demo
    repl.timeout_duration = 5     # Shorter timeout for demo
    
    print("Simulating rapid typing to trigger fatigue detection...")
    
    # Simulate rapid typing with short intervals
    rapid_commands = [
        "ls", "pwd", "cd", "ls", "cat", "echo", "grep", "find",
        "ps", "top", "kill", "git", "npm", "pip", "vim"
    ]
    
    for i, cmd in enumerate(rapid_commands):
        # Simulate fast typing by updating metrics with short intervals
        timestamp = time.time() - (len(rapid_commands) - i) * 0.05  # Very fast typing
        repl.typing_metrics.update(timestamp)
        
        fatigue = repl.typing_metrics.fatigue_score
        print(f"Command {i+1}: '{cmd}' (fatigue: {fatigue:.2f})")
        
        result = repl.process_input(cmd)
        
        if repl.is_timeout_active:
            print("   ⏰ Timeout triggered! Waiting...")
            time.sleep(repl.timeout_duration + 1)  # Wait for timeout to complete
            break
        
        time.sleep(0.1)


def demo_cellular_automata():
    """Demo cellular automata integration"""
    print("\n\n🔮 === Cellular Automata Demo ===\n")
    
    repl = IntelligentREPL()
    repl.ca_visualization = True
    repl.debug_mode = True
    
    print("Testing CA-enhanced command rendering...")
    
    ca_test_commands = [
        "simple",
        "complex command with pipes | and stuff",
        "mathematical calculation",
        "network operation",
        "file system access",
    ]
    
    for cmd in ca_test_commands:
        print(f"\n🧮 Command: '{cmd}'")
        
        # Show CA evolution for this command
        try:
            ca_result = repl.ca_engine.generate()
            if hasattr(ca_result, 'grid'):
                print(f"   CA pattern (last row): {ca_result.grid[-1][:10]}")
        except Exception as e:
            print(f"   CA error: {e}")
        
        # Process and show rendered result
        result = repl.render_command_with_ca(cmd)
        print(f"   Rendered: {result}")


def main():
    """Main demo function"""
    print("🚀 Starting Intelligent REPL comprehensive demo...\n")
    
    try:
        # Run all demos
        repl = demo_basic_features()
        demo_special_commands(repl)
        demo_command_healing(repl)
        demo_unicode_symbols(repl)
        demo_pattern_banks(repl)
        demo_fatigue_detection()
        demo_cellular_automata()
        
        print("\n\n✨ === Demo Complete ===")
        print("🎯 All features demonstrated successfully!")
        print("\n💡 To run the interactive REPL:")
        print("   python -m aemergent.intelligent_repl")
        print("\n   Or import and use:")
        print("   from aemergent.intelligent_repl import IntelligentREPL")
        print("   repl = IntelligentREPL()")
        print("   repl.run()")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()