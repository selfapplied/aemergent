#!/usr/bin/env python3
"""
Quaternion Pascal CA Demo
Detects 4D symmetry patterns in cellular automata
"""

import numpy as np
from aemergent.pascell import CAEngine, CASetup, play_pascal911 as pascal911, play_pascal



def simple_rule(left, right, mod_base=2):
    """Simple Pascal rule: just add adjacent elements"""
    return (left + right) % mod_base

def test_evolution():
    """Actually run the engine to see if patterns evolve"""
    print("🎯 testing ca evolution with overflow prevention")
    print("=" * 50)

    # Create engine with overflow prevention rule
    setup = CASetup.by_rows(20)
    engine = CAEngine(setup)
    engine.active_players.clear()
    engine.add_player('overflow', pascal911)

    # Generate grid
    game = engine.generate()
    print(game)



    # Check for overflow prevention (9+1=1 cases)
    print("\n🔍 checking for overflow prevention:")
    print("-" * 50)
    overflow_found = False
    for i in range(len(game.grid)-1):
        current = game.grid[i]
        next_row = game.grid[i+1]
        for j in range(len(current)-1):
            left = current[j]
            right = current[j+1]
            if left == 9 and right == 1:
                result = next_row[j+1] if j+1 < len(next_row) else 0
                print(f"✅ Overflow prevented at Row {i+2}, Col {j+1}: 9+1={result}")
                overflow_found = True

    if not overflow_found:
        print("❌ No overflow cases found")

    # Check if any rows are identical (fixed points)
    print("\n🔍 checking for fixed points:")
    print("-" * 50)
    fixed_found = False
    for i in range(len(game.grid)-1):
        current = game.grid[i]
        next_row = game.grid[i+1]
        if np.array_equal(current, next_row):
            pattern = ''.join(str(x) for x in current if x != 0) or '0'
            print(f"✅ Fixed point at Row {i+2}: {pattern}")
            fixed_found = True

    if not fixed_found:
        print("❌ No fixed points found")

    print(f"\n✅ done!")

if __name__ == "__main__":
    test_evolution() 
