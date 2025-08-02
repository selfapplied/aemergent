#!/usr/bin/env python3
"""
Build binary version of wikic
"""

import subprocess
import sys
import shutil
from pathlib import Path

def build_binary():
    """Build binary version using PyInstaller"""
    
    print("Building wikic binary...")
    
    # Check if PyInstaller is available
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Build the binary
    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", "wikic",
        "--console",
        "wikic.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Move binary to current directory
        binary_path = Path("dist/wikic")
        if binary_path.exists():
            shutil.copy2(binary_path, "wikic")
            print("✅ Binary created: ./wikic")
            print("")
            print("Usage:")
            print("  ./wikic output.html directory")
            print("  ./wikic wikihtml .")
            print("")
        else:
            print("❌ Binary not found in dist/")
    else:
        print(f"❌ PyInstaller failed: {result.stderr}")

def create_shell_script():
    """Create a simple shell script wrapper as alternative"""
    
    script_content = f"""#!/bin/bash
# wikic - TiddlyWiki Compiler
# Simple wrapper script

# Get the directory where this script is located
DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )"

# Run the Python script
python3 "$DIR/wikic.py" "$@"
"""
    
    with open("wikic", "w") as f:
        f.write(script_content)
    
    # Make it executable
    subprocess.run(["chmod", "+x", "wikic"])
    
    print("✅ Shell script created: ./wikic")
    print("")
    print("Usage:")
    print("  ./wikic output.html directory")
    print("  ./wikic wikihtml .")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build wikic binary")
    parser.add_argument("--shell", action="store_true", help="Create shell script instead of binary")
    
    args = parser.parse_args()
    
    if args.shell:
        create_shell_script()
    else:
        try:
            build_binary()
        except Exception as e:
            print(f"Binary build failed: {e}")
            print("Creating shell script instead...")
            create_shell_script()