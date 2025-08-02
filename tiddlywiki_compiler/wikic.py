#!/usr/bin/env python3
"""
wikic - TiddlyWiki Compiler
Simple zip-like interface for compiling directories into TiddlyWiki files

Usage:
    wikic output.html directory
    wikic wikihtml .
    wikic myproject.html src/
"""

import sys
import argparse
from pathlib import Path

# Import our compiler
from main import TiddlyWikiCompiler


def main():
    """Main CLI interface with zip-like syntax"""
    
    # Simple argument parsing - zip-like interface
    parser = argparse.ArgumentParser(
        description='Compile directories into TiddlyWiki files',
        usage='wikic OUTPUT.html DIRECTORY [options]',
        add_help=False
    )
    
    # Positional arguments (like zip)
    parser.add_argument('output', help='Output wiki file (e.g., wiki.html)')
    parser.add_argument('directory', default='.', nargs='?', help='Directory to compile (default: current directory)')
    
    # Optional flags
    parser.add_argument('-h', '--help', action='help', help='Show this help message')
    parser.add_argument('--no-compression', action='store_true', help='Disable compression')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version='wikic 1.0.0')
    
    # Handle no arguments
    if len(sys.argv) == 1:
        print("wikic - TiddlyWiki Compiler")
        print("")
        print("Usage:")
        print("  wikic output.html directory     # Compile directory into wiki")
        print("  wikic wikihtml .                # Compile current directory")
        print("  wikic project.html src/         # Compile src/ directory")
        print("")
        print("Options:")
        print("  --no-compression               # Disable compression")
        print("  -v, --verbose                  # Verbose output")
        print("  -h, --help                     # Show help")
        print("  --version                      # Show version")
        print("")
        print("Examples:")
        print("  wikic docs.html .              # Current directory -> docs.html")
        print("  wikic research.html papers/    # papers/ -> research.html")
        print("  wikic wiki.html ~/Documents    # Documents -> wiki.html")
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Validate arguments
    output_file = args.output
    source_dir = Path(args.directory)
    
    # Ensure output has .html extension
    if not output_file.endswith('.html'):
        output_file += '.html'
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Directory '{source_dir}' does not exist")
        sys.exit(1)
    
    if not source_dir.is_dir():
        print(f"Error: '{source_dir}' is not a directory")
        sys.exit(1)
    
    # Show what we're doing (like zip)
    if args.verbose:
        print(f"wikic: compiling '{source_dir}' -> '{output_file}'")
    else:
        print(f"  compiling: {source_dir} -> {output_file}")
    
    try:
        # Create and run compiler
        compiler = TiddlyWikiCompiler(str(source_dir), output_file)
        compiler.compression_enabled = not args.no_compression
        
        # Compile with progress indication
        if args.verbose:
            print(f"  scanning: {source_dir}")
        
        compiler.compile()
        
        # Success message (like zip)
        tiddler_count = len(compiler.tiddlers)
        file_size = Path(output_file).stat().st_size
        
        if args.verbose:
            print(f"  created: {output_file} ({file_size:,} bytes, {tiddler_count} tiddlers)")
            
            # Show tag statistics in verbose mode
            all_tags = []
            for tiddler in compiler.tiddlers:
                all_tags.extend(tiddler.tags)
            
            if all_tags:
                tag_counts = {}
                for tag in all_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                print("  top tags:", end="")
                top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for tag, count in top_tags:
                    print(f" {tag}({count})", end="")
                print()
        else:
            print(f"     added: {tiddler_count} tiddlers -> {output_file}")
    
    except Exception as e:
        print(f"wikic: error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()