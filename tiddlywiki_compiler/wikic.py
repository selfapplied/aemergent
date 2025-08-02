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
    parser.add_argument('-c', '--no-compression', action='store_true', help='Disable compression')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-o', '--optimize', action='store_true', help='Generate optimal wiki using distance-metric theory')
    parser.add_argument('-d', '--disassemble', action='store_true', help='Disassemble existing wiki instead of creating one')
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
        print("  -c, --no-compression           # Disable compression")
        print("  -o, --optimize                 # Generate optimal wiki using distance-metric theory")
        print("  -d, --disassemble              # Disassemble existing wiki file")
        print("  -v, --verbose                  # Verbose output")
        print("  -h, --help                     # Show help")
        print("  --version                      # Show version")
        print("")
        print("Examples:")
        print("  wikic docs.html .              # Current directory -> docs.html")
        print("  wikic research.html papers/    # papers/ -> research.html")
        print("  wikic optimal.html data/ -o    # Optimized wiki")
        print("  wikic extracted/ wiki.html -d  # Extract from wiki")
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.disassemble:
        # Disassembly mode: wikic output_dir input.html --disassemble
        wiki_file = args.directory  # Second argument is wiki file in disassemble mode
        output_dir = args.output    # First argument is output directory
        
        if not Path(wiki_file).exists():
            print(f"Error: Wiki file '{wiki_file}' does not exist")
            sys.exit(1)
        
        if args.verbose:
            print(f"wikic: disassembling '{wiki_file}' -> '{output_dir}'")
        else:
            print(f"  disassembling: {wiki_file} -> {output_dir}")
        
        try:
            from wiki_disassembler import TiddlyWikiDisassembler
            
            disassembler = TiddlyWikiDisassembler(wiki_file)
            results = disassembler.disassemble(output_dir, "markdown")
            
            tiddler_count = len(results['tiddlers'])
            cluster_count = len(results['analysis']['clusters'])
            
            if args.verbose:
                print(f"  extracted: {tiddler_count} tiddlers in {cluster_count} clusters")
                print("  analysis report:")
                print(results['report'])
            else:
                print(f"  extracted: {tiddler_count} tiddlers -> {output_dir}")
            
        except ImportError:
            print("Error: numpy required for disassembly. Install with: uv add numpy")
            sys.exit(1)
        except Exception as e:
            print(f"wikic: disassembly error: {e}")
            sys.exit(1)
    
    else:
        # Compilation mode (standard or optimized)
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
        
        # Show what we're doing
        mode = "optimizing" if args.optimize else "compiling"
        if args.verbose:
            print(f"wikic: {mode} '{source_dir}' -> '{output_file}'")
        else:
            print(f"  {mode}: {source_dir} -> {output_file}")
        
        try:
            if args.optimize:
                # Use optimal wiki generator
                try:
                    from optimal_wiki import OptimalWikiGenerator
                    
                    compiler = OptimalWikiGenerator(str(source_dir), output_file)
                    compiler.compression_enabled = not args.no_compression
                    
                    compiler.compile()
                    
                    # Show optimization results
                    tiddler_count = len(compiler.tiddlers)
                    cluster_count = len(compiler.optimal_clusters)
                    
                    if args.verbose:
                        file_size = Path(output_file).stat().st_size
                        print(f"  optimized: {output_file} ({file_size:,} bytes)")
                        print(f"  clusters: {cluster_count} optimal groups")
                        print(f"  navigation: {len(compiler.navigation_graph)} links")
                    else:
                        print(f"  optimized: {tiddler_count} tiddlers in {cluster_count} clusters -> {output_file}")
                
                except ImportError:
                    print("Error: numpy required for optimization. Install with: uv add numpy")
                    sys.exit(1)
            
            else:
                # Standard compilation
                compiler = TiddlyWikiCompiler(str(source_dir), output_file)
                compiler.compression_enabled = not args.no_compression
                
                if args.verbose:
                    print(f"  scanning: {source_dir}")
                
                compiler.compile()
                
                # Success message
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