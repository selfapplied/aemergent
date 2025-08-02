"""
Real TiddlyWiki Template Generator
Uses the actual TiddlyWiki npm package to generate proper wiki files
"""

import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any


def generate_real_tiddlywiki(tiddlers_data: List[Dict[str, Any]], output_file: str, title: str = "Compiled Wiki") -> str:
    """Generate a real TiddlyWiki HTML file using the TiddlyWiki npm package"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        wiki_folder = temp_path / "temp_wiki"
        
        # Initialize empty TiddlyWiki
        result = subprocess.run([
            "npx", "tiddlywiki", str(wiki_folder), "--init", "empty"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to initialize TiddlyWiki: {result.stderr}")
        
        # Create tiddlers directory
        tiddlers_dir = wiki_folder / "tiddlers"
        tiddlers_dir.mkdir(exist_ok=True)
        
        # Add our compiled tiddlers as .tid files
        for tiddler in tiddlers_data:
            tiddler_file = tiddlers_dir / f"{sanitize_filename(tiddler['title'])}.tid"
            
            # Convert to TiddlyWiki .tid format
            tid_content = []
            
            # Add metadata fields
            tid_content.append(f"title: {tiddler['title']}")
            tid_content.append(f"tags: {tiddler.get('tags', '')}")
            tid_content.append(f"type: {tiddler.get('type', 'text/vnd.tiddlywiki')}")
            tid_content.append(f"created: {tiddler.get('created', '')}")
            tid_content.append(f"modified: {tiddler.get('modified', '')}")
            
            # Add custom fields
            for key, value in tiddler.items():
                if key not in ['title', 'text', 'tags', 'type', 'created', 'modified']:
                    tid_content.append(f"{key}: {value}")
            
            # Add empty line and content
            tid_content.append("")
            tid_content.append(tiddler.get('text', ''))
            
            # Write tiddler file
            tiddler_file.write_text('\n'.join(tid_content), encoding='utf-8')
        
        # Update tiddlywiki.info with our title and configuration
        info_file = wiki_folder / "tiddlywiki.info"
        with open(info_file, 'r') as f:
            config = json.load(f)
        
        # Add some useful plugins for mathematical content
        config["plugins"] = [
            "tiddlywiki/katex",  # For mathematical formulas
            "tiddlywiki/highlight",  # For code syntax highlighting
            "tiddlywiki/codemirror",  # Better code editing
        ]
        
        # Set description
        config["description"] = title
        
        with open(info_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Add a custom site title tiddler
        site_title_tiddler = tiddlers_dir / "$__SiteTitle.tid"
        site_title_tiddler.write_text(f"""title: $:/SiteTitle
type: text/vnd.tiddlywiki

{title}""")
        
        # Add a custom site subtitle
        site_subtitle_tiddler = tiddlers_dir / "$__SiteSubtitle.tid"
        site_subtitle_tiddler.write_text(f"""title: $:/SiteSubtitle
type: text/vnd.tiddlywiki

Compiled with wikic - {len(tiddlers_data)} tiddlers""")
        
        # Build the wiki to HTML
        build_result = subprocess.run([
            "npx", "tiddlywiki", str(wiki_folder), "--build", "index"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if build_result.returncode != 0:
            raise RuntimeError(f"Failed to build TiddlyWiki: {build_result.stderr}")
        
        # Copy the generated HTML to the target location
        generated_html = wiki_folder / "output" / "index.html"
        if not generated_html.exists():
            raise RuntimeError("TiddlyWiki build did not generate index.html")
        
        shutil.copy2(generated_html, output_file)
        
        return str(Path(output_file).resolve())


def sanitize_filename(title: str) -> str:
    """Sanitize tiddler title for use as filename"""
    # Replace problematic characters
    sanitized = title.replace('/', '_').replace('\\', '_').replace(':', '_')
    sanitized = sanitized.replace('*', '_').replace('?', '_').replace('"', '_')
    sanitized = sanitized.replace('<', '_').replace('>', '_').replace('|', '_')
    
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    
    return sanitized