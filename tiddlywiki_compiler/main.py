#!/usr/bin/env python3
"""
TiddlyWiki Compiler
Compiles files from directories into a TiddlyWiki file with support for:
- Text files (markdown, txt, etc.)
- Images (png, jpg, gif, etc.) 
- Videos (mp4, webm, etc.)
- Compression using blockprimes-style distance metrics
- Automatic tagging and structuring
"""

import os
import json
import base64
import mimetypes
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
from datetime import datetime


@dataclass
class Tiddler:
    """Represents a TiddlyWiki tiddler"""
    title: str
    text: str
    tags: List[str] = None
    type: str = "text/vnd.tiddlywiki"
    created: str = None
    modified: str = None
    fields: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.fields is None:
            self.fields = {}
        if self.created is None:
            self.created = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        if self.modified is None:
            self.modified = self.created


class TiddlyWikiCompiler:
    """Main compiler class for creating TiddlyWiki files"""
    
    def __init__(self, source_dir: str, output_file: str = "wiki.html"):
        self.source_dir = Path(source_dir)
        self.output_file = output_file
        self.tiddlers: List[Tiddler] = []
        self.compression_enabled = True
        
    def detect_file_patterns(self, content: str) -> List[str]:
        """Detect patterns in content and return appropriate tags"""
        tags = []
        
        # Mathematical patterns
        if re.search(r'\$.*\$|\\\(.*\\\)|\\\[.*\\\]', content):
            tags.append("mathematics")
        
        # Proof patterns
        if re.search(r'\b(proof|theorem|lemma|corollary|proposition)\b', content, re.IGNORECASE):
            tags.append("proof")
            
        # Demo/example patterns
        if re.search(r'\b(demo|example|illustration|show)\b', content, re.IGNORECASE):
            tags.append("demo")
            
        # Question patterns
        if re.search(r'\?|what if|how do|can we|would it', content, re.IGNORECASE):
            tags.append("question")
            
        # Code patterns
        if re.search(r'```|`[^`]+`|def |class |import ', content):
            tags.append("code")
            
        return tags
    
    def compress_content(self, content: str) -> str:
        """Apply blockprimes-style compression using distance metrics"""
        try:
            from .compression import create_compressor
            compressor = create_compressor("hybrid")
            return compressor.compress(content)
        except ImportError:
            # Fallback to simple compression
            patterns = {
                r'\bcomputation\b': 'comp',
                r'\brepresentation\b': 'repr', 
                r'\bmathematical\b': 'math',
                r'\balgorithm\b': 'algo',
                r'\bfunction\b': 'func',
            }
            
            compressed = content
            for pattern, replacement in patterns.items():
                compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
                
            return compressed
    
    def process_text_file(self, file_path: Path) -> List[Tiddler]:
        """Process a text file and create tiddlers"""
        content = file_path.read_text(encoding='utf-8')
        
        # Special handling for blockprimes.md-style content
        if file_path.name.endswith('.md') and 'blockprimes' in file_path.name:
            return self.parse_blockprimes_content(content, file_path.stem)
        
        # Regular text file processing
        tags = self.detect_file_patterns(content)
        tags.append("text")
        tags.append(file_path.suffix[1:])  # file extension as tag
        
        if self.compression_enabled:
            content = self.compress_content(content)
            tags.append("compressed")
        
        tiddler = Tiddler(
            title=file_path.stem,
            text=content,
            tags=tags,
            type="text/x-markdown" if file_path.suffix == ".md" else "text/plain"
        )
        
        return [tiddler]
    
    def parse_blockprimes_content(self, content: str, base_name: str) -> List[Tiddler]:
        """Parse blockprimes.md-style content into multiple structured tiddlers"""
        tiddlers = []
        
        # Split content by sections (assuming sections start with # or ##)
        sections = re.split(r'\n(?=#{1,6}\s)', content)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            lines = section.split('\n')
            title_line = lines[0] if lines else f"{base_name}_section_{i}"
            
            # Extract title from markdown header
            title_match = re.match(r'^#{1,6}\s*(.+)$', title_line)
            title = title_match.group(1) if title_match else title_line
            
            # Create summary from first paragraph
            content_lines = [line for line in lines[1:] if line.strip()]
            paragraph_end = next((i for i, line in enumerate(content_lines) 
                                if line.strip() == ""), len(content_lines))
            summary_lines = content_lines[:paragraph_end]
            summary = ' '.join(summary_lines) if summary_lines else ""
            
            # Enhanced tag detection
            tags = self.detect_file_patterns(section)
            tags.extend([base_name, "blockprimes"])
            
            # Specific blockprimes pattern detection
            if re.search(r'\*\*Proof\*\*|\*\*Theorem\*\*|\*\*Lemma\*\*|\*\*Corollary\*\*', section):
                tags.append("proof")
            
            if re.search(r'\*\*Demo\*\*|\*\*Example\*\*|\*\*Illustration\*\*', section):
                tags.append("demo")
                
            if re.search(r'\*\*Algorithm\*\*|\*\*Procedure\*\*|```', section):
                tags.append("algorithm")
                
            if re.search(r'\*\*Security\*\*|\*\*Implication\*\*|RSA|cryptography', section, re.IGNORECASE):
                tags.append("security")
                
            if re.search(r'distance|metric|d_[EM]|geometric', section):
                tags.append("distance-metric")
                
            if re.search(r'Fermat|prime|factorization', section, re.IGNORECASE):
                tags.append("number-theory")
                
            # Check for follow-up questions and research directions
            question_patterns = [
                r'What.*\?', r'How.*\?', r'Can.*\?', r'Could.*\?', r'Would.*\?',
                r'\*\*Open Question\*\*', r'\*\*Research Direction\*\*', r'\*\*Future Work\*\*'
            ]
            
            for pattern in question_patterns:
                if re.search(pattern, section):
                    tags.append("follow-up")
                    break
            
            # Detect mathematical complexity
            if re.search(r'O\(.*\)|complexity|algorithm', section, re.IGNORECASE):
                tags.append("complexity")
                
            # Structure the tiddler content with enhanced formatting
            structured_content = f"""!! Summary
{summary}

!! Key Concepts
{self.extract_key_concepts(section)}

!! Content
{section}

!! Tags
{' '.join(tags)}"""
            
            # Extract metadata
            metadata = {
                "summary": summary,
                "key_concepts": self.extract_key_concepts(section),
                "question_count": len(re.findall(r'\?', section)),
                "has_proof": "proof" in tags,
                "has_demo": "demo" in tags,
                "complexity_level": self.estimate_complexity(section)
            }
            
            tiddler = Tiddler(
                title=f"{base_name}: {title}",
                text=structured_content,
                tags=tags,
                type="text/x-markdown",
                fields=metadata
            )
            
            tiddlers.append(tiddler)
        
        return tiddlers
    
    def extract_key_concepts(self, text: str) -> str:
        """Extract key mathematical and computational concepts from text"""
        concepts = []
        
        # Mathematical terms
        math_terms = re.findall(r'\b(theorem|lemma|proof|algorithm|distance|metric|prime|factorization|complexity)\b', 
                               text, re.IGNORECASE)
        concepts.extend(list(set(math_terms)))
        
        # Technical symbols and notations
        symbols = re.findall(r'd_[A-Z]|O\([^)]+\)|[≈≡∴∵]', text)
        concepts.extend(symbols)
        
        # Formatted definitions
        definitions = re.findall(r'\*\*([^*]+)\*\*:', text)
        concepts.extend(definitions)
        
        return ', '.join(list(set(concepts))[:10])  # Limit to top 10
    
    def estimate_complexity(self, text: str) -> str:
        """Estimate the complexity level of the content"""
        indicators = {
            "basic": ["example", "demo", "simple"],
            "intermediate": ["theorem", "proof", "algorithm"],
            "advanced": ["research", "open question", "cryptography", "quantum"]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for level, terms in indicators.items():
            scores[level] = sum(1 for term in terms if term in text_lower)
        
        return max(scores, key=scores.get) if any(scores.values()) else "basic"
    
    def process_binary_file(self, file_path: Path) -> Tiddler:
        """Process binary files (images, videos, etc.) as data tiddlers"""
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Encode as base64
        encoded_content = base64.b64encode(content).decode('utf-8')
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"
        
        tags = ["binary", "attachment"]
        if mime_type.startswith("image/"):
            tags.append("image")
        elif mime_type.startswith("video/"):
            tags.append("video")
        elif mime_type.startswith("audio/"):
            tags.append("audio")
        
        tiddler = Tiddler(
            title=file_path.name,
            text=encoded_content,
            tags=tags,
            type=mime_type,
            fields={"_canonical_uri": f"data:{mime_type};base64,"}
        )
        
        return tiddler
    
    def compile_directory(self):
        """Compile all files in the source directory"""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory {self.source_dir} not found")
        
        for file_path in self.source_dir.rglob("*"):
            if file_path.is_file():
                try:
                    # Determine file type and process accordingly
                    mime_type, _ = mimetypes.guess_type(str(file_path))
                    
                    if file_path.suffix in ['.txt', '.md', '.py', '.json', '.yaml', '.yml']:
                        tiddlers = self.process_text_file(file_path)
                        self.tiddlers.extend(tiddlers)
                    else:
                        # Binary file
                        tiddler = self.process_binary_file(file_path)
                        self.tiddlers.append(tiddler)
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    def generate_wiki_html(self) -> str:
        """Generate the final TiddlyWiki HTML file"""
        # Create the tiddlers JSON data
        tiddlers_data = []
        for tiddler in self.tiddlers:
            tiddler_dict = {
                "title": tiddler.title,
                "text": tiddler.text,
                "tags": " ".join(tiddler.tags) if tiddler.tags else "",
                "type": tiddler.type,
                "created": tiddler.created,
                "modified": tiddler.modified
            }
            tiddler_dict.update(tiddler.fields)
            tiddlers_data.append(tiddler_dict)
        
        # Use the advanced template
        try:
            from .wiki_template import generate_tiddlywiki_html
            return generate_tiddlywiki_html(tiddlers_data, f"Compiled Wiki - {self.source_dir.name}")
        except ImportError:
            # Fallback to basic template
            html_template = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Compiled Wiki</title>
<script>
var $tw = {{
    preloadTiddlers: {json.dumps(tiddlers_data, indent=2)}
}};
</script>
</head>
<body>
<div id="tiddlywiki">
<h1>Compiled Wiki</h1>
<p>This wiki contains {len(self.tiddlers)} tiddlers compiled from {self.source_dir}</p>
</div>
</body>
</html>"""
            
            return html_template
    
    def compile(self):
        """Main compilation method"""
        print(f"Compiling directory: {self.source_dir}")
        self.compile_directory()
        
        html_content = self.generate_wiki_html()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Wiki compiled to {self.output_file} with {len(self.tiddlers)} tiddlers")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compile files into TiddlyWiki")
    parser.add_argument("source_dir", help="Source directory to compile")
    parser.add_argument("-o", "--output", default="wiki.html", help="Output wiki file")
    parser.add_argument("--no-compression", action="store_true", help="Disable compression")
    
    args = parser.parse_args()
    
    compiler = TiddlyWikiCompiler(args.source_dir, args.output)
    if args.no_compression:
        compiler.compression_enabled = False
    
    compiler.compile()