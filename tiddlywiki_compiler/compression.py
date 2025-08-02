"""
Advanced compression system for TiddlyWiki compiler
Integrates with aemergent codec system for distance-metric based compression
"""

import sys
import os
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any

# Add aemergent to path if available
if Path("../aemergent").exists():
    sys.path.append(str(Path("../aemergent").resolve()))

try:
    from aemergent.codec_system import CodecSystem
    from aemergent.combit import Combit
    from aemergent.pascell import Pascell
    AEMERGENT_AVAILABLE = True
except ImportError:
    AEMERGENT_AVAILABLE = False


class DistanceMetricCompressor:
    """Distance-metric based compressor using blockprimes-style computation representation"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.distance_cache = {}
        
        # Mathematical pattern templates
        self.math_patterns = {
            "prime_sequence": r"2,\s*3,\s*5,\s*7,\s*11",
            "fibonacci": r"1,\s*1,\s*2,\s*3,\s*5,\s*8",
            "factorial": r"1,\s*2,\s*6,\s*24,\s*120",
            "powers_of_2": r"1,\s*2,\s*4,\s*8,\s*16",
            "powers_of_3": r"1,\s*3,\s*9,\s*27,\s*81"
        }
        
        # Computational distance representations
        self.distance_encodings = {
            "linear_growth": "L",
            "exponential_growth": "E", 
            "logarithmic_growth": "G",
            "polynomial_growth": "P",
            "prime_distribution": "R",
            "recursive_pattern": "X"
        }
        
    def compute_pattern_distance(self, text: str) -> Dict[str, float]:
        """Compute distances to various mathematical/computational patterns"""
        distances = {}
        
        # Check for mathematical sequences
        for pattern_name, pattern_regex in self.math_patterns.items():
            matches = re.findall(pattern_regex, text)
            if matches:
                # Distance inversely proportional to number of matches
                distances[pattern_name] = 1.0 / (len(matches) + 1)
            else:
                distances[pattern_name] = 1.0
        
        # Check for computational complexity patterns
        if re.search(r'O\([n\^2]+\)', text):
            distances["quadratic_complexity"] = 0.2
        elif re.search(r'O\(n\s*log\s*n\)', text):
            distances["linearithmic_complexity"] = 0.3
        elif re.search(r'O\(n\)', text):
            distances["linear_complexity"] = 0.1
            
        # Check for proof structures
        if re.search(r'assume|suppose|given|therefore|thus|hence', text, re.IGNORECASE):
            distances["proof_structure"] = 0.1
            
        return distances
    
    def encode_via_distances(self, text: str) -> str:
        """Encode text using distance metrics to computational patterns"""
        distances = self.compute_pattern_distance(text)
        
        # Create compact representation
        encoding_parts = []
        
        # Sort by distance (closest patterns first)
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        
        for pattern, distance in sorted_distances[:3]:  # Top 3 closest patterns
            # Convert distance to compact representation
            distance_code = int(distance * 100)  # 0-100 scale
            encoding_parts.append(f"{pattern[:2]}{distance_code:02d}")
        
        # Add length indicator
        length_code = min(len(text) // 10, 99)
        encoding_parts.append(f"L{length_code:02d}")
        
        compressed_header = "|".join(encoding_parts)
        
        # Apply basic text compression
        compressed_text = self.compress_text_content(text)
        
        return f"[DIST:{compressed_header}]{compressed_text}"
    
    def compress_text_content(self, text: str) -> str:
        """Apply content-aware text compression"""
        # Common mathematical/technical term abbreviations
        abbreviations = {
            "computation": "comp",
            "algorithm": "algo", 
            "function": "func",
            "variable": "var",
            "parameter": "param",
            "theorem": "thm",
            "proposition": "prop",
            "corollary": "cor",
            "definition": "def",
            "example": "ex",
            "demonstration": "demo",
            "therefore": "∴",
            "because": "∵",
            "approximately": "≈",
            "equivalent": "≡"
        }
        
        compressed = text
        for full_term, abbrev in abbreviations.items():
            compressed = re.sub(r'\b' + full_term + r'\b', abbrev, compressed, flags=re.IGNORECASE)
        
        # Compress repeated whitespace
        compressed = re.sub(r'\s+', ' ', compressed)
        
        return compressed


class AemergentCompressor:
    """Compression using the aemergent codec system"""
    
    def __init__(self):
        self.available = AEMERGENT_AVAILABLE
        if self.available:
            self.codec_system = CodecSystem()
            
    def compress(self, text: str) -> str:
        """Compress text using aemergent codecs"""
        if not self.available:
            return text
            
        try:
            # Convert text to numeric representation for codec processing
            numeric_data = [ord(c) for c in text]
            
            # Use Combit for bit-level compression
            combit = Combit()
            compressed_bits = combit.compress_sequence(numeric_data)
            
            # Encode result
            return f"[AEMERGENT:{len(compressed_bits)}]{compressed_bits}"
            
        except Exception as e:
            print(f"Aemergent compression failed: {e}")
            return text


class HybridCompressor:
    """Hybrid compression combining distance metrics and aemergent codecs"""
    
    def __init__(self):
        self.distance_compressor = DistanceMetricCompressor()
        self.aemergent_compressor = AemergentCompressor()
        
    def compress(self, text: str, prefer_distance_metrics: bool = True) -> str:
        """Apply hybrid compression strategy"""
        
        # Try distance metric compression first
        distance_result = self.distance_compressor.encode_via_distances(text)
        
        # Try aemergent compression
        aemergent_result = self.aemergent_compressor.compress(text)
        
        # Choose best compression
        if prefer_distance_metrics:
            return distance_result
        else:
            # Choose based on compression ratio
            if len(aemergent_result) < len(distance_result):
                return aemergent_result
            else:
                return distance_result
    
    def decompress(self, compressed_text: str) -> str:
        """Decompress text (basic implementation)"""
        if compressed_text.startswith("[DIST:"):
            # Extract header and content
            header_end = compressed_text.find("]")
            header = compressed_text[6:header_end]
            content = compressed_text[header_end+1:]
            
            # Basic decompression (expand abbreviations)
            decompressed = content.replace("comp", "computation")
            decompressed = decompressed.replace("algo", "algorithm")
            decompressed = decompressed.replace("func", "function")
            # ... (add reverse mappings for other abbreviations)
            
            return decompressed
            
        elif compressed_text.startswith("[AEMERGENT:"):
            # Aemergent decompression would be implemented here
            return compressed_text
            
        else:
            return compressed_text


def create_compressor(compression_type: str = "hybrid") -> Any:
    """Factory function to create appropriate compressor"""
    if compression_type == "distance":
        return DistanceMetricCompressor()
    elif compression_type == "aemergent":
        return AemergentCompressor()
    else:
        return HybridCompressor()