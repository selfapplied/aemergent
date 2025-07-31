#!/usr/bin/env python3
"""
TextSignal - Text as Signal Processing with Masked Updates

Revolutionary file manipulation system that treats text as signals:
- Text files as 1D signal arrays (character-level or token-level)
- Attention masks for selective region updates  
- Template rendering for piecewise file modifications
- Quaternion-dimensional operations for multi-file processing
- Signal processing techniques applied to code/text

Architecture:
- TextSignal: Core text-as-signal processor
- AttentionMask: Geometric masks for selective updates
- TemplateRenderer: Advanced template-based file updates
- FileSignalProcessor: Multi-file signal operations

Example:
    >>> from aemergent.textsignal import TextSignal, AttentionMask
    >>> # Load file as signal
    >>> signal = TextSignal.from_file("code.py")
    >>> # Create mask for function definitions
    >>> mask = AttentionMask.from_pattern(r"def \w+\(")
    >>> # Apply template update to masked regions
    >>> updated = signal.apply_template(template, mask)
    >>> # Write back to file
    >>> updated.to_file("code.py")
"""

import numpy as np
import jax.numpy as jnp
import re
from typing import Union, List, Dict, Optional, Callable, Tuple, Pattern
from pathlib import Path
from dataclasses import dataclass
from .combit import render_template_ca, template_op, create_template_operator
from .gravtree import GravTree
from .combinion import CombitArray, QDim


@dataclass
class AttentionMask:
    """
    Geometric attention mask for selective text signal processing
    """
    mask: jnp.ndarray
    regions: List[Tuple[int, int]]  # (start, end) pairs
    pattern: Optional[str] = None
    intensity: float = 1.0
    
    @classmethod
    def from_pattern(cls, pattern: Union[str, Pattern], text: str, intensity: float = 1.0) -> 'AttentionMask':
        """Create mask from regex pattern"""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        
        matches = list(pattern.finditer(text))
        regions = [(m.start(), m.end()) for m in matches]
        
        # Create binary mask
        mask = jnp.zeros(len(text), dtype=bool)
        for start, end in regions:
            mask = mask.at[start:end].set(True)
        
        return cls(mask=mask, regions=regions, pattern=pattern.pattern, intensity=intensity)
    
    @classmethod
    def from_lines(cls, line_numbers: List[int], text: str, intensity: float = 1.0) -> 'AttentionMask':
        """Create mask from line numbers"""
        lines = text.split('\n')
        regions = []
        mask = jnp.zeros(len(text), dtype=bool)
        
        current_pos = 0
        for i, line in enumerate(lines):
            if i in line_numbers:
                start = current_pos
                end = current_pos + len(line)
                regions.append((start, end))
                mask = mask.at[start:end].set(True)
            current_pos += len(line) + 1  # +1 for newline
        
        return cls(mask=mask, regions=regions, intensity=intensity)
    
    @classmethod
    def from_functions(cls, text: str, intensity: float = 1.0) -> 'AttentionMask':
        """Create mask for function definitions"""
        return cls.from_pattern(r'def\s+\w+\([^)]*\):[^}]*?(?=\n\S|\nclass|\ndef|\Z)', text, intensity)
    
    @classmethod
    def from_classes(cls, text: str, intensity: float = 1.0) -> 'AttentionMask':
        """Create mask for class definitions"""
        return cls.from_pattern(r'class\s+\w+[^:]*:[^}]*?(?=\n\S|\nclass|\ndef|\Z)', text, intensity)
    
    @classmethod
    def geometric_focus(cls, text: str, center: float = 0.5, width: float = 0.3, intensity: float = 1.0) -> 'AttentionMask':
        """Create geometric attention mask with Gaussian focus"""
        length = len(text)
        center_pos = int(center * length)
        width_chars = int(width * length)
        
        positions = jnp.arange(length)
        gaussian = jnp.exp(-((positions - center_pos) ** 2) / (2 * width_chars ** 2))
        mask = gaussian > 0.1  # Threshold for binary mask
        
        # Find continuous regions
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i))
                in_region = False
        
        if in_region:
            regions.append((start, len(mask)))
        
        return cls(mask=mask, regions=regions, intensity=intensity)
    
    def __and__(self, other: 'AttentionMask') -> 'AttentionMask':
        """Logical AND of masks"""
        combined_mask = self.mask & other.mask
        # Recompute regions from combined mask
        regions = self._mask_to_regions(combined_mask)
        return AttentionMask(mask=combined_mask, regions=regions, intensity=min(self.intensity, other.intensity))
    
    def __or__(self, other: 'AttentionMask') -> 'AttentionMask':
        """Logical OR of masks"""
        combined_mask = self.mask | other.mask
        regions = self._mask_to_regions(combined_mask)
        return AttentionMask(mask=combined_mask, regions=regions, intensity=max(self.intensity, other.intensity))
    
    def _mask_to_regions(self, mask: jnp.ndarray) -> List[Tuple[int, int]]:
        """Convert binary mask to regions"""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i))
                in_region = False
        
        if in_region:
            regions.append((start, len(mask)))
        
        return regions


class TextSignal:
    """
    Text as Signal Processor
    
    Treats text files as 1D signals for advanced signal processing operations
    including filtering, convolution, template rendering, and masked updates.
    """
    
    def __init__(self, text: str, encoding: str = 'character'):
        self.text = text
        self.encoding = encoding  # 'character', 'token', 'word', 'line'
        self.signal = self._text_to_signal(text, encoding)
        self.gravtree = GravTree()  # For geometric text processing
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path], encoding: str = 'character') -> 'TextSignal':
        """Load text file as signal"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return cls(text, encoding)
    
    def _text_to_signal(self, text: str, encoding: str) -> jnp.ndarray:
        """Convert text to signal array"""
        if encoding == 'character':
            return jnp.array([ord(c) for c in text], dtype=jnp.float32)
        elif encoding == 'word':
            words = text.split()
            return jnp.array([hash(word) % 1000 for word in words], dtype=jnp.float32)
        elif encoding == 'line':
            lines = text.split('\n')
            return jnp.array([len(line) for line in lines], dtype=jnp.float32)
        elif encoding == 'token':
            # Simple tokenization - could be enhanced with real tokenizer
            tokens = re.findall(r'\w+|[^\w\s]', text)
            return jnp.array([hash(token) % 1000 for token in tokens], dtype=jnp.float32)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
    
    def _signal_to_text(self, signal: jnp.ndarray, encoding: str) -> str:
        """Convert signal array back to text"""
        if encoding == 'character':
            # Clamp to valid character range
            chars = jnp.clip(signal, 32, 126).astype(int)
            return ''.join(chr(int(c)) for c in chars)
        else:
            # For other encodings, we need to store original tokens
            # This is a simplified version
            return self.text  # Return original for now
    
    def apply_mask(self, mask: AttentionMask) -> 'TextSignal':
        """Apply attention mask to signal"""
        masked_signal = jnp.where(mask.mask[:len(self.signal)], 
                                  self.signal * mask.intensity, 
                                  self.signal * 0.1)  # Reduce non-masked regions
        
        new_text = self._signal_to_text(masked_signal, self.encoding)
        return TextSignal(new_text, self.encoding)
    
    def apply_template(self, template: str, mask: AttentionMask, 
                      values: Optional[List[str]] = None) -> 'TextSignal':
        """Apply template to masked regions"""
        if values is None:
            values = [template] * len(mask.regions)
        
        result_text = self.text
        
        # Apply template to each masked region
        for i, (start, end) in enumerate(mask.regions):
            if i < len(values):
                # Replace the masked region with template value
                result_text = (result_text[:start] + 
                              values[i] + 
                              result_text[end:])
        
        return TextSignal(result_text, self.encoding)
    
    def convolve(self, kernel: jnp.ndarray) -> 'TextSignal':
        """Apply convolution to text signal"""
        convolved_signal = jnp.convolve(self.signal, kernel, mode='same')
        new_text = self._signal_to_text(convolved_signal, self.encoding)
        return TextSignal(new_text, self.encoding)
    
    def geometric_convolve(self) -> 'TextSignal':
        """Apply geometric convolution using GravTree"""
        kernel = self.gravtree.generate_kernel()
        return self.convolve(kernel)
    
    def filter_signal(self, cutoff_freq: float = 0.1) -> 'TextSignal':
        """Apply low-pass filter to text signal"""
        # Simple low-pass filter using moving average
        window_size = max(1, int(1.0 / cutoff_freq))
        kernel = jnp.ones(window_size) / window_size
        return self.convolve(kernel)
    
    def detect_patterns(self) -> List[Tuple[int, int, float]]:
        """Detect patterns in text signal using GravTree"""
        patterns = self.gravtree.detect_patterns(self.text)
        return [(0, len(self.text), 1.0)]  # Simplified for now
    
    def to_combit_array(self, pascal_order: int = 3, qdim: Optional[QDim] = None) -> CombitArray:
        """Convert to quaternion-dimensional array for advanced operations"""
        if qdim is None:
            from .combinion import identity_qdim
            qdim = identity_qdim()
        
        return CombitArray(self.signal, pascal_order, qdim)
    
    def to_file(self, filepath: Union[str, Path]):
        """Save text signal back to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.text)
    
    def __repr__(self) -> str:
        signal_stats = f"signal_len={len(self.signal)}, mean={jnp.mean(self.signal):.2f}"
        return f"TextSignal(text_len={len(self.text)}, encoding={self.encoding}, {signal_stats})"


class TemplateRenderer:
    """
    Advanced template rendering for piecewise file updates
    """
    
    def __init__(self, gravtree: Optional[GravTree] = None):
        self.gravtree = gravtree or GravTree()
        self.templates: Dict[str, str] = {}
    
    def register_template(self, name: str, template: str):
        """Register a named template"""
        self.templates[name] = template
    
    def render_function_template(self, function_name: str, args: List[str], body: str = "pass") -> str:
        """Render function template"""
        args_str = ", ".join(args)
        return f"def {function_name}({args_str}):\n    {body}"
    
    def render_class_template(self, class_name: str, methods: List[str], 
                            inheritance: str = "") -> str:
        """Render class template"""
        inherit_str = f"({inheritance})" if inheritance else ""
        methods_str = "\n    ".join(methods)
        return f"class {class_name}{inherit_str}:\n    {methods_str}"
    
    def render_import_template(self, modules: List[str], from_module: str = "") -> str:
        """Render import template"""
        if from_module:
            return f"from {from_module} import {', '.join(modules)}"
        else:
            return "\n".join(f"import {module}" for module in modules)
    
    def apply_to_signal(self, signal: TextSignal, template_name: str, 
                       mask: AttentionMask, **kwargs) -> TextSignal:
        """Apply named template to text signal with mask"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name].format(**kwargs)
        return signal.apply_template(template, mask)


class FileSignalProcessor:
    """
    Multi-file signal processing using quaternion dimensions
    """
    
    def __init__(self):
        self.files: Dict[str, TextSignal] = {}
        self.renderer = TemplateRenderer()
    
    def load_files(self, filepaths: List[Union[str, Path]]) -> 'FileSignalProcessor':
        """Load multiple files as signals"""
        for filepath in filepaths:
            key = str(Path(filepath).name)
            self.files[key] = TextSignal.from_file(filepath)
        return self
    
    def apply_cross_file_template(self, template: str, pattern: str, 
                                 file_filter: Optional[Callable[[str], bool]] = None) -> 'FileSignalProcessor':
        """Apply template across multiple files with pattern matching"""
        for filename, signal in self.files.items():
            if file_filter is None or file_filter(filename):
                mask = AttentionMask.from_pattern(pattern, signal.text)
                if len(mask.regions) > 0:  # Only apply if pattern found
                    updated_signal = signal.apply_template(template, mask)
                    self.files[filename] = updated_signal
        return self
    
    def sync_imports(self, import_template: str) -> 'FileSignalProcessor':
        """Synchronize imports across all Python files"""
        python_files = {k: v for k, v in self.files.items() if k.endswith('.py')}
        
        for filename, signal in python_files.items():
            # Find import section (top of file)
            import_mask = AttentionMask.from_pattern(r'^import.*$|^from.*import.*$', 
                                                    signal.text, intensity=1.0)
            if len(import_mask.regions) > 0:
                updated_signal = signal.apply_template(import_template, import_mask)
                self.files[filename] = updated_signal
        
        return self
    
    def save_all(self, output_dir: Optional[Union[str, Path]] = None):
        """Save all files back to disk"""
        for filename, signal in self.files.items():
            if output_dir:
                filepath = Path(output_dir) / filename
            else:
                filepath = filename
            signal.to_file(filepath)


# Convenience functions for common text signal operations
def load_and_process_file(filepath: Union[str, Path], 
                         pattern: str, 
                         template: str) -> TextSignal:
    """Load file, apply pattern mask, and render template"""
    signal = TextSignal.from_file(filepath)
    mask = AttentionMask.from_pattern(pattern, signal.text)
    return signal.apply_template(template, mask)

def batch_update_files(filepaths: List[Union[str, Path]], 
                      pattern: str, 
                      template: str, 
                      output_dir: Optional[Union[str, Path]] = None):
    """Batch update multiple files with pattern and template"""
    processor = FileSignalProcessor()
    processor.load_files(filepaths)
    processor.apply_cross_file_template(template, pattern)
    processor.save_all(output_dir)

def create_geometric_mask(text: str, focus_center: float = 0.5, 
                         focus_width: float = 0.3) -> AttentionMask:
    """Create geometric attention mask for selective focus"""
    return AttentionMask.geometric_focus(text, focus_center, focus_width)


__all__ = [
    "TextSignal", "AttentionMask", "TemplateRenderer", "FileSignalProcessor",
    "load_and_process_file", "batch_update_files", "create_geometric_mask"
]