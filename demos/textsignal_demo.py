#!/usr/bin/env python3
"""
TextSignal Demo - Revolutionary Text as Signal Processing

Demonstrates the groundbreaking approach of treating text files as signals
with attention masks for selective updates and template rendering.
"""

import numpy as np
from aemergent.textsignal import (
    TextSignal, AttentionMask, TemplateRenderer, FileSignalProcessor,
    load_and_process_file, create_geometric_mask
)

def demo_basic_text_signals():
    """Demo basic text-as-signal processing"""
    print("📡 TEXTSIGNAL DEMO - TEXT AS SIGNAL PROCESSING")
    print("=" * 60)
    
    # Sample Python code
    code = '''def calculate_sum(a, b):
    return a + b

def calculate_product(a, b):
    result = a * b
    return result

class Calculator:
    def add(self, x, y):
        return x + y
'''
    
    # Create text signal
    signal = TextSignal(code)
    print(f"📊 Text Signal: {signal}")
    print(f"   Signal stats: min={np.min(signal.signal)}, max={np.max(signal.signal)}")

def demo_attention_masks():
    """Demo various attention mask types"""
    print("\n🎯 ATTENTION MASKS - SELECTIVE FOCUS")
    print("=" * 40)
    
    code = '''import numpy as np
import pandas as pd

def process_data(data):
    cleaned = data.dropna()
    return cleaned.values

class DataProcessor:
    def __init__(self):
        self.data = None
    
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        return self.data
'''
    
    # Function mask
    func_mask = AttentionMask.from_functions(code)
    print(f"🔧 Function mask: {len(func_mask.regions)} functions found")
    for i, (start, end) in enumerate(func_mask.regions[:2]):
        print(f"   Function {i+1}: chars {start}-{end}")
    
    # Pattern mask for imports
    import_mask = AttentionMask.from_pattern(r'^import.*$|^from.*import.*$', code)
    print(f"📦 Import mask: {len(import_mask.regions)} imports found")
    
    # Geometric focus mask
    geo_mask = create_geometric_mask(code, focus_center=0.3, focus_width=0.2)
    print(f"🌐 Geometric mask: {len(geo_mask.regions)} regions with Gaussian focus")
    
    # Combine masks
    combined = func_mask & import_mask
    print(f"🔗 Combined mask: {len(combined.regions)} regions (functions AND imports)")

def demo_template_rendering():
    """Demo template rendering with masks"""
    print("\n🏗️ TEMPLATE RENDERING - PIECEWISE UPDATES")
    print("=" * 45)
    
    original_code = '''def old_function(x):
    return x * 2

def another_old_function(y):
    return y + 1
'''
    
    # Create signal and function mask
    signal = TextSignal(original_code)
    mask = AttentionMask.from_functions(original_code)
    
    print("📝 Original code:")
    print(original_code)
    
    # Apply template to update functions
    new_template = '''def enhanced_function(value, multiplier=3):
    """Enhanced function with better parameters"""
    return value * multiplier'''
    
    updated = signal.apply_template(new_template, mask)
    print("\n🚀 After template application:")
    print(updated.text[:150] + "..." if len(updated.text) > 150 else updated.text)

def demo_signal_processing():
    """Demo signal processing operations on text"""
    print("\n📈 SIGNAL PROCESSING - CONVOLUTION & FILTERING")
    print("=" * 50)
    
    # Create noisy text signal
    text = "This is a test string with some NOISE and irregularities!!!"
    signal = TextSignal(text)
    
    print(f"📊 Original signal stats:")
    print(f"   Mean: {np.mean(signal.signal):.2f}")
    print(f"   Std:  {np.std(signal.signal):.2f}")
    
    # Apply smoothing filter
    filtered = signal.filter_signal(cutoff_freq=0.15)
    print(f"\n🔄 After filtering:")
    print(f"   Mean: {np.mean(filtered.signal):.2f}")
    print(f"   Std:  {np.std(filtered.signal):.2f}")
    
    # Apply geometric convolution
    geo_convolved = signal.geometric_convolve()
    print(f"\n🌀 After geometric convolution:")
    print(f"   Mean: {np.mean(geo_convolved.signal):.2f}")

def demo_quaternion_integration():
    """Demo integration with quaternion-dimensional arrays"""
    print("\n🔮 QUATERNION INTEGRATION - 4D TEXT PROCESSING")
    print("=" * 50)
    
    text = "def quantum_function(): return 'amazing'"
    signal = TextSignal(text)
    
    # Convert to quaternion-dimensional array
    combit_array = signal.to_combit_array(pascal_order=4)
    print(f"🎯 Quaternion array: {combit_array}")
    
    # Rotate dimensions while preserving text signal structure
    rotated_array = combit_array.rotate_dims('y', np.pi/6)
    print(f"🌀 After rotation: {rotated_array}")
    print(f"   Pascal order preserved: {rotated_array.pascal_order}")

def demo_advanced_templates():
    """Demo advanced template rendering system"""
    print("\n🎨 ADVANCED TEMPLATES - CODE GENERATION")
    print("=" * 40)
    
    renderer = TemplateRenderer()
    
    # Register templates
    renderer.register_template("api_endpoint", '''
@app.route('/{endpoint}', methods=['GET', 'POST'])
def {function_name}():
    """Generated API endpoint"""
    return jsonify({{"status": "success"}})
''')
    
    # Create sample code
    code = '''# API endpoints will be generated here
def placeholder():
    pass
'''
    
    signal = TextSignal(code)
    mask = AttentionMask.from_pattern(r'def placeholder.*?pass', code)
    
    # Apply template
    updated = renderer.apply_to_signal(
        signal, "api_endpoint", mask,
        endpoint="users", function_name="handle_users"
    )
    
    print("🏗️ Generated API endpoint:")
    print(updated.text)

def demo_multi_file_processing():
    """Demo multi-file signal processing"""
    print("\n📁 MULTI-FILE PROCESSING - BATCH OPERATIONS")
    print("=" * 45)
    
    # Simulate multiple Python files
    file_contents = {
        "utils.py": '''def helper_function():
    return "old implementation"

def another_helper():
    pass
''',
        "main.py": '''from utils import helper_function

def main():
    result = helper_function()
    print(result)
''',
        "config.py": '''DEBUG = True
VERSION = "1.0.0"

def get_config():
    return {"debug": DEBUG}
'''
    }
    
    # Create file processor
    processor = FileSignalProcessor()
    
    # Simulate loading files
    for filename, content in file_contents.items():
        processor.files[filename] = TextSignal(content)
    
    print(f"📁 Loaded {len(processor.files)} files")
    
    # Apply cross-file template update
    function_template = '''def updated_function():
    """Updated implementation with better logic"""
    return "new and improved"'''
    
    processor.apply_cross_file_template(
        function_template, 
        r'def \w+\([^)]*\):\s*.*?(?=\ndef|\nclass|\Z)',
        file_filter=lambda f: f.endswith('.py')
    )
    
    print("🔄 Applied template across all Python files")
    print(f"   Example update in utils.py:")
    print(processor.files["utils.py"].text[:100] + "...")

def demo_practical_use_cases():
    """Demo practical real-world use cases"""
    print("\n🌟 PRACTICAL USE CASES - REAL-WORLD APPLICATIONS")
    print("=" * 55)
    
    print("🔧 Use Case 1: API Documentation Generation")
    print("   - Load Python files as signals")
    print("   - Create masks for function definitions")
    print("   - Apply docstring templates")
    print("   - Generate documentation automatically")
    
    print("\n🔄 Use Case 2: Code Refactoring")
    print("   - Pattern-based function detection")
    print("   - Template-based modernization")
    print("   - Batch updates across codebases")
    
    print("\n🎯 Use Case 3: Configuration Management")
    print("   - Geometric masks for config sections")
    print("   - Environment-specific templates")
    print("   - Cross-file consistency")
    
    print("\n🌐 Use Case 4: Multi-language Code Generation")
    print("   - Signal processing for syntax patterns")
    print("   - Template rendering for different languages")
    print("   - Quaternion dimensions for multi-file operations")

if __name__ == "__main__":
    demo_basic_text_signals()
    demo_attention_masks()
    demo_template_rendering()
    demo_signal_processing()
    demo_quaternion_integration()
    demo_advanced_templates()
    demo_multi_file_processing()
    demo_practical_use_cases()
    
    print("\n🎉 TEXTSIGNAL DEMO COMPLETE!")
    print("=" * 35)
    print("Revolutionary capabilities demonstrated:")
    print("✅ Text files as 1D signal arrays")
    print("✅ Attention masks for selective updates")
    print("✅ Template rendering for piecewise modifications")
    print("✅ Signal processing (convolution, filtering)")
    print("✅ Quaternion-dimensional text operations")
    print("✅ Multi-file batch processing")
    print("✅ Pattern-based code generation")
    print("✅ Geometric focus and signal analysis")
    print("\n🚀 File editing will never be the same!")