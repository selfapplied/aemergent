"""Change detector for outputs of functions to help preserve meaning.

Use: 

@printout
def main():
    "Hello, world!"
    print("Hello, world!")
"""

import os
import pickle
import difflib
from functools import wraps
from pathlib import Path
import hashlib
import inspect
import ast
import re

def setdoc(file, fname, doc):
    """Set documentation for a function output"""
    doc_path = Path(file).parent / f"{fname}_doc.txt"
    with open(doc_path, 'w') as f:
        f.write(doc)

def showchange(out1, out2):
    """Show differences between two outputs using unified diff"""
    if out1 == out2:
        return "No changes detected."
    
    # Convert outputs to string representations for comparison
    str1 = str(out1).splitlines(keepends=True)
    str2 = str(out2).splitlines(keepends=True)
    
    diff = list(difflib.unified_diff(
        str1, str2, 
        fromfile='previous', tofile='current', 
        lineterm=''
    ))
    
    if not diff:
        return "No changes detected."
    
    return ''.join(diff)

def printout(fn):
    """Decorator that injects function output directly into the source code as comments"""
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Get the source file where this function is defined
        source_file = inspect.getfile(fn)
        
        # Capture function output
        import sys
        from io import StringIO
        
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Execute the function
            result = fn(*args, **kwargs)
            
            # Get the captured output
            output_text = captured_output.getvalue()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Print the output (so user still sees it)
            print(output_text, end='')
            
            # Inject output into source code
            inject_output_into_source(source_file, fn.__name__, output_text)
            
            return result
            
        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = old_stdout
            raise e
        finally:
            # Ensure stdout is always restored
            sys.stdout = old_stdout
    
    return wrapper

def inject_output_into_source(file_path, function_name, output_text):
    """Inject captured output directly into the function's docstring using inspect"""
    try:
        # First, get the function object from the module
        import importlib.util
        import sys
        
        # Load the module to get the function object
        spec = importlib.util.spec_from_file_location("temp_module", file_path)
        if spec is None or spec.loader is None:
            print(f"⚠️  Could not load module spec from {file_path}")
            return
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function object
        func_obj = getattr(module, function_name)
        
        # Use inspect to get the source lines and starting line number
        source_lines, start_line_num = inspect.getsourcelines(func_obj)
        
        # Read the entire file
        with open(file_path, 'r') as f:
            file_lines = f.readlines()
        
        # Find where to inject the docstring (right after the def line)
        def_line_index = start_line_num - 1  # Convert to 0-based indexing
        
        # Look for existing docstring
        docstring_start = None
        docstring_end = None
        
        for i, line in enumerate(source_lines):
            stripped = line.strip()
            if i > 0 and (stripped.startswith('"""') or stripped.startswith("'''")):
                docstring_start = def_line_index + i
                # Find the end of the docstring
                quote_type = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(quote_type) >= 2:
                    # Single line docstring
                    docstring_end = docstring_start + 1
                else:
                    # Multi-line docstring
                    for j in range(i + 1, len(source_lines)):
                        if quote_type in source_lines[j]:
                            docstring_end = def_line_index + j + 1
                            break
                break
        
        # Create new docstring with output
        indent = "    "  # Standard Python indentation
        docstring_lines = []
        docstring_lines.append(f'{indent}"""\n')
        
        for line in output_text.strip().split('\n'):
            docstring_lines.append(f'{indent}{line}\n')
        
        docstring_lines.append(f'{indent}"""\n')
        
        # Insert or replace the docstring
        if docstring_start is not None and docstring_end is not None:
            # Replace existing docstring
            new_file_lines = (file_lines[:docstring_start] + 
                            docstring_lines + 
                            file_lines[docstring_end:])
        else:
            # Insert new docstring after the def line
            insert_point = def_line_index + 1
            new_file_lines = (file_lines[:insert_point] + 
                            docstring_lines + 
                            file_lines[insert_point:])
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.writelines(new_file_lines)
            
        print(f"✨ Output injected into docstring of {function_name} at line {start_line_num}")
        
    except Exception as e:
        print(f"⚠️  Failed to inject output into docstring: {e}")
        import traceback
        traceback.print_exc()

def remove_existing_output_block(content, function_name):
    """Remove any existing output block for the specified function"""
    # Pattern to match output blocks: # OUTPUT: ... # END OUTPUT
    pattern = r'# OUTPUT:.*?# END OUTPUT\n?'
    return re.sub(pattern, '', content, flags=re.DOTALL)