#!/usr/bin/env python3
"""
Enhanced Python Dependency Scanner - FIXED VERSION
Scans all Python files in a directory tree and extracts import statements
Creates a test file only for installable dependencies (excludes stdlib and local modules)
"""

import os
import re
import sys
import ast
from collections import defaultdict, Counter
from pathlib import Path
import pkgutil
import importlib.util

class DependencyScanner:
    def __init__(self):
        self.imports = set()
        self.from_imports = defaultdict(set)
        self.file_imports = defaultdict(set)
        self.failed_files = []
        
        # Enhanced standard library detection
        self.stdlib_modules = self._get_stdlib_modules()
        
    def _get_stdlib_modules(self):
        """Get comprehensive list of standard library modules"""
        # Core standard library modules (Python 3.8+)
        stdlib_modules = {
            # Built-in modules
            'sys', 'os', 'time', 'math', 'random', 'json', 're', 'io', 'gc',
            
            # Collections and data structures
            'collections', 'array', 'bisect', 'heapq', 'copy', 'weakref',
            'types', 'enum', 'dataclasses',
            
            # Text processing
            'string', 'textwrap', 'unicodedata', 'stringprep', 'readline',
            'difflib', 'locale',
            
            # File and directory access
            'pathlib', 'glob', 'fnmatch', 'linecache', 'tempfile', 'shutil',
            'fileinput', 'filecmp', 'stat', 'mmap',
            
            # Data persistence
            'pickle', 'copyreg', 'shelve', 'dbm', 'sqlite3',
            
            # Data compression and archiving
            'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
            
            # File formats
            'csv', 'configparser', 'netrc', 'plistlib', 'xdrlib',
            
            # Cryptographic services
            'hashlib', 'hmac', 'secrets', 'ssl',
            
            # Generic operating system services
            'argparse', 'logging', 'getpass', 'curses', 'platform', 'errno',
            'ctypes',
            
            # Concurrent execution
            'threading', 'multiprocessing', 'concurrent', 'subprocess', 'queue',
            'sched', 'signal', 'asyncio',
            
            # Context variables
            'contextvars',
            
            # Networking and interprocess communication
            'socket', 'selectors', 'asyncore', 'asynchat', 'uuid', 'socketserver',
            'http', 'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib',
            'telnetlib', 'urllib', 'wsgiref', 'xmlrpc',
            
            # Internet data handling
            'email', 'mailcap', 'mailbox', 'mimetypes', 'base64', 'binascii',
            'quopri', 'uu',
            
            # Structured markup processing
            'html', 'xml',
            
            # Internet protocols and support
            'cgi', 'cgitb', 'wsgiref',
            
            # Multimedia services
            'audioop', 'aifc', 'sunau', 'wave', 'chunk', 'colorsys', 'imghdr',
            'sndhdr', 'ossaudiodev',
            
            # Internationalization
            'gettext',
            
            # Program frameworks
            'cmd', 'shlex',
            
            # Tk GUI
            'tkinter',
            
            # Development tools
            'typing', 'pydoc', 'doctest', 'unittest', 'test', '2to3', 'lib2to3',
            
            # Debugging and profiling
            'bdb', 'faulthandler', 'pdb', 'profile', 'pstats', 'timeit', 'trace',
            'tracemalloc',
            
            # Software packaging and distribution
            'distutils', 'ensurepip', 'venv', 'zipapp',
            
            # Python runtime services
            'atexit', 'traceback', 'future', '__future__', 'warnings',
            'contextlib', 'abc', 'rlcompleter', 'reprlib',
            
            # Importing modules
            'zipimport', 'pkgutil', 'modulefinder', 'runpy', 'importlib',
            
            # Python language services
            'parser', 'ast', 'symtable', 'symbol', 'token', 'keyword', 'tokenize',
            'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis', 'pickletools',
            'formatter',
            
            # MS Windows specific
            'msilib', 'msvcrt', 'winreg', 'winsound',
            
            # Unix specific
            'posix', 'pwd', 'spwd', 'grp', 'crypt', 'termios', 'tty', 'pty',
            'fcntl', 'pipes', 'resource', 'nis', 'syslog',
            
            # Other
            'calendar', 'codecs', 'functools', 'itertools', 'operator',
            'statistics', 'decimal', 'fractions', 'numbers', 'cmath',
            'inspect', 'site'
        }
        
        # Add dynamically discovered stdlib modules
        try:
            import sys
            for module_info in pkgutil.iter_modules():
                if module_info.name in sys.builtin_module_names:
                    stdlib_modules.add(module_info.name)
        except:
            pass
            
        return stdlib_modules
    
    def extract_imports_from_file(self, filepath):
        """Extract all import statements from a Python file using AST"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse with AST for more reliable extraction
            try:
                tree = ast.parse(content)
            except SyntaxError:
                # If AST fails, try regex as fallback
                return self.extract_imports_regex(content, filepath)
                
            file_imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]  # Get top-level module
                        self.imports.add(module)
                        file_imports.add(module)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]  # Get top-level module
                        self.imports.add(module)
                        file_imports.add(module)
                        
                        # Track specific from imports
                        for alias in node.names:
                            self.from_imports[module].add(alias.name)
            
            self.file_imports[filepath] = file_imports
            return True
            
        except Exception as e:
            self.failed_files.append((filepath, str(e)))
            return False
    
    def extract_imports_regex(self, content, filepath):
        """Fallback regex-based import extraction"""
        file_imports = set()
        
        # Regular import pattern: import module, module2
        import_pattern = r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
        # From import pattern: from module import ...
        from_pattern = r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
        
        for line in content.split('\n'):
            # Skip comments
            if line.strip().startswith('#'):
                continue
                
            # Regular imports
            match = re.match(import_pattern, line)
            if match:
                module = match.group(1).split('.')[0]
                self.imports.add(module)
                file_imports.add(module)
            
            # From imports
            match = re.match(from_pattern, line)
            if match:
                module = match.group(1).split('.')[0]
                self.imports.add(module)
                file_imports.add(module)
        
        self.file_imports[filepath] = file_imports
        return True
    
    def scan_directory(self, directory, recursive=True):
        """Scan directory for Python files and extract imports"""
        directory = Path(directory)
        
        if not directory.exists():
            print(f"Directory {directory} does not exist!")
            return
        
        pattern = "**/*.py" if recursive else "*.py"
        python_files = list(directory.glob(pattern))
        
        print(f"Scanning {len(python_files)} Python files in {directory}...")
        
        for py_file in python_files:
            self.extract_imports_from_file(py_file)
        
        print(f"Scan complete! Found {len(self.imports)} unique imports.")
        if self.failed_files:
            print(f"Warning: Failed to parse {len(self.failed_files)} files.")
    
    def _is_local_module(self, module_name, scan_directory):
        """Check if module is a local project module"""
        scan_path = Path(scan_directory)
        
        # Check for .py file
        if (scan_path / f"{module_name}.py").exists():
            return True
        
        # Check for package directory
        if (scan_path / module_name / "__init__.py").exists():
            return True
        
        # Check recursively in subdirectories
        for subdir in scan_path.rglob(module_name):
            if subdir.is_dir() and (subdir / "__init__.py").exists():
                return True
        
        return False
    
    def categorize_imports(self, scan_directory="."):
        """Enhanced categorization of imports"""
        standard_lib = set()
        third_party = set()
        local_modules = set()
        unknown = set()
        
        for module in self.imports:
            # Check if it's a standard library module
            if module in self.stdlib_modules:
                standard_lib.add(module)
                continue
            
            # Check if it's a local module
            if self._is_local_module(module, scan_directory):
                local_modules.add(module)
                continue
            
            # Try to find the module to determine if it's third-party
            try:
                spec = importlib.util.find_spec(module)
                if spec is None:
                    unknown.add(module)
                elif spec.origin:
                    if 'site-packages' in spec.origin or 'dist-packages' in spec.origin:
                        third_party.add(module)
                    elif any(path in spec.origin for path in ['/usr/lib/python', '/usr/local/lib/python']):
                        # System-installed packages that might be third-party
                        third_party.add(module)
                    else:
                        # Likely local or standard library
                        if self._is_local_module(module, scan_directory):
                            local_modules.add(module)
                        else:
                            standard_lib.add(module)
                else:
                    # Built-in module
                    standard_lib.add(module)
                    
            except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
                # If we can't find it, it's likely a third-party package that needs installation
                unknown.add(module)
        
        return standard_lib, third_party, local_modules, unknown
    
    def generate_test_file(self, output_file="installable_dependencies_test.py", scan_directory="."):
        """Generate a test file ONLY for installable dependencies"""
        standard_lib, third_party, local_modules, unknown = self.categorize_imports(scan_directory)
        
        # Only test third-party and unknown modules (potentially installable)
        installable_modules = third_party.union(unknown)
        
        if not installable_modules:
            print("No installable dependencies found! All imports are either standard library or local modules.")
            return None
        
        test_content = '''#!/usr/bin/env python3
"""
AUTO-GENERATED Installable Dependencies Test Script
Generated by scanning Python files for third-party package imports
Tests ONLY packages that need to be installed (pip/conda/etc.)
Excludes: Standard library modules, local project modules
"""

import sys
import traceback
import subprocess
import pkg_resources

def check_if_installed_via_pip(package_name):
    """Check if package is installed via pip"""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def suggest_install_command(module_name):
    """Suggest installation command for failed imports"""
    # Common module name to package name mappings
    package_mappings = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'dateutil': 'python-dateutil',
        'serial': 'pyserial',
        'psutil': 'psutil',
        'requests': 'requests',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'torch': 'torch',
        'tensorflow': 'tensorflow',
        'transformers': 'transformers',
        'whisper': 'openai-whisper',
        'TTS': 'TTS',
        'pydub': 'pydub',
        'librosa': 'librosa',
    }
    
    package_name = package_mappings.get(module_name, module_name)
    return f"pip install {package_name}"

def test_import(module_name, import_statement=None, test_function=None):
    """Test importing a module and optionally run a test function"""
    try:
        if import_statement:
            exec(import_statement)
        else:
            __import__(module_name)
        
        # Check if installed via pip for additional info
        pip_installed = check_if_installed_via_pip(module_name)
        pip_info = " (pip)" if pip_installed else " (system/conda?)"
        
        result = f"✅ INSTALLED{pip_info}"
        if test_function:
            try:
                test_function()
                result += " + FUNCTIONAL"
            except Exception as e:
                result += f" - FUNCTION ERROR: {str(e)}"
        
        print(f"{module_name:<25}: {result}")
        return True
        
    except Exception as e:
        install_cmd = suggest_install_command(module_name)
        print(f"{module_name:<25}: ❌ MISSING - {str(e)}")
        print(f"{'':25}   💡 Try: {install_cmd}")
        return False

# Test functions for specific modules (only for installable packages)
def test_torch():
    """Test PyTorch with CUDA"""
    import torch
    print(f"{'':25}   → Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"{'':25}   → CUDA: {torch.version.cuda}")
        print(f"{'':25}   → GPU: {torch.cuda.get_device_name()}")
        # Test basic CUDA operation
        x = torch.randn(2, 2).cuda()
        y = x + x
    else:
        print(f"{'':25}   → CUDA: Not available")

def test_numpy():
    """Test NumPy"""
    import numpy as np
    print(f"{'':25}   → Version: {np.__version__}")
    # Test basic operation
    arr = np.array([1, 2, 3])
    print(f"{'':25}   → Basic test: {arr.sum()}")

def test_pandas():
    """Test Pandas"""
    import pandas as pd
    print(f"{'':25}   → Version: {pd.__version__}")
    # Test basic operation
    df = pd.DataFrame({'a': [1, 2, 3]})
    print(f"{'':25}   → Basic test: {len(df)} rows")

def test_requests():
    """Test Requests"""
    import requests
    print(f"{'':25}   → Version: {requests.__version__}")

def test_whisper():
    """Test Whisper"""
    import whisper
    models = whisper.available_models()
    print(f"{'':25}   → Available models: {len(models)}")

def test_transformers():
    """Test Transformers"""
    import transformers
    print(f"{'':25}   → Version: {transformers.__version__}")

def test_tokenizers():
    """Test Tokenizers"""
    import tokenizers
    print(f"{'':25}   → Version: {tokenizers.__version__}")
    # Try to get CUDA info if available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"{'':25}   → CUDA: {torch.version.cuda}")
            print(f"{'':25}   → GPU: {torch.cuda.get_device_name()}")
    except:
        pass

def test_regex():
    """Test regex"""
    import regex
    print(f"{'':25}   → Version: {regex.__version__}")

def test_num2words():
    """Test num2words"""
    import num2words
    print(f"{'':25}   → Version: {num2words.__version__}")
    # Test basic functionality
    result = num2words.num2words(6)
    print(f"{'':25}   → Basic test: {result}")

# Enhanced test functions mapping
test_functions = {
    'torch': test_torch,
    'numpy': test_numpy,
    'pandas': test_pandas,
    'requests': test_requests,
    'whisper': test_whisper,
    'transformers': test_transformers,
    'tokenizers': test_tokenizers,
    'regex': test_regex,
    'num2words': test_num2words,
}

print("="*80)
print("INSTALLABLE DEPENDENCIES TEST")
print("="*80)
print("Testing only third-party packages (excludes stdlib & local modules)")
print()

'''

        # Add tests for installable modules only
        if installable_modules:
            test_content += '''print("THIRD-PARTY PACKAGES FOUND IN YOUR CODE:")
print("-" * 50)
'''
            for module in sorted(installable_modules):
                # Add specific import statements for complex modules
                special_imports = {
                    'TTS': 'from TTS.api import TTS',
                    'pydub': 'from pydub import AudioSegment',
                    'PIL': 'from PIL import Image',
                    'cv2': 'import cv2',
                    'sklearn': 'import sklearn',
                }
                
                if module in special_imports:
                    import_stmt = f'"{special_imports[module]}"'
                else:
                    import_stmt = 'None'
                
                test_func = 'test_functions.get("{}")'.format(module)
                test_content += f'test_import("{module}", import_statement={import_stmt}, test_function={test_func})\n'
        
        # FIXED: Complete the summary section properly
        test_content += '''
print()
print("="*80)
print("INSTALLATION HELP:")
print("="*80)
print("For missing packages, you can typically install them using:")
print("  • pip install <package_name>")
print("  • conda install <package_name>")
print("  • pip install -r requirements.txt (if you have a requirements file)")
print()
print("For GPU support (PyTorch, TensorFlow), visit their official websites")
print("for platform-specific installation instructions.")
print()

# FIXED: Summary counter
print("SUMMARY:")
print("-" * 20)

# Re-run a quick test to count failures
missing_modules = []
total_installable = len({})

for module_name in sorted({}):
    try:
        __import__(module_name)
    except ImportError:
        missing_modules.append(module_name)

installed_count = total_installable - len(missing_modules)
print(f"Total installable packages found: {{total_installable}}")
print(f"Currently installed: {{installed_count}}")
print(f"Missing/Need installation: {{len(missing_modules)}}")

if missing_modules:
    print(f"Missing modules: {{', '.join(missing_modules)}}")

print()
print("Run this script regularly to check your environment setup!")
'''.format(
            repr(sorted(installable_modules)), 
            repr(sorted(installable_modules))
        )

        # Write the test file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print(f"Generated test file: {output_file}")
        print(f"Found {len(installable_modules)} installable dependencies:")
        for module in sorted(installable_modules):
            print(f"  • {module}")
        
        print(f"\nExcluded from test:")
        print(f"  • {len(standard_lib)} standard library modules")
        print(f"  • {len(local_modules)} local project modules")
        
        return output_file
    
    def print_report(self, scan_directory="."):
        """Print a comprehensive report of all findings"""
        standard_lib, third_party, local_modules, unknown = self.categorize_imports(scan_directory)
        
        print("\n" + "="*80)
        print("DEPENDENCY SCAN REPORT")
        print("="*80)
        
        print(f"\nTOTAL IMPORTS FOUND: {len(self.imports)}")
        print(f"FILES SCANNED: {len(self.file_imports)}")
        if self.failed_files:
            print(f"FAILED FILES: {len(self.failed_files)}")
        
        print(f"\nSTANDARD LIBRARY ({len(standard_lib)}):")
        for module in sorted(standard_lib):
            print(f"  • {module}")
        
        print(f"\nTHIRD-PARTY PACKAGES ({len(third_party)}):")
        for module in sorted(third_party):
            print(f"  • {module}")
        
        print(f"\nLOCAL/PROJECT MODULES ({len(local_modules)}):")
        for module in sorted(local_modules):
            print(f"  • {module}")
        
        if unknown:
            print(f"\nUNKNOWN/UNINSTALLED ({len(unknown)}):")
            for module in sorted(unknown):
                print(f"  • {module} (may need installation)")
        
        if self.failed_files:
            print(f"\nFAILED TO PARSE:")
            for filepath, error in self.failed_files:
                print(f"  • {filepath}: {error}")

def main():
    """Main function to run the dependency scanner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scan Python files for dependencies')
    parser.add_argument('directory', nargs='?', default='.', 
                       help='Directory to scan (default: current directory)')
    parser.add_argument('--output', '-o', default='installable_dependencies_test.py',
                       help='Output file for dependency test script')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Don\'t scan subdirectories recursively')
    parser.add_argument('--no-test', action='store_true',
                       help='Don\'t generate test file, just show report')
    
    args = parser.parse_args()
    
    # Create scanner and scan directory
    scanner = DependencyScanner()
    scanner.scan_directory(args.directory, recursive=not args.no_recursive)
    
    # Print comprehensive report
    scanner.print_report(args.directory)
    
    # Generate test file unless disabled
    if not args.no_test:
        print("\n" + "="*80)
        test_file = scanner.generate_test_file(args.output, args.directory)
        if test_file:
            print(f"\nTo test your environment, run:")
            print(f"  python {test_file}")

if __name__ == "__main__":
    main()