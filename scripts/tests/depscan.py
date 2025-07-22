#!/usr/bin/env python3
"""
Enhanced Python Dependency Scanner
Scans .py files, categorizes imports, and generates an auto-installing tester.
"""

import os
import re
import sys
import ast
import warnings
import subprocess
import urllib.request
import importlib.util
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------------
# Helper to suppress warnings in generated test file
WARN_SUPPRESS = '''import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
'''

# ------------------------------------------------------------------
class DependencyScanner:
    def __init__(self):
        self.imports = set()
        self.from_imports = defaultdict(set)
        self.file_imports = defaultdict(set)
        self.failed_files = []
        # comprehensive stdlib list (trimmed for brevity)
        self.stdlib_modules = {
            "sys", "os", "time", "re", "json", "math", "random", "pathlib", "typing",
            "collections", "itertools", "functools", "threading", "subprocess",
            "importlib", "pkgutil", "ast", "io", "gc", "ssl", "http", "urllib",
            "argparse", "logging", "shutil", "tempfile", "zipfile", "csv",
            "sqlite3", "hashlib", "secrets", "configparser", "pickle",
            "xml", "email", "html", "tkinter", "unittest", "warnings",
            "ctypes", "mmap", "asyncio", "concurrent", "multiprocessing"
        }

    # ------------------------------------------------------------------
    # Standard library discovery
    def _get_stdlib_modules(self):
        try:
            import pkgutil
            for m in pkgutil.iter_modules():
                if m.name in sys.builtin_module_names:
                    self.stdlib_modules.add(m.name)
        except Exception:
            pass
        return self.stdlib_modules

    # ------------------------------------------------------------------
    # Import extraction
    def extract_imports_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            try:
                tree = ast.parse(content)
            except SyntaxError:
                return self.extract_imports_regex(content, filepath)

            file_imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        self.imports.add(module)
                        file_imports.add(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        self.imports.add(module)
                        file_imports.add(module)
                        for alias in node.names:
                            self.from_imports[module].add(alias.name)
            self.file_imports[filepath] = file_imports
            return True
        except Exception as e:
            self.failed_files.append((filepath, str(e)))
            return False

    def extract_imports_regex(self, content, filepath):
        file_imports = set()
        import_re = re.compile(r'^\s*import\s+([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)')
        from_re = re.compile(r'^\s*from\s+([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s+import')
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('#'):
                continue
            m = import_re.match(line) or from_re.match(line)
            if m:
                module = m.group(1).split('.')[0]
                self.imports.add(module)
                file_imports.add(module)
        self.file_imports[filepath] = file_imports
        return True

    # ------------------------------------------------------------------
    # Directory scan
    def scan_directory(self, directory, recursive=True):
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

    # ------------------------------------------------------------------
    # Local module detection
    def _is_local_module(self, module_name, scan_directory):
        scan_path = Path(scan_directory)
        if (scan_path / f"{module_name}.py").exists():
            return True
        if (scan_path / module_name / "__init__.py").exists():
            return True
        for subdir in scan_path.rglob(module_name):
            if subdir.is_dir() and (subdir / "__init__.py").exists():
                return True
        return False

    # ------------------------------------------------------------------
    # PyPI existence check (lightweight)
    def _pypi_exists(self, package_name):
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{package_name}/json", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Categorize imports
    def categorize_imports(self, scan_directory="."):
        standard, third, local, unknown = set(), set(), set(), set()
        for mod in self.imports:
            if mod in self.stdlib_modules:
                standard.add(mod)
            elif self._is_local_module(mod, scan_directory):
                local.add(mod)
            else:
                try:
                    spec = importlib.util.find_spec(mod)
                    if spec and ("site-packages" in (spec.origin or "") or "dist-packages" in (spec.origin or "")):
                        third.add(mod)
                    elif spec is None:
                        unknown.add(mod)
                    else:
                        standard.add(mod)
                except Exception:
                    unknown.add(mod)
        return standard, third, local, unknown

    def generate_test_file(self, output_file="installable_dependencies_test.py", scan_directory="."):
        """Generate a test file for installable deps + auto-install & retest."""
        standard, third, local, unknown = self.categorize_imports(scan_directory)
        installable = {m for m in third.union(unknown) if not self._is_local_module(m, scan_directory)}

        # PyPI mapping overrides
        pypi_map = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'sklearn': 'scikit-learn',
            'yaml': 'PyYAML',
            'serial': 'pyserial',
            'whisper': 'openai-whisper',
            'TTS': 'TTS',
            'lws': 'lws-python',
            'insightface': 'insightface',
            'dateutil': 'python-dateutil',
            'requests': 'requests',
            'matplotlib': 'matplotlib',
            'transformers': 'transformers',
            'tokenizers': 'tokenizers',
            'torch': 'torch',
            'torchvision': 'torchvision',
            'torchaudio': 'torchaudio',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'scipy': 'scipy',
            'gradio': 'gradio',
            'diffusers': 'diffusers',
            'einops': 'einops',
            'librosa': 'librosa',
            'pydub': 'pydub',
            'imageio': 'imageio',
            'soundfile': 'soundfile',
            'decord': 'decord',
            'mediapipe': 'mediapipe',
            'spacy': 'spacy',
            'nltk': 'nltk',
            'regex': 'regex',
            'num2words': 'num2words',
            'omegaconf': 'omegaconf',
            'accelerate': 'accelerate',
            'more_itertools': 'more-itertools',
            'packaging': 'packaging',
            'scenedetect': 'scenedetect',
            'python_speech_features': 'python_speech_features',
            'pypinyin': 'pypinyin',
            'cutlet': 'cutlet',
            'hangul_romanize': 'hangul-romanize',
            'cog': 'cog',
        }

        # keep only packages we can actually install
        safe = []
        for m in sorted(installable):
            if not self._pypi_exists(m) and m not in pypi_map:
                continue
            safe.append(m)

        if not safe:
            print("No installable dependencies found.")
            return None

        # Build the final script
        test_content = f'''#!/usr/bin/env python3
"""
AUTO-GENERATED Installable Dependencies Test Script
(Enhanced – auto-install missing packages via pip, CUDA-aware for torch-family)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import os, sys, subprocess, pkg_resources, re, importlib.util

pypi_map = {repr(pypi_map)}
def pypi(mod): return pypi_map.get(mod, mod)

def installed(mod):
    try:
        pkg_resources.get_distribution(pypi(mod))
        return True
    except Exception:
        return importlib.util.find_spec(mod) is not None

def torch_cuda_info():
    \"\"\"Return (cuda_ver, cudnn_ver) or (None, None) if torch not present.\"\"\"
    try:
        import torch
        cuda = torch.version.cuda or ""
        cudnn = torch.backends.cudnn.version()
        if cudnn is not None:
            cudnn = str(cudnn)[:-2] if len(str(cudnn)) > 2 else str(cudnn)
        return cuda, cudnn
    except Exception:
        return None, None

def build_torch_flags(cuda_ver):
    if not cuda_ver:
        return []
    major, minor = cuda_ver.split(".")[:2]
    return ["--extra-index-url", f"https://download.pytorch.org/whl/cu{{major}}{{minor}}"]

def pip_install(pkg, *flags):
    cmd = [sys.executable, "-m", "pip", "install", *flags, pkg]
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode == 0

cuda_ver, cudnn_ver = torch_cuda_info()
torch_flags = build_torch_flags(cuda_ver)

missing = [m for m in {safe} if not installed(m)]

if missing:
    print("Missing packages:", ", ".join(missing))
    if input("Install them now? [y/N] ").strip().lower() == "y":
        for m in missing:
            pkg = pypi(m)
            if m in ("torch", "torchvision", "torchaudio") and cuda_ver:
                cuda_suffix = f"+cu{{''.join(cuda_ver.split('.')[:2])}}"
                pkg = f"{{pkg}}{{cuda_suffix}}"
            pip_install(pkg, *torch_flags)

print("\\n=== FINAL STATUS CHECK ===")
for m in sorted({safe}):
    try:
        __import__(m)
        print(f"{{m:<25}} ✅ OK")
    except Exception as e:
        print(f"{{m:<25}} ❌ {{e}}")
'''

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        print(f"Generated: {output_file}")
        return output_file

    def print_report(self, scan_directory="."):
        standard, third, local, unknown = self.categorize_imports(scan_directory)
        print("\n" + "=" * 80)
        print("DEPENDENCY SCAN REPORT")
        print("=" * 80)
        print(f"TOTAL IMPORTS: {len(self.imports)}")
        print(f"FILES SCANNED: {len(self.file_imports)}")
        if self.failed_files:
            print(f"FAILED FILES: {len(self.failed_files)}")
        for cat, name in [(standard, "STANDARD LIBRARY"),
                          (third, "THIRD-PARTY"),
                          (local, "LOCAL/PROJECT"),
                          (unknown, "UNKNOWN")]:
            if cat:
                print(f"\n{name} ({len(cat)}):")
                for m in sorted(cat):
                    print(f"  • {m}")

# ------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scan Python files for dependencies and generate auto-install tester.")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to scan")
    parser.add_argument("-o", "--output", default="installable_dependencies_test.py", help="Output tester filename")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirs")
    parser.add_argument("--no-test", action="store_true", help="Skip test file generation")
    args = parser.parse_args()

    scanner = DependencyScanner()
    scanner.scan_directory(args.directory, recursive=not args.no_recursive)
    scanner.print_report(args.directory)

    if not args.no_test:
        test_file = scanner.generate_test_file(args.output, args.directory)
        if test_file:
            print(f"\nRun: python {test_file}")

if __name__ == "__main__":
    main()