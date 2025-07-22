#!/usr/bin/env python3
"""
AUTO-GENERATED Installable Dependencies Test Script
(Enhanced – auto-install missing packages via pip, CUDA-aware for torch-family)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import os, sys, subprocess, pkg_resources, re, importlib.util

pypi_map = {'cv2': 'opencv-python', 'PIL': 'Pillow', 'sklearn': 'scikit-learn', 'yaml': 'PyYAML', 'serial': 'pyserial', 'whisper': 'openai-whisper', 'TTS': 'TTS', 'lws': 'lws-python', 'insightface': 'insightface', 'dateutil': 'python-dateutil', 'requests': 'requests', 'matplotlib': 'matplotlib', 'transformers': 'transformers', 'tokenizers': 'tokenizers', 'torch': 'torch', 'torchvision': 'torchvision', 'torchaudio': 'torchaudio', 'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'gradio': 'gradio', 'diffusers': 'diffusers', 'einops': 'einops', 'librosa': 'librosa', 'pydub': 'pydub', 'imageio': 'imageio', 'soundfile': 'soundfile', 'decord': 'decord', 'mediapipe': 'mediapipe', 'spacy': 'spacy', 'nltk': 'nltk', 'regex': 'regex', 'num2words': 'num2words', 'omegaconf': 'omegaconf', 'accelerate': 'accelerate', 'more_itertools': 'more-itertools', 'packaging': 'packaging', 'scenedetect': 'scenedetect', 'python_speech_features': 'python_speech_features', 'pypinyin': 'pypinyin', 'cutlet': 'cutlet', 'hangul_romanize': 'hangul-romanize', 'cog': 'cog'}
def pypi(mod): return pypi_map.get(mod, mod)

def installed(mod):
    try:
        pkg_resources.get_distribution(pypi(mod))
        return True
    except Exception:
        return importlib.util.find_spec(mod) is not None

def torch_cuda_info():
    """Return (cuda_ver, cudnn_ver) or (None, None) if torch not present."""
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
    return ["--extra-index-url", f"https://download.pytorch.org/whl/cu{major}{minor}"]

def pip_install(pkg, *flags):
    cmd = [sys.executable, "-m", "pip", "install", *flags, pkg]
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode == 0

cuda_ver, cudnn_ver = torch_cuda_info()
torch_flags = build_torch_flags(cuda_ver)

missing = [m for m in ['DeepCache', 'PIL', 'TTS', 'accelerate', 'affine_transform', 'attention', 'audio', 'cog', 'cutlet', 'cv2', 'decoding', 'decord', 'diffusers', 'einops', 'english', 'eval', 'face_detector', 'ffmpeg', 'gradio', 'hangul_romanize', 'imageio', 'insightface', 'kornia', 'librosa', 'lpips', 'lws', 'matplotlib', 'mediapipe', 'model', 'models', 'more_itertools', 'nets', 'nltk', 'num2words', 'numpy', 'omegaconf', 'packaging', 'preprocess', 'pydub', 'pypinyin', 'python_speech_features', 'regex', 'requests', 'resnet', 'scenedetect', 'scipy', 'soundfile', 'spacy', 'tokenizers', 'torch', 'torchvision', 'tqdm', 'trainer', 'transcribe', 'transformers'] if not installed(m)]

if missing:
    print("Missing packages:", ", ".join(missing))
    if input("Install them now? [y/N] ").strip().lower() == "y":
        for m in missing:
            pkg = pypi(m)
            if m in ("torch", "torchvision", "torchaudio") and cuda_ver:
                cuda_suffix = f"+cu{''.join(cuda_ver.split('.')[:2])}"
                pkg = f"{pkg}{cuda_suffix}"
            pip_install(pkg, *torch_flags)

print("\n=== FINAL STATUS CHECK ===")
for m in sorted(['DeepCache', 'PIL', 'TTS', 'accelerate', 'affine_transform', 'attention', 'audio', 'cog', 'cutlet', 'cv2', 'decoding', 'decord', 'diffusers', 'einops', 'english', 'eval', 'face_detector', 'ffmpeg', 'gradio', 'hangul_romanize', 'imageio', 'insightface', 'kornia', 'librosa', 'lpips', 'lws', 'matplotlib', 'mediapipe', 'model', 'models', 'more_itertools', 'nets', 'nltk', 'num2words', 'numpy', 'omegaconf', 'packaging', 'preprocess', 'pydub', 'pypinyin', 'python_speech_features', 'regex', 'requests', 'resnet', 'scenedetect', 'scipy', 'soundfile', 'spacy', 'tokenizers', 'torch', 'torchvision', 'tqdm', 'trainer', 'transcribe', 'transformers']):
    try:
        __import__(m)
        print(f"{m:<25} ✅ OK")
    except Exception as e:
        print(f"{m:<25} ❌ {e}")
