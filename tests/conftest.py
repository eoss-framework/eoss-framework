"""Pytest configuration for the EoSS framework tests.

Unit tests run without a GPU. Tests marked ``@pytest.mark.gpu`` require a
CUDA-capable GPU and are skipped automatically when none is present.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# The repository root IS the `eoss` package directory, so the parent
# directory must be importable for `import eoss` to work.
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _gpu_available() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()
NVCC_AVAILABLE = shutil.which("nvcc") is not None

requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="no CUDA GPU available")
requires_nvcc = pytest.mark.skipif(not NVCC_AVAILABLE, reason="nvcc not available")


def pytest_collection_modifyitems(config, items):
    """Auto-skip gpu-marked tests when no GPU is present."""
    if GPU_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="no CUDA GPU available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
