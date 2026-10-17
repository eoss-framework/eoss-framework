"""GPU discovery helpers (single source of truth).

Previously duplicated across Makefiles, run_campaign.py (SM detection) and
the Contention scripts (SM counts with inconsistent 84 vs 170 fallbacks).
"""

import shutil
import subprocess
from functools import lru_cache


def _nvidia_smi() -> str:
    path = shutil.which("nvidia-smi")
    if path is None:
        raise RuntimeError("nvidia-smi not found on PATH")
    return path


@lru_cache(maxsize=None)
def detect_sm(device: int = 0) -> str:
    """Detect the SM (compute capability) version of a GPU, e.g. "120".

    Raises RuntimeError if detection fails.
    """
    try:
        result = subprocess.run(
            [_nvidia_smi(), f"--id={device}",
             "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise RuntimeError(f"Failed to detect SM version: {e}") from e
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return result.stdout.strip().splitlines()[0].strip().replace(".", "")


def _cuda_device_attribute(attr: int, device: int = 0) -> int:
    """Query a CUDA driver device attribute via libcuda (ctypes)."""
    import ctypes
    for lib_name in ("libcuda.so.1", "libcuda.so", "libcuda.dylib",
                     "nvcuda.dll"):
        try:
            lib = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("libcuda not found")
    res = lib.cuInit(0)
    if res != 0:
        raise RuntimeError(f"cuInit failed: {res}")
    dev = ctypes.c_int()
    if lib.cuDeviceGet(ctypes.byref(dev), device) != 0:
        raise RuntimeError(f"cuDeviceGet failed for device {device}")
    val = ctypes.c_int()
    if lib.cuDeviceGetAttribute(ctypes.byref(val), attr, dev) != 0:
        raise RuntimeError(f"cuDeviceGetAttribute({attr}) failed")
    return val.value


# CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16


@lru_cache(maxsize=None)
def sm_count(device: int = 0) -> int:
    """Number of streaming multiprocessors on the GPU.

    Uses the CUDA driver API (libcuda). Raises RuntimeError on failure
    (no silent wrong fallback).
    """
    return _cuda_device_attribute(_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                  device)


@lru_cache(maxsize=None)
def gpu_name(device: int = 0) -> str:
    """Human-readable GPU name, e.g. "NVIDIA GeForce RTX 5090"."""
    try:
        result = subprocess.run(
            [_nvidia_smi(), f"--id={device}",
             "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise RuntimeError(f"Failed to query GPU name: {e}") from e
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return result.stdout.strip().splitlines()[0].strip()
