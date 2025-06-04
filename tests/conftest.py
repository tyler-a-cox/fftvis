"""Global configuration for pytest in fftvis tests."""

import pytest

# Check GPU availability once at the start of the test session
def check_gpu_available():
    """Check if GPU implementation can be used."""
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except ImportError:
        return False


# Store GPU availability as a session-scoped variable
GPU_AVAILABLE = check_gpu_available()

# Create a marker for GPU tests
gpu_test = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="GPU not available (cupy not installed or no CUDA device)"
)

# Check for optional dependencies
def check_tabulate_available():
    """Check if tabulate is installed."""
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


TABULATE_AVAILABLE = check_tabulate_available()