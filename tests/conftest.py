"""Global configuration for pytest in fftvis tests."""

import pytest

from fftvis import GPU_AVAILABLE

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