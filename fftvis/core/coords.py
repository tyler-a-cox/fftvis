import numpy as np

try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False


def select_chunk():
    """
    Select
    """
    pass