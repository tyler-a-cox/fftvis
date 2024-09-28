try:
    import cupy

    HAVE_GPU = True
except ImportError:
    HAVE_GPU = False

from . import cpu
from .wrapper import simulate_vis
from . import beams, utils, simulate

if HAVE_GPU:
    from . import gpu