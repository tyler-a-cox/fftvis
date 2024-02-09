from . import cpu
from . import gpu
import numpy as np
from typing import Callable
from matvis import conversions


def simulate_vis(
    antpos: dict,
    sources: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    lsts: np.ndarray,
    beam: Callable,
    precision: int = 1,
    polarized: bool = False,
    latitude: float = -0.5361913261514378,
    use_redundancy: bool = True,
):
    """
    antpos
    freqs"""
    # Source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Get coordinate transforms as a function of LST
    eq2tops = np.array([conversions.eci_to_enu_matrix(lst, latitude) for lst in lsts])

    return cpu.simulate_cpu(
        antpos=antpos,
        freqs=freqs,
        sources=sources,
        beam=beam,
        crd_eq=crd_eq,
        eq2tops=eq2tops,
        precision=precision,
        polarized=polarized,
        use_redundancy=use_redundancy,
    )
