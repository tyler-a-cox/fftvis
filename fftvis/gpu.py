import numpy as np
import cupy as cp
import cufinufft
from pyuvdata import UVBeam

def simulate_gpu(
    ants: dict,
    freqs: np.ndarray,
    fluxes: np.ndarray,
    beam,
    crd_eq: np.ndarray,
    eq2tops: np.ndarray,
    baselines: list[tuple[int, int]] | None = None,
    precision: int = 2,
    polarized: bool = False,
    eps: float | None = None,
    beam_spline_opts: dict = None,
    max_progress_reports: int = 100,
    live_progress: bool = True,
    flat_array_tol: float = 0.0,
):
    """
    GPU accelerated version of simulate_vis
    """
    pass

def do_beam_interpolation(
    beam: UVBeam,
    az: np.ndarray,
    za: np.ndarray,
    freq: float,
    polarized: bool,
    spline_opts: dict = None,
):
    """
    Perform beam interpolation, choosing between CPU and GPU implementations
    """
    pass

def _evaluate_beam_gpu(
    A_s: np.ndarray,
    beam: UVBeam,
    az: np.ndarray,
    za: np.ndarray,
    polarized: bool,
    freq: float,
    check: bool = False,
    spline_opts: dict = None,
):
    pass