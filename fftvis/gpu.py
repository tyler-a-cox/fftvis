import itertools
import numpy as np
from pyuvdata import UVBeam
from pyuvsim import AnalyticBeam

from matvis import conversions, _uvbeam_to_raw

try:
    import cufinufft
    import cupy as cp
    from cupyx.scipy import interpolate, ndimage
    
    HAVE_CUDA = True
except ImportError:
    # if not installed, don't warn
    HAVE_CUDA = False
except Exception as e:
    print(e)
    HAVE_CUDA = False

def simulate_gpu(
    ants: dict,
    freqs: np.ndarray,
    fluxes: np.ndarray,
    beam: UVBeam | AnalyticBeam,
    crd_eq: np.ndarray,
    eq2tops: np.ndarray,
    baselines: list[tuple[int, int]] | None = None,
    precision: int = 2,
    polarized: bool = False,
    eps: float | None = None,
    beam_spline_opts: dict = None,
    flat_array_tol: float = 0.0,
):
    """
    GPU accelerated version of simulate_vis
    """
    if not HAVE_CUDA:
        raise ImportError("You need to install cufinufft and cupy to use this function")

    if precision == 1:
        real_dtype, complex_dtype = np.float32, np.complex64
    elif precision == 2:
        real_dtype, complex_dtype = np.float64, np.complex128
    else:
        raise ValueError("Precision must be 1 or 2")
    
    for eq2top in range(len(eq2tops)):
        # Convert sky coordinates to topocentric
        
        _uvbeam_to_raw.uvbeam_to_azza_grid(beam)

def do_beam_interpolation(
    *,
    beam: UVBeam | AnalyticBeam,
    tx: np.ndarray = None,
    ty: np.ndarray = None,
    polarized: bool = True,
    interpolator: str = 'map_coordinates',
    spline_opts: dict = {},
):

    if isinstance(beam, UVBeam):
        az_array, za_array = conversions.enu_to_az_za(
            enu_e=tx, enu_n=ty, orientation="uvbeam"
        )
        beam_grid = cp.asarray(beam.data_array)
        az_grid = cp.asarray(beam.axis1_array)
        za_grid = cp.asarray(beam.axis2_array)
        
        # Interpolate beam
        out_beam = _evaluate_beam_gpu(
            beam_grid=beam_grid,
            az_grid=az_grid,
            za_grid=za_grid,
            az_array=az_array,
            za_array=za_array,
            interpolator=interpolator,
            spline_opts=spline_opts,
        )
    else:
        nax, nfeed, nsrcs = 1, 1, 1
        A = np.zeros((nax, nfeed, nsrcs))

        _evaluate_beam_cpu(
            A,
            beam, 
            tx, 
            ty, 
            polarized, 
            interpolator=interpolator,
            spline_opts=spline_opts,
        )
        
        # Beam to GPU
        out_beam = cp.asarray(A)

    return out_beam

def _evaluate_beam_gpu(
    beam_grid: cp.ndarray,
    az_grid: cp.ndarray,
    za_grid: cp.ndarray,
    az_array: cp.ndarray,
    za_array: cp.ndarray,
    interpolator: str="map_coordinates",
    spline_opts: dict={}
):
    """
    Interpolate beam values from a regular az/za grid using GPU

    Parameters
    ----------
    beam_grid: cp.ndarray
    az_grid: cp.ndarray
    za_grid: cp.ndarray
    az_array: cp.ndarray
    za_array: cp.ndarray
    interpolator: str,
    spline_opts: {} 
    """
    assert interpolator in ["map_coordinates", "RegularGridInterpolator"], "Not recognized"

    if beam.dtype in (cp.dtype("float32"), cp.dtype("complex64")):
        real_dtype, complex_dtype = cp.dtype("float32"), cp.dtype("complex64")
    elif beam.dtype in (cp.dtype("float64"), cp.dtype("complex128")):
        real_dtype, complex_dtype = cp.dtype("float64"), cp.dtype("complex128")
    else:
        raise ValueError("HERE")

    # Get beam size
    nax, nfeed = beam_grid.shape[:2]
    out_beam = cp.zeros((nax, nfeed, za_array.size))

    if interpolator == "map_coordinates":
        x = (az_array - cp.min(az_grid)) / (cp.max(az_grid) - cp.min(az_grid))
        y = (za_array - cp.min(za_grid)) / (cp.max(za_grid) - cp.min(za_grid))

        for pol_i, pol_j in itertools.product(range(nax), range(nfeed)):
            ndimage.map_coordinates(
                beam_grid[pol_i, pol_j], 
                cp.array([x, y]), 
                output=out_beam[pol_i, pol_j], 
                **spline_opts
            )

    elif interpolator == "RegularGridInterpolator":
        raise NotImplemented("Not yet")

    return out_beam