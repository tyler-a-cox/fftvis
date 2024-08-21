import itertools
import numpy as np
from pyuvdata import UVBeam
from pyuvsim import AnalyticBeam

from matvis import conversions, _uvbeam_to_raw
from ..core.beams import BeamInterpolator
from ..cpu.beams import CPUBeamEvaluator

try:
    import cupy as cp
    from cupyx.scipy import ndimage
    
    HAVE_CUDA = True
except ImportError:
    # if not installed, don't warn
    HAVE_CUDA = False
except Exception as e:
    print(e)
    HAVE_CUDA = False


class GPUBeamInterpolator(BeamInterpolator):
    """
    Interpolator class for beam evaluation on GPU

    
    """
    def setup(self):
        """
        Method
        """
        self.use_interp = isinstance(self.beam, UVBeam)

        if self.use_interp:
            d0, self.daz, self.dza = uvbeam_to_azza_grid(self.beam)
            naz = 2 * np.pi / self.daz + 1
            assert np.isclose(int(naz), naz), "Azimuthal grid not evenly spaced"

            self.beam_data = cp.asarray(
                d0, dtype=self.complex_dtype if self.polarized else self.real_dtype
            )
        else:
            self._eval = CPUBeamEvaluator.interp
            self._np_beam = np.zeros(
                (self.nax, self.nfeed, self.nsrcs), dtype=self.complex_dtype
            )

        self.interpolated_beam = cp.zeros(
            (self.nax, self.nfeed, self.nsrcs),
            dtype=self.complex_dtype,
        )

    def interp(self, tx: cp.ndarray, ty: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
        """
        Method
        """
        if self.use_interp:
            self._interp(
                tx,
                ty,
                out,
            )
        else:
            self._eval(
                tx.get(),
                ty.get(),
                self._np_beam,
            )
            out.set(self._np_beam)

    def _interp(self, tx: cp.ndarray, ty: cp.ndarray, out: cp.ndarray):
        """
        Method
        """
        # Interpolate beam
        az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")

        # TODO: This is probably wrong
        self.interpolated_beam[..., len(az) :] = 0.0

        # Interpolate beam
        out_beam = gpu_beam_interpolation(
            beam_grid=self.beam_data,
            az_grid=self.daz,
            za_grid=self.dza,
            az_array=az,
            za_array=za,
            beam_at_srcs=out,
            interpolator=self.interpolator,
            spline_opts=self.spline_opts,
        )

        out[:] = out_beam

def gpu_beam_interpolation(
    *,
    beam: cp.ndarray,
    az_grid: cp.ndarray,
    za_grid: cp.ndarray,
    az_array: cp.ndarray,
    za_array: cp.ndarray,
    out_beam: cp.ndarray,
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
    
    complex_beam = beam.dtype.name.startswith("complex")

    # Get beam size
    nax, nfeed = beam.shape[:2]
    out_beam = cp.zeros((nax, nfeed, za_array.size))

    if interpolator == "map_coordinates":
        x = (az_array - cp.min(az_grid)) / (cp.max(az_grid) - cp.min(az_grid))
        y = (za_array - cp.min(za_grid)) / (cp.max(za_grid) - cp.min(za_grid))

        for pol_i, pol_j in itertools.product(range(nax), range(nfeed)):
            ndimage.map_coordinates(
                beam[pol_i, pol_j], 
                cp.array([x, y]), 
                output=out_beam[pol_i, pol_j], 
                **spline_opts
            )

    elif interpolator == "RegularGridInterpolator":
        raise NotImplemented("Not yet")
    
    if not complex_beam:
        cp.sqrt(out_beam, out=out_beam)
        out_beam = out_beam.astype(complex_dtype)

    cp.cuda.Device().synchronize()
    return out_beam