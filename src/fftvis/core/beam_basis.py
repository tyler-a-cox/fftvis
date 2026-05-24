"""
Beam basis decomposition via SVD.

This module provides utilities for compressing a heterogeneous set of UVBeam
objects into a compact set of eigenbeams plus per-antenna scalar coefficients,
suitable for use with the basis-visibility path in CPUSimulationEngine.
"""

import logging
import numpy as np
from pyuvdata import UVBeam, BeamInterface
from matvis.core.beams import prepare_beam_unpolarized

logger = logging.getLogger(__name__)


def compute_beam_basis(
    beam_list,
    freq: float,
    polarized: bool,
    threshold: float = 1e-12,
    axis1_array=None,
    axis2_array=None,
    n_axis1: int = 361,
    n_axis2: int = 181,
):
    if len(beam_list) == 0:
        raise ValueError("beam_list must contain at least one beam.")
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold must be in the interval (0, 1].")

    freq_grid = np.atleast_1d(freq).astype(float)
    if freq_grid.size != 1:
        raise ValueError("compute_beam_basis currently expects a scalar freq.")

    beams = []
    for beam in beam_list:
        bi = BeamInterface(beam)

        if polarized:
            if bi.beam_type != "efield":
                raise ValueError("polarized=True requires efield beams.")
        else:
            bi = bi.as_power_beam(include_cross_pols=False)
            bi = prepare_beam_unpolarized(bi)

        beams.append(bi if isinstance(bi, BeamInterface) else BeamInterface(bi))

    if (axis1_array is None) != (axis2_array is None):
        raise ValueError("axis1_array and axis2_array must be supplied together.")

    if axis1_array is None:
        for bi in beams:
            b = bi.beam
            if (
                getattr(b, "pixel_coordinate_system", None) == "az_za"
                and getattr(b, "axis1_array", None) is not None
                and getattr(b, "axis2_array", None) is not None
            ):
                axis1_array = b.axis1_array
                axis2_array = b.axis2_array
                break
        else:
            axis1_array = np.linspace(0.0, 2.0 * np.pi, n_axis1)
            axis2_array = np.linspace(0.0, np.pi, n_axis2)

    axis1_array = np.asarray(axis1_array, dtype=float)
    axis2_array = np.asarray(axis2_array, dtype=float)

    interp_beams = []
    for bi in beams:
        if bi._isuvbeam:
            uvb = bi.beam.interp(
                az_array=axis1_array,
                za_array=axis2_array,
                freq_array=freq_grid,
                az_za_grid=True,
                new_object=True,
            )
        else:
            uvb = bi.beam.to_uvbeam(
                freq_array=freq_grid,
                beam_type=bi.beam_type,
                pixel_coordinate_system="az_za",
                axis1_array=axis1_array,
                axis2_array=axis2_array,
            )
        interp_beams.append(uvb)

    ref = interp_beams[0]
    slice_shape = ref.data_array[:, :, 0].shape

    slices = [b.data_array[:, :, 0] for b in interp_beams]
    for idx, data_slice in enumerate(slices):
        if data_slice.shape != slice_shape:
            raise ValueError(
                f"Beam {idx} evaluates to shape {data_slice.shape}, "
                f"expected {slice_shape}."
            )

    flat_vecs = np.stack([data_slice.ravel() for data_slice in slices], axis=0)

    U, s, Vh = np.linalg.svd(flat_vecs, full_matrices=False)
    
    s_norm = s / s[0]
    K = int(np.sum(s_norm >= threshold))

    beam_coefs = U[:, :K] * s[:K][None, :]

    eigenbeams = []
    for k in range(K):
        eb = ref.copy()
        eigenbeam_slice = Vh[k].reshape(slice_shape)
        eb.data_array = eigenbeam_slice[:, :, np.newaxis, ...]
        eigenbeams.append(eb)

    return eigenbeams, beam_coefs