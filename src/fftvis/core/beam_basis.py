"""
Beam basis decomposition via SVD.

This module provides utilities for compressing a heterogeneous set of UVBeam
objects into a compact set of eigenbeams plus per-antenna scalar coefficients,
suitable for use with the basis-visibility path in CPUSimulationEngine.
"""

import logging
import numpy as np
from pyuvdata import UVBeam

logger = logging.getLogger(__name__)


def compute_beam_basis(
    beam_list,
    freq: float,
    threshold: float = 1e-12,
):
    """
    """
    if len(beam_list) == 0:
        raise ValueError("beam_list must contain at least one beam.")

    n_beams = len(beam_list)
    freq_grid = np.atleast_1d(freq)

    interp_beams = []
    for idx, beam in enumerate(beam_list):
        interp_beams.append(
            beam.interp(freq_array=freq_grid, new_object=True)
        )

    ref = interp_beams[0]

    # Shape of the slice we care about: 
    # (Naxes_vec, Nfeeds, Nfreqs, Nax1) for healpix beams
    # (Naxes_vec, Nfeeds, Nfreqs, Nax1, Nax2) for gridded beams
    slice_shape = ref.data_array[:, :, 0].shape
    flat_vecs = np.array(
        [b.data_array[:, :, 0].ravel() for b in interp_beams]
    )  # (N_beams, Naxes_vec * Nfeeds * Nfreqs * Nax1 * Nax2)

    # ------------------------------------------------------------------
    # Step 3: SVD.
    # B = U @ diag(s) @ Vh  →  beam_i ≈ sum_k (U[i,k]*s[k]) * Vh[k]
    # ------------------------------------------------------------------
    U, s, Vh = np.linalg.svd(flat_vecs, full_matrices=False)

    # ------------------------------------------------------------------
    # Step 4: Truncate at normalised singular value threshold.
    # ------------------------------------------------------------------
    s_norm = s / s[0]
    K = int(np.sum(s_norm >= threshold))

    U_k  = U[:, :K]   # (N_beams, K)
    s_k  = s[:K]       # (K,)
    Vh_k = Vh[:K, :]  # (K, N_flat)

    # ------------------------------------------------------------------
    # Step 5: Compute per-antenna coefficients.
    # beam_coefs[i, k] = U[i, k] * s[k]  so that  flat_vecs ≈ beam_coefs @ Vh_k
    # ------------------------------------------------------------------
    beam_coefs = U_k * s_k[None, :]  # (N_beams, K)
    
    # ------------------------------------------------------------------
    # Step 6: Build eigenbeam UVBeam objects by copying reference metadata
    # and replacing data_array with the reshaped Vh rows.
    # ------------------------------------------------------------------
    eigenbeams = []
    for k in range(K):
        eb = ref.copy()
        eigenbeam_slice = Vh_k[k].reshape(slice_shape)
        eb.data_array = eigenbeam_slice[:, :, np.newaxis]
        eigenbeams.append(eb)

    return eigenbeams, beam_coefs
