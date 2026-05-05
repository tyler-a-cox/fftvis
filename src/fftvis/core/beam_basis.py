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
    beam_list: list[UVBeam],
    freq: float,
    threshold: float = 1e-12,
    polarized: bool = False,
) -> tuple[list[UVBeam], np.ndarray]:
    """Decompose a set of UVBeams into an SVD eigenbeam basis.

    Each beam is first interpolated onto a common frequency grid to keep memory
    usage predictable, then stacked into a matrix and SVD'd.  The basis is
    truncated at the first component whose normalised singular value drops below
    ``threshold``.

    The decomposition satisfies::

        beam_i(az, za, freq) ≈ sum_k  coeffs[i, k] * eigenbeams[k](az, za, freq)

    so ``eigenbeams`` can be passed directly as ``beam_list`` to
    ``CPUSimulationEngine.simulate`` and ``coeffs`` as ``beam_coeffs``.

    Parameters
    ----------
    beam_list : list of UVBeam
        Input beams to decompose.  All beams must share the same spatial grid
        (``Nax1``, ``Nax2``) and ``beam_type`` (``efield`` or ``power``).
    freq : float
        Frequency in Hz at which every beam is evaluated before the SVD.
    threshold : float, optional
        Normalised singular value cutoff.  Basis components ``k`` with
        ``s[k] / s[0] < threshold`` are discarded.  Default ``1e-3``.
    polarized : bool, optional
        If ``True``, the full Jones matrix (all ``Naxes_vec`` and ``Nfeeds``
        components) is included in the SVD.  If ``False``, only the first
        component ``data_array[0, 0, 0, ...]`` is used, matching the scalar
        beam path in the simulator.  Default ``False``.

    Returns
    -------
    eigenbeams : list of UVBeam
        The ``K`` truncated eigenbeam objects.  Each has the same metadata and
        spatial/frequency grid as the interpolated input beams.
    coeffs : np.ndarray
        Per-antenna coefficients of shape ``(N_beams, K)``.  These are the
        values to pass as ``beam_coeffs`` to the simulator.

    Notes
    -----
    The function allocates two large intermediate arrays simultaneously: the
    full stacked beam matrix ``B`` of shape ``(N_beams, N_flat)`` and the ``Vh``
    factor of the same shape.  Peak memory is therefore roughly
    ``2 * N_beams * N_flat * itemsize``.  Choosing a sparse ``freq_grid``
    directly reduces ``N_flat``.

    Only ``efield`` beams have been tested; ``power`` beams should work but
    the resulting eigenbeams will have real-valued ``data_array`` only if all
    inputs are real.
    """
    if len(beam_list) == 0:
        raise ValueError("beam_list must contain at least one beam.")

    n_beams = len(beam_list)
    freq_grid = np.atleast_1d(freq)

    # ------------------------------------------------------------------
    # Step 1: Interpolate every beam to the common frequency grid.
    # ------------------------------------------------------------------
    logger.info(
        f"Interpolating {n_beams} beams to {len(freq_grid)} frequencies "
        f"({freq_grid[0]/1e6:.1f}–{freq_grid[-1]/1e6:.1f} MHz)."
    )
    interp_beams = []
    for idx, beam in enumerate(beam_list):
        interp_beams.append(
            beam.interp(freq_array=freq_grid, new_object=True)
        )
        logger.debug(f"  Interpolated beam {idx + 1}/{n_beams}.")

    # ------------------------------------------------------------------
    # Step 2: Extract the relevant slice of data_array and flatten.
    #
    # data_array shape: (Naxes_vec, 1, Nfeeds, Nfreqs, Nax1, Nax2)
    #
    # Polarized  → use axes_vec and feeds: data_array[:, 0, :, ...]
    #              flat shape per beam: Naxes_vec * Nfeeds * Nfreqs * Nax1 * Nax2
    # Unpolarized → match evaluate_beam scalar path: data_array[0, 0, 0, ...]
    #              flat shape per beam: Nfreqs * Nax1 * Nax2
    # ------------------------------------------------------------------
    ref = interp_beams[0]

    if polarized:
        # Shape of the slice we care about: (Naxes_vec, Nfeeds, Nfreqs, Nax1, Nax2)
        slice_shape = ref.data_array[:, 0, :, :, :, :].shape
        flat_vecs = np.array(
            [b.data_array[:, 0, :, :, :, :].ravel() for b in interp_beams]
        )  # (N_beams, Naxes_vec * Nfeeds * Nfreqs * Nax1 * Nax2)
    else:
        # Shape: (Nfreqs, Nax1, Nax2)
        slice_shape = ref.data_array[0, 0, 0, :, :, :].shape
        flat_vecs = np.array(
            [b.data_array[0, 0, 0, :, :, :].ravel() for b in interp_beams]
        )  # (N_beams, Nfreqs * Nax1 * Nax2)

    logger.info(
        f"Beam matrix shape: {flat_vecs.shape}  "
        f"({flat_vecs.nbytes / 1024**2:.1f} MB)."
    )

    # ------------------------------------------------------------------
    # Step 3: SVD.
    # B = U @ diag(s) @ Vh  →  beam_i ≈ sum_k (U[i,k]*s[k]) * Vh[k]
    # ------------------------------------------------------------------
    logger.info("Computing SVD...")
    U, s, Vh = np.linalg.svd(flat_vecs, full_matrices=False)

    # ------------------------------------------------------------------
    # Step 4: Truncate at normalised singular value threshold.
    # ------------------------------------------------------------------
    s_norm = s / s[0]
    K = int(np.sum(s_norm >= threshold))
    logger.info(
        f"Retaining {K}/{len(s)} basis components "
        f"(threshold={threshold}, "
        f"min retained s_norm={s_norm[K-1]:.3e})."
    )

    U_k  = U[:, :K]   # (N_beams, K)
    s_k  = s[:K]       # (K,)
    Vh_k = Vh[:K, :]  # (K, N_flat)

    # ------------------------------------------------------------------
    # Step 5: Compute per-antenna coefficients.
    # coeffs[i, k] = U[i, k] * s[k]  so that  flat_vecs ≈ coeffs @ Vh_k
    # ------------------------------------------------------------------
    coeffs = U_k * s_k[None, :]  # (N_beams, K)

    # ------------------------------------------------------------------
    # Step 6: Build eigenbeam UVBeam objects by copying reference metadata
    # and replacing data_array with the reshaped Vh rows.
    # ------------------------------------------------------------------
    eigenbeams = []
    for k in range(K):
        eb = ref.copy()
        eigenbeam_slice = Vh_k[k].reshape(slice_shape)

        if polarized:
            # Restore the dropped size-1 axis: (Naxes_vec, 1, Nfeeds, Nfreqs, Nax1, Nax2)
            eb.data_array = eigenbeam_slice[:, np.newaxis, :, :, :, :]
        else:
            # Restore to full data_array shape; fill non-primary components with zeros.
            eb.data_array = np.zeros_like(ref.data_array)
            eb.data_array[0, 0, 0, :, :, :] = eigenbeam_slice

        eigenbeams.append(eb)

    logger.info(f"Basis computation complete: {K} eigenbeams.")
    return eigenbeams, coeffs
