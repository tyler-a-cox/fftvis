import numpy as np

try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False


def compute_apparent_coherency(beam, coherency_matrix):
    """
    Compute the apparent coherency of the sky model given the beam.

    Parameters
    ----------
    beam : array-like
        Array of shape (nfeed, nfeed, ...) containing the beam.
    coherency_matrix : array-like
        Array of shape (nfeed, nfeed, ...) containing the sky model.
    """
    if coherency_matrix.ndim != beam.ndim and coherency_matrix.shape != beam.shape[2:]:
        raise ValueError(
            "The sky model and beam must have the same shape or the same number of sources/frequencies."
        )

    xp = cp if HAVE_CUDA else np

    # Compute the apparent coherency
    if coherency_matrix.ndim == 1:
        apparent_coherency = xp.einsum(
            "ijs,s,jks->ijs", beam, coherency_matrix, beam.conj()
        )
    elif coherency_matrix.ndim > 2:
        apparent_coherency = xp.einsum(
            "ij...,jk...lk...->il...", beam, coherency_matrix, beam.conj()
        )

    return apparent_coherency
