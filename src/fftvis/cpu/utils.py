import numpy as np
import numba as nb


@nb.jit(nopython=True)
def inplace_rot(rot: np.ndarray, b: np.ndarray):  # pragma: no cover
    """
    CPU implementation of in-place rotation of coordinates using Numba.

    Parameters:
    ----------
    rot : np.ndarray
        3x3 rotation matrix
    b : np.ndarray
        Array of shape (3, n) containing coordinates to rotate
    """
    nsrc = b.shape[1]
    out = np.zeros(3, dtype=b.dtype)

    for n in range(nsrc):
        out[0] = rot[0, 0] * b[0, n] + rot[0, 1] * b[1, n] + rot[0, 2] * b[2, n]
        out[1] = rot[1, 0] * b[0, n] + rot[1, 1] * b[1, n] + rot[1, 2] * b[2, n]
        out[2] = rot[2, 0] * b[0, n] + rot[2, 1] * b[1, n] + rot[2, 2] * b[2, n]
        b[:, n] = out

def prepare_source_catalog(sky_model: np.ndarray, polarized_beam: bool) -> tuple[np.ndarray, bool]:
    """
    Prepare the source catalog for the given sky model by building its coherency matrix.

    Parameters
    ----------
    sky_model : np.ndarray
        - Unpolarized: shape (nsources, nfreqs)
        - Polarized:   shape (nsources, nfreqs, 4)
    polarized_beam : bool
        If True, you may pass either a 2D unpolarized sky_model (nsources, nfreqs) or
        a 3D polarized array (nsources, nfreqs, Nstokes) with last axis length 4.
        If False, only 2D unpolarized sky_models are allowed.

    Returns
    -------
    coherency : np.ndarray
        - If input was 2D: shape (nsources, nfreqs) (unpolarized)
        - If input was 3D: shape (nsources, nfreqs, 2, 2) (full coherency matrix)
    is_sky_model_polarized : bool
        True if you passed a 3D, 4pol cube; False otherwise.
    """
    # 1) Shape validation
    if sky_model.ndim == 2:
        # always OK
        polarized_sky_model = False
    elif polarized_beam and sky_model.ndim == 3 and sky_model.shape[-1] == 4:
        polarized_sky_model = True
    else:
        if polarized_beam:
            raise ValueError(
                f"polarized_beam=True requires sky_model to be either:\n"
                f"  2D unpolarized, or\n"
                f"  3D with last axis of length 4; "
                f"got ndim={sky_model.ndim}, shape={sky_model.shape}"
            )
        else:
            raise ValueError(
                f"polarized_beam=False requires sky_model to be 2D; "
                f"got ndim={sky_model.ndim}, shape={sky_model.shape}"
            )
    
    # If the shape is (nsources, nfreqs), we assume it's unpolarized
    if not polarized_sky_model:
        coherency = 0.5 * sky_model
    else:
        coherency = 0.5 * np.array(
            [
                [sky_model[..., 0] + sky_model[..., 3], sky_model[..., 1] + 1j * sky_model[..., 2]],
                [sky_model[..., 1] - 1j * sky_model[..., 2], sky_model[..., 0] - sky_model[..., 3]]
            ]
        )
        coherency = np.transpose(coherency, (2, 3, 0, 1))

    
    return coherency, polarized_sky_model