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

def prepare_source_catalog(sky_model: np.ndarray, polarized_beam: bool) -> np.ndarray:
    """
    Prepare the source catalog for the given sky model. The function checks the shape of the sky model and
    converts it to a coherency matrix if necessary. It also returns a boolean indicating whether the sky model
    is polarized or not. 

    Parameters
    ----------
    sky_model : np.ndarray
        Stokes parameters of the sky model. If polarized is True, it should have shape (nsources, nfreqs, 4).
        If polarized is False, it should have shape (nsources, nfreqs).
    polarized : bool

    
    Returns
    -------
    coherency : np.ndarray
        Coherency matrix.
    is_sky_model_polarized : bool
        True if the sky model is polarized, False otherwise.
    """
    # Sky model should either be unpolarized or have shape (nsources, nfreqs, 4)
    if polarized_beam:
        if sky_model.ndim != 3 or sky_model.shape[-1] != 4:
            raise ValueError(
                f"polarized_beam=True requires sky_model.ndim==3 and "
                f"sky_model.shape[-1]==4, but got ndim={sky_model.ndim}, "
                f"shape={sky_model.shape}"
            )
    else:
        if sky_model.ndim != 2:
            raise ValueError(
                f"polarized_beam=False requires sky_model.ndim==2, "
                f"but got ndim={sky_model.ndim}, shape={sky_model.shape}"
            )
    
    # If the shape is (nsources, nfreqs), we assume it's unpolarized
    if coherency.ndim == 2:
        coherency = 0.5 * sky_model
        polarized_sky_model = False
    else:
        coherency = 0.5 * np.array(
            [
                [sky_model[..., 0] + sky_model[..., 3], sky_model[..., 1] + 1j * sky_model[..., 2]],
                [sky_model[..., 1] - 1j * sky_model[..., 2], sky_model[..., 0] - sky_model[..., 3]]
            ]
        )
        coherency = np.transpose(coherency, (2, 3, 0, 1))
        polarized_sky_model = True

    
    return coherency, polarized_sky_model