import pytest
import numpy as np
from fftvis import utils


def test_get_plane_to_xy_rotation_matrix():
    """ """
    # Rotate the array to the xy-plane
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    z = 0.125 * x + 0.5 * y
    antvecs = np.array([x, y, z]).T

    rotation_matrix = utils.get_plane_to_xy_rotation_matrix(antvecs)
    rm_antvecs = np.dot(rotation_matrix.T, antvecs.T)

    # Check that all elements of the z-axis are close to zero
    np.testing.assert_allclose(rm_antvecs[-1], 0, atol=1e-12)

    # Check that the lengths of the vectors are preserved
    np.testing.assert_allclose(
        np.linalg.norm(antvecs, axis=1), np.linalg.norm(rm_antvecs, axis=0), atol=1e-12
    )


def test_get_plane_to_xy_rotation_matrix_errors():
    """ """
    # Rotate the array to the xy-plane
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    z = 0.125 * x + 0.5 * y
    antvecs = np.array([x, y, z]).T

    # Check that method is robust to errors
    rng = np.random.default_rng(42)
    random_antvecs = antvecs.copy()
    random_antvecs[:, -1] += rng.standard_normal(100)

    # Rotate the array to the xy-plane
    rotation_matrix = utils.get_plane_to_xy_rotation_matrix(random_antvecs)
    rm_antvecs = np.dot(rotation_matrix.T, antvecs.T)

    # Check that all elements of the z-axis within 5-sigma of zero
    np.testing.assert_array_less(np.abs(rm_antvecs[-1]), 5)
