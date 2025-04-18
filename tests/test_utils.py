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


def test_get_pos_reds():
    """ """
    antpos = {
        ant_index: np.array([ant_index * 10.0, 0.0, 0.0]) for ant_index in range(10)
    }
    reds = utils.get_pos_reds(antpos)

    for red in reds:
        ai, aj = red[0]
        blmag = np.linalg.norm(antpos[ai] - antpos[aj])
        for ai, aj in red:
            assert np.isclose(blmag, np.linalg.norm(antpos[ai] - antpos[aj]))

    # Check that a non-redundant array returns list of single element lists
    rng = np.random.default_rng(seed=42)
    antpos = {
        ant_index: np.array([rng.uniform(-100, 100), rng.uniform(-100, 100), 0])
        for ant_index in range(10)
    }
    reds = utils.get_pos_reds(antpos, include_autos=False)

    for red in reds:
        assert len(red) == 1


def test_get_task_chunks():
    """ """
    ntimes, nfreqs, nprocesses = 2, 3, 10

    # Test when 2 * nproc > ntasks
    nproc, fslice, tslice, nfreq_chunks, ntime_chunks = utils.get_task_chunks(
        nprocesses=nprocesses, ntimes=ntimes, nfreqs=nfreqs
    )

    # Check sizes
    assert nproc == 1
    assert ntimes == ntime_chunks
    assert nfreqs == nfreq_chunks

    # Test when 2 * nproc < ntasks
    nproc, fslice, tslice, nfreq_chunks, ntime_chunks = utils.get_task_chunks(
        nprocesses=nprocesses, ntimes=ntimes * 4, nfreqs=nfreqs
    )

    # Check sizes
    assert nproc == nprocesses
    assert len(fslice) == nproc
    assert len(tslice) == nproc

    # All frequency slices should be the same
    for fslc in fslice[1:]:
        assert fslc == fslice[0]
