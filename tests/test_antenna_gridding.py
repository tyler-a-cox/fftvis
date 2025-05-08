import pytest
import numpy as np
from fftvis.core.antenna_gridding import check_antpos_griddability

def test_check_array_griddability():
    """Test that check_array_griddability works as expected."""
    # Create a simple linear array
    antpos = {i: np.array([i * 1.5, 0.0, 0.0]) for i in range(10)}

    # Check griddability
    is_griddable, modified_antpos, _ = check_antpos_griddability(antpos)
    assert is_griddable

    for i in range(10):
        assert np.allclose(
            modified_antpos[i], 
            np.round(modified_antpos[i]).astype(int)
        ), f"Ant {i} position not griddable: {modified_antpos[i]}"

    # Create a 5x5 grid of antennas
    antpos = {}
    ci = 0
    for i in range(5):
        for j in range(5):
            antpos[ci] = np.array([i, j, 0])
            ci += 1

    is_griddable, modified_antpos, _ = check_antpos_griddability(antpos)
    assert is_griddable
    for i in range(25):
        assert np.allclose(
            modified_antpos[i], 
            np.round(modified_antpos[i]).astype(int)
        ), f"Ant {i} position not griddable: {modified_antpos[i]}"

    # Remove some antennas, but keep the grid
    del antpos[3]
    del antpos[16]
    is_griddable, modified_antpos, _ = check_antpos_griddability(antpos)
    assert is_griddable

    # Create a non-griddable array
    antpos = {i: np.array([np.random.uniform(0, 10), 0, 0]) for i in range(10)}
    is_griddable, modified_antpos, _ = check_antpos_griddability(antpos,)
    assert not is_griddable

    # Check hexagonal grid
    antpos = {
        0: np.array([-0.5,  np.sqrt(3) / 2,  0.]),
        1: np.array([0.5, np.sqrt(3) / 2, 0.]),
        2: np.array([-1.,  0.,  0.]),
        3: np.array([0., 0., 0.]),
        4: np.array([1., 0., 0.]),
        5: np.array([-0.5, -np.sqrt(3) / 2,  0.]),
        6: np.array([ 0.5, -np.sqrt(3) / 2,  0.]),
    }
    is_griddable, modified_antpos, _ = check_antpos_griddability(antpos,)
    assert is_griddable