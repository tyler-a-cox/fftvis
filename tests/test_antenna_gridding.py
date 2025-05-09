import numpy as np
import pytest

from fftvis.core.antenna_gridding import check_antpos_griddability


# ------------------------------------------------------------------
# helpers / fixtures
# ------------------------------------------------------------------
def _linear_array(n=10, spacing=1.5):
    return {i: np.array([i * spacing, 0.0, 0.0]) for i in range(n)}

def _square_grid(n_side=5):
    """Full n×n Cartesian grid of antennas in the xy‑plane."""
    antpos = {}
    idx = 0
    for i in range(n_side):
        for j in range(n_side):
            antpos[idx] = np.array([i, j, 0.0])
            idx += 1
    return antpos

def _square_grid_with_holes():
    antpos = _square_grid()
    for k in (3, 16):        # remove a couple of elements
        antpos.pop(k)
    return antpos

def _scattered_xy(n=15, seed=42):
    """
    Deterministic pseudo‑random scatter in the xy‑plane.
    No single linear transform can map these points onto
    an integer lattice within sane tolerance.
    """
    rng = np.random.default_rng(seed)
    return {
        i: np.array([rng.uniform(0, 10), rng.uniform(0, 10), 0.0])
        for i in range(n)
    }

def _hex_grid():
    return {
        0: np.array([-0.5,  np.sqrt(3) / 2,  0.0]),
        1: np.array([ 0.5,  np.sqrt(3) / 2,  0.0]),
        2: np.array([-1.0,  0.0,             0.0]),
        3: np.array([ 0.0,  0.0,             0.0]),
        4: np.array([ 1.0,  0.0,             0.0]),
        5: np.array([-0.5, -np.sqrt(3) / 2,  0.0]),
        6: np.array([ 0.5, -np.sqrt(3) / 2,  0.0]),
    }


# ------------------------------------------------------------------
# parametrised test
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "antpos, expected_griddable",
    [
        pytest.param(_linear_array(), True, id="linear"),
        pytest.param(_square_grid(), True, id="square-full"),
        pytest.param(_square_grid_with_holes(), True, id="square-holey"),
        pytest.param(_hex_grid(), True, id="hex-grid"),
        pytest.param(_scattered_xy(), False, id="non‑griddable"),
    ],
)
def test_check_antpos_griddability(antpos, expected_griddable):
    """All griddability scenarios in one parametrised test."""
    is_griddable, modified_antpos, _ = check_antpos_griddability(antpos)

    # 1) overall flag matches expectation
    assert is_griddable is expected_griddable

    # 2) if griddable, every coordinate should round cleanly to ints
    if expected_griddable:
        for ant_id, pos in modified_antpos.items():
            msg = f"Antenna {ant_id} not on integer grid: {pos}"
            assert np.allclose(pos, np.round(pos).astype(int)), msg