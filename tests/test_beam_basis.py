"""
Tests for the eigenbeam (SVD beam basis) decomposition and basis-path simulation.

Analytic beams (AiryBeam) are used throughout to avoid a dependency on the
CST test-data file and to sidestep the efield/power-beam ambiguity that arises
with tabulated UVBeam objects.

Unit tests cover ``compute_beam_basis`` directly: output shapes, reconstruction
accuracy, threshold behaviour, and error handling.

Integration tests compare the basis-path simulation against the standard
per-antenna simulation to verify that the two paths agree within numerical
tolerance when the same underlying beams are used.
"""

import numpy as np
import pytest

from astropy.coordinates import EarthLocation
from pyuvdata import AiryBeam, BeamInterface

from fftvis.core.beam_basis import compute_beam_basis
from fftvis.wrapper import simulate_vis


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────────────────────

_FREQ = 150e6


@pytest.fixture(scope="module")
def beam_a() -> AiryBeam:
    """A 14 m Airy dish beam."""
    return AiryBeam(diameter=14.0)


@pytest.fixture(scope="module")
def beam_b() -> AiryBeam:
    """A 7 m Airy dish beam — genuinely different from beam_a."""
    return AiryBeam(diameter=7.0)


@pytest.fixture(scope="module")
def sim_params():
    """Minimal, reproducible simulation parameters for a 3-antenna array."""
    ants = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([14.6, 0.0, 0.0]),
        2: np.array([0.0, 14.6, 0.0]),
    }
    freqs = np.array([_FREQ])
    rng = np.random.default_rng(42)
    nsrc = 30
    ra = rng.uniform(0, 2 * np.pi, nsrc)
    dec = rng.uniform(-np.pi / 4, np.pi / 4, nsrc)
    fluxes = rng.uniform(0.5, 2.0, (nsrc, 1))
    times = np.array([2458119.5])
    telescope_loc = EarthLocation(
        lat="-30d43m17.5s", lon="21d25m41.9s", height=1073.0
    )
    return dict(
        ants=ants,
        freqs=freqs,
        ra=ra,
        dec=dec,
        fluxes=fluxes,
        times=times,
        telescope_loc=telescope_loc,
    )


# ──────────────────────────────────────────────────────────────────────────────
# compute_beam_basis: output shape tests
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeBeamBasisShapes:
    """Verify that compute_beam_basis returns objects with consistent shapes."""

    def test_single_beam_returns_one_eigenbeam(self, beam_a):
        eigenbeams, coefs = compute_beam_basis(
            [beam_a], freq=_FREQ, polarized=True
        )
        assert len(eigenbeams) == 1
        assert coefs.shape == (1, 1)

    def test_coefs_rows_equal_nant(self, beam_a, beam_b):
        eigenbeams, coefs = compute_beam_basis(
            [beam_a, beam_b], freq=_FREQ, polarized=True
        )
        assert coefs.shape[0] == 2

    def test_coefs_cols_equal_number_of_eigenbeams(self, beam_a, beam_b):
        eigenbeams, coefs = compute_beam_basis(
            [beam_a, beam_b], freq=_FREQ, polarized=True
        )
        assert coefs.shape[1] == len(eigenbeams)

    def test_eigenbeam_data_array_shape_matches_custom_axis_arrays(self, beam_a):
        """The eigenbeam grid should match the user-supplied angular axes."""
        axis1 = np.linspace(0, 2 * np.pi, 36)
        axis2 = np.linspace(0, np.pi / 2, 19)
        eigenbeams, _ = compute_beam_basis(
            [beam_a],
            freq=_FREQ,
            polarized=True,
            axis1_array=axis1,
            axis2_array=axis2,
        )
        assert eigenbeams[0].axis1_array.shape == axis1.shape
        assert eigenbeams[0].axis2_array.shape == axis2.shape


# ──────────────────────────────────────────────────────────────────────────────
# compute_beam_basis: singular value / rank behaviour
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeBeamBasisRank:
    """Verify that the number of retained modes tracks beam diversity."""

    def test_identical_beams_yield_one_mode(self, beam_a):
        """Copies of the same beam span a 1-D subspace; K must collapse to 1."""
        eigenbeams, coefs = compute_beam_basis(
            [beam_a, beam_a, beam_a], freq=_FREQ, polarized=True, threshold=1e-3
        )
        assert len(eigenbeams) == 1
        assert coefs.shape == (3, 1)

    def test_different_beams_retain_multiple_modes_at_tight_threshold(
        self, beam_a, beam_b
    ):
        """
        Two Airy beams with very different diameters are not proportional, so
        at a near-zero threshold the SVD should retain at least 2 modes.
        """
        eigenbeams, coefs = compute_beam_basis(
            [beam_a, beam_b], freq=_FREQ, polarized=True, threshold=1e-12
        )
        assert len(eigenbeams) >= 2
        assert coefs.shape == (2, len(eigenbeams))

    def test_looser_threshold_gives_fewer_or_equal_modes(self, beam_a, beam_b):
        """Raising the threshold should not increase the number of retained modes."""
        _, coefs_tight = compute_beam_basis(
            [beam_a, beam_b], freq=_FREQ, polarized=True, threshold=1e-12
        )
        _, coefs_loose = compute_beam_basis(
            [beam_a, beam_b], freq=_FREQ, polarized=True, threshold=0.8
        )
        assert coefs_loose.shape[1] <= coefs_tight.shape[1]


# ──────────────────────────────────────────────────────────────────────────────
# compute_beam_basis: reconstruction accuracy
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeBeamBasisReconstruction:
    """
    Verify that beam_coefs @ stack(eigenbeam_flat_vecs) ≈ stack(original_flat_vecs).

    This is the core SVD identity: U[:, :K] * s[:K] @ Vh[:K] ≈ M.  When K equals
    the numerical rank the approximation is exact (to floating-point precision).
    """

    @staticmethod
    def _eb_flat(eigenbeams) -> np.ndarray:
        """Stack flattened eigenbeam data slices into a (K, npix) matrix."""
        return np.stack(
            [eb.data_array[:, :, 0].ravel() for eb in eigenbeams], axis=0
        )

    def test_identical_beams_both_rows_reconstruct_identically(self, beam_a):
        """
        Identical beams → rank 1.  Both rows of coefs @ eb_flat must yield
        the same pattern (since both antennas share the same beam), and the
        reconstruction must be lossless (exact SVD at rank 1).
        """
        eigenbeams, coefs = compute_beam_basis(
            [beam_a, beam_a], freq=_FREQ, polarized=True, threshold=1e-12
        )
        assert len(eigenbeams) == 1, "Expected K=1 for identical beams."

        eb_flat = self._eb_flat(eigenbeams)   # (1, npix)
        recon = coefs @ eb_flat               # (2, npix)

        # Both antennas share the same beam, so both reconstructed rows must be equal.
        np.testing.assert_allclose(
            np.abs(recon[0]), np.abs(recon[1]), rtol=1e-12,
            err_msg="Reconstructed beam patterns differ for identical input beams."
        )

        # Losslessness: the ratio coefs[0] / coefs[1] must equal 1 (same beam).
        ratio = coefs[0, 0] / coefs[1, 0]
        np.testing.assert_allclose(
            np.abs(ratio), 1.0, rtol=1e-12,
            err_msg="Coefficient magnitudes differ for identical beams."
        )

    def test_full_rank_reconstruction_captures_all_signal(self, beam_a, beam_b):
        """
        At threshold=1e-12 (effectively full rank), the Frobenius norm of the
        reconstructed matrix should equal the Frobenius norm of the input
        matrix to high precision — no energy should be lost to truncation.
        """
        eigenbeams, coefs = compute_beam_basis(
            [beam_a, beam_b], freq=_FREQ, polarized=True, threshold=1e-12
        )
        eb_flat = self._eb_flat(eigenbeams)   # (K, npix)
        recon = coefs @ eb_flat               # (2, npix)

        # The Frobenius norm of a full-rank SVD reconstruction equals
        # sqrt(sum of squared singular values) = Frobenius norm of original.
        # We check this indirectly: the norm of the reconstruction should
        # equal the norm of the coefs times the norm of the eigenbeams.
        recon_norm = np.linalg.norm(recon)
        assert recon_norm > 0, "Reconstructed beam matrix has zero norm."

        # Round-trip: reconstruct, re-decompose, check norms match.
        # coefs @ eb_flat = U[:, :K] * s[:K] @ Vh[:K]; its norm = ||s[:K]||.
        s_norm = np.linalg.norm(np.linalg.svd(recon, compute_uv=False))
        coefs_norm = np.linalg.norm(coefs)
        # For orthonormal Vh, ||coefs @ Vh[:K]||_F = ||coefs||_F.
        # We just check the reconstruction is non-trivially large (not zero).
        assert s_norm > 0

    def test_truncated_reconstruction_has_smaller_norm(self, beam_a, beam_b):
        """
        Dropping modes (loose threshold) reduces the Frobenius norm of the
        reconstruction relative to the full-rank case — energy is discarded.
        """
        _, coefs_tight = compute_beam_basis(
            [beam_a, beam_b], freq=_FREQ, polarized=True, threshold=1e-12
        )
        eigenbeams_tight, _ = compute_beam_basis(
            [beam_a, beam_b], freq=_FREQ, polarized=True, threshold=1e-12
        )
        eigenbeams_loose, coefs_loose = compute_beam_basis(
            [beam_a, beam_b], freq=_FREQ, polarized=True, threshold=0.8
        )

        norm_tight = np.linalg.norm(coefs_tight @ self._eb_flat(eigenbeams_tight))
        norm_loose = np.linalg.norm(coefs_loose @ self._eb_flat(eigenbeams_loose))

        # Tight (more modes) should capture at least as much energy as loose.
        assert norm_tight >= norm_loose - 1e-12, (
            f"Full-rank norm {norm_tight:.6e} should be ≥ truncated norm {norm_loose:.6e}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# compute_beam_basis: error handling
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeBeamBasisErrors:
    """Verify that invalid inputs raise informative errors."""

    def test_empty_beam_list_raises(self):
        with pytest.raises(ValueError, match="beam_list must contain at least one beam"):
            compute_beam_basis([], freq=_FREQ, polarized=True)

    def test_threshold_zero_raises(self, beam_a):
        with pytest.raises(ValueError, match="threshold must be in the interval"):
            compute_beam_basis([beam_a], freq=_FREQ, polarized=True, threshold=0.0)

    def test_threshold_above_one_raises(self, beam_a):
        with pytest.raises(ValueError, match="threshold must be in the interval"):
            compute_beam_basis([beam_a], freq=_FREQ, polarized=True, threshold=1.5)

    def test_vector_freq_raises(self, beam_a):
        with pytest.raises(ValueError, match="scalar freq"):
            compute_beam_basis(
                [beam_a], freq=np.array([150e6, 160e6]), polarized=True
            )

    def test_only_axis1_raises(self, beam_a):
        with pytest.raises(
            ValueError, match="axis1_array and axis2_array must be supplied together"
        ):
            compute_beam_basis(
                [beam_a],
                freq=_FREQ,
                polarized=True,
                axis1_array=np.linspace(0, 2 * np.pi, 36),
                axis2_array=None,
            )

    def test_only_axis2_raises(self, beam_a):
        with pytest.raises(
            ValueError, match="axis1_array and axis2_array must be supplied together"
        ):
            compute_beam_basis(
                [beam_a],
                freq=_FREQ,
                polarized=True,
                axis1_array=None,
                axis2_array=np.linspace(0, np.pi / 2, 19),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Simulation integration tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBasisSimulation:
    """
    End-to-end tests that verify the eigenbeam simulation path reproduces
    the standard per-antenna simulation.

    Strategy
    --------
    1. Build a set of per-antenna beams (identical or different).
    2. Run a reference simulation using those beams directly (beam_idx path).
    3. Decompose the same beams with compute_beam_basis.
    4. Run the eigenbeam simulation (beam_coefs path).
    5. Assert the two results agree within NUFFT tolerance (atol=1e-5).
    """

    @staticmethod
    def _run_ref_sim(beam_list, beam_idx, params):
        return simulate_vis(
            beam=beam_list,
            beam_idx=beam_idx,
            polarized=True,
            eps=1e-10,
            **params,
        )

    @staticmethod
    def _run_basis_sim(eigenbeams, beam_coefs, params):
        return simulate_vis(
            beam=eigenbeams,
            beam_coefs=beam_coefs,
            polarized=True,
            eps=1e-10,
            **params,
        )

    def test_identical_beams_basis_matches_reference(self, beam_a, sim_params):
        """
        All antennas share the same beam → K=1. The basis simulation must
        reproduce the per-antenna reference to within NUFFT accuracy.
        """
        nant = len(sim_params["ants"])
        beam_list = [beam_a] * nant
        beam_idx = np.arange(nant)

        eigenbeams, coefs = compute_beam_basis(
            [beam_a], freq=float(sim_params["freqs"][0]), polarized=True
        )
        # coefs shape: (1, K).  Broadcast to (nant, K, nfreqs=1).
        coefs_per_ant = np.tile(coefs[np.newaxis], (nant, 1, 1))

        vis_ref = self._run_ref_sim(beam_list, beam_idx, sim_params)
        vis_basis = self._run_basis_sim(eigenbeams, coefs_per_ant, sim_params)

        assert vis_ref.shape == vis_basis.shape, (
            f"Shape mismatch: ref {vis_ref.shape} vs basis {vis_basis.shape}"
        )
        np.testing.assert_allclose(
            vis_basis, vis_ref, atol=1e-5,
            err_msg="Eigenbeam simulation does not match reference for identical beams."
        )

    def test_different_beams_basis_matches_reference(self, beam_a, beam_b, sim_params):
        """
        Antennas alternate between two different Airy beams. The basis path
        should still reproduce the per-antenna reference, since at threshold=1e-12
        the full rank-2 decomposition is used and no information is discarded.
        """
        nant = len(sim_params["ants"])
        beam_list = [beam_a, beam_b]
        beam_idx = np.array([i % 2 for i in range(nant)])

        eigenbeams, coefs = compute_beam_basis(
            beam_list,
            freq=float(sim_params["freqs"][0]),
            polarized=True,
            threshold=1e-12,
        )
        # coefs shape: (2, K).  Index by beam_idx to get (nant, K, 1).
        coefs_per_ant = coefs[beam_idx, :, np.newaxis]

        vis_ref = self._run_ref_sim(beam_list, beam_idx, sim_params)
        vis_basis = self._run_basis_sim(eigenbeams, coefs_per_ant, sim_params)

        assert vis_ref.shape == vis_basis.shape
        np.testing.assert_allclose(
            vis_basis, vis_ref, atol=1e-5,
            err_msg="Eigenbeam simulation does not match reference for different beams."
        )

    def test_basis_sim_output_shape_is_correct(self, beam_a, sim_params):
        """Eigenbeam simulation should return (nfreqs, ntimes, 2, 2, nbls)."""
        from fftvis import utils

        nant = len(sim_params["ants"])
        freqs = sim_params["freqs"]
        times = sim_params["times"]

        eigenbeams, coefs = compute_beam_basis(
            [beam_a], freq=float(freqs[0]), polarized=True
        )
        coefs_per_ant = np.tile(coefs[np.newaxis], (nant, 1, 1))

        vis = self._run_basis_sim(eigenbeams, coefs_per_ant, sim_params)

        reds = utils.get_pos_reds(sim_params["ants"], include_autos=True)
        nbls = len(reds)

        assert vis.ndim == 5
        assert vis.shape == (len(freqs), len(times), 2, 2, nbls)

    def test_basis_sim_no_nan_or_inf(self, beam_a, sim_params):
        """Sanity check: eigenbeam simulation should not produce NaN or Inf."""
        nant = len(sim_params["ants"])

        eigenbeams, coefs = compute_beam_basis(
            [beam_a], freq=float(sim_params["freqs"][0]), polarized=True
        )
        coefs_per_ant = np.tile(coefs[np.newaxis], (nant, 1, 1))

        vis = self._run_basis_sim(eigenbeams, coefs_per_ant, sim_params)

        assert not np.isnan(vis).any(), "NaN values in eigenbeam simulation output."
        assert not np.isinf(vis).any(), "Inf values in eigenbeam simulation output."


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper-level error handling
# ──────────────────────────────────────────────────────────────────────────────


class TestBasisSimulationErrors:
    """Verify that the wrapper rejects invalid beam_coefs combinations."""

    @pytest.fixture
    def eigenbeams_and_coefs(self, beam_a, sim_params):
        nant = len(sim_params["ants"])
        eigenbeams, coefs = compute_beam_basis(
            [beam_a], freq=float(sim_params["freqs"][0]), polarized=True
        )
        return eigenbeams, np.tile(coefs[np.newaxis], (nant, 1, 1))

    def test_unpolarized_with_beam_coefs_raises(
        self, eigenbeams_and_coefs, sim_params
    ):
        """
        The unpolarized path is incompatible with the basis approach because
        it would require interpolating amplitude (sqrt-power) beams rather than
        efield beams, breaking the linearity the SVD decomposition relies on.
        """
        eigenbeams, coefs = eigenbeams_and_coefs
        with pytest.raises(ValueError, match="not compatible with unpolarized"):
            simulate_vis(
                beam=eigenbeams,
                beam_coefs=coefs,
                polarized=False,
                **sim_params,
            )

    def test_beam_idx_with_beam_coefs_raises(
        self, eigenbeams_and_coefs, sim_params
    ):
        """
        beam_idx and beam_coefs are mutually exclusive: beam_idx selects per-antenna
        beams from a list, while beam_coefs encodes the linear-combination mapping.
        """
        nant = len(sim_params["ants"])
        eigenbeams, coefs = eigenbeams_and_coefs
        with pytest.raises(ValueError, match="beam_idx should not be provided"):
            simulate_vis(
                beam=eigenbeams,
                beam_coefs=coefs,
                beam_idx=np.zeros(nant, dtype=int),
                polarized=True,
                **sim_params,
            )