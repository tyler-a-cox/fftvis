"""
CPU-specific simulation implementation for fftvis.

This module provides a concrete implementation of the simulation engine for CPU.
"""

from multiprocessing import cpu_count
import ray
from threadpoolctl import threadpool_limits
import os
import memray
import numpy as np
import time
import psutil
import logging
from typing import Literal, Union
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as un
from astropy.time import Time
from pyuvdata import UVBeam
from pyuvdata.beam_interface import BeamInterface
from matvis import coordinates
from matvis.core.coords import CoordinateRotation

from ..core.simulate import SimulationEngine, default_accuracy_dict
from ..core.antenna_gridding import check_antpos_griddability
from .. import utils

# Import the CPU beam evaluator
from .beams import CPUBeamEvaluator
from .nufft import cpu_nufft2d, cpu_nufft3d, cpu_nufft2d_type1
from . import utils as cpu_utils
logger = logging.getLogger(__name__)

# Create a global instance of CPUBeamEvaluator to use for beam evaluation
_cpu_beam_evaluator = CPUBeamEvaluator()

def _evaluate_beam_list(
    beam_list: list,
    az: np.ndarray,
    za: np.ndarray,
    polarized: bool,
    freq: float,
    beam_spline_opts: dict,
    interpolation_function: str,
    complex_dtype: np.dtype,
) -> list:
    """Evaluate every beam in beam_list at the given az/za positions.

    Parameters
    ----------
    beam_list : list
        List of beam objects (UVBeam or BeamInterface).
    az, za : np.ndarray
        Azimuth and zenith angle arrays (radians), shape (nsrc,).
    polarized : bool
        Whether to evaluate polarized beam components.
    freq : float
        Frequency in Hz.
    beam_spline_opts : dict or None
        Options passed to the spline interpolator.
    interpolation_function : str
        Name of the interpolation function to use.
    complex_dtype : np.dtype
        Complex dtype to cast the result to if needed.

    Returns
    -------
    list of np.ndarray
        One evaluated beam array per beam in beam_list. Shape is
        (nfeeds, nfeeds, nsrc) for polarized, (nsrc,) otherwise.
    """
    beam_evaluations = []
    for beam in beam_list:
        be = _cpu_beam_evaluator.evaluate_beam(
            beam,
            az,
            za,
            polarized,
            freq,
            spline_opts=beam_spline_opts,
            interpolation_function=interpolation_function,
        )
        beam_evaluations.append(
            be if be.dtype == complex_dtype else be.astype(complex_dtype)
        )
    return beam_evaluations


def _compute_apparent_coherency(
    beam_evaluations: list,
    bi: int,
    bj: int,
    flux_here: np.ndarray,
    freqidx: int,
    polarized: bool,
    polarized_sky_model: bool,
    nfeeds: int,
    nsim_sources: int,
    complex_dtype: np.dtype,
    apparent_buf: np.ndarray,
) -> np.ndarray:
    """Compute beam-weighted sky coherency for a single beam pair.

    Writes into *apparent_buf* in-place where possible to avoid allocation,
    then returns the reshaped (nfeeds**2, nsrc) array ready for the NUFFT.

    Parameters
    ----------
    beam_evaluations : list of np.ndarray
        Pre-evaluated beams, one per unique beam in beam_list.
    bi, bj : int
        Indices into beam_evaluations for the two antennas of this pair.
    flux_here : np.ndarray
        Source flux array, shape (nsrc, nfreqs) or (nsrc, nfreqs, nfeeds, nfeeds).
    freqidx : int
        Frequency index into flux_here.
    polarized : bool
        Whether the simulation is polarized.
    polarized_sky_model : bool
        Whether the sky model is itself polarized.
    nfeeds : int
        Number of feed dimensions (1 or 2).
    nsim_sources : int
        Number of simulated sources in this chunk.
    complex_dtype : np.dtype
        Complex dtype to ensure the output array matches.
    apparent_buf : np.ndarray
        Pre-allocated work buffer, shape (nfeeds, nfeeds, nsrc) or (nsrc,).

    Returns
    -------
    np.ndarray
        Apparent coherency shaped (nfeeds**2, nsrc).
    """
    is_cross_pair = bi != bj

    if polarized and polarized_sky_model:
        logger.debug(
            "Using polarized sky model. Computing apparent flux for polarized sources."
        )
        if is_cross_pair:
            apparent_buf[:] = 0
            apparent_coherency = apparent_buf
            _cpu_beam_evaluator.get_apparent_flux_polarized_pair(
                beam_i=np.flip(beam_evaluations[bi], axis=0),
                beam_j=np.flip(beam_evaluations[bj], axis=0),
                coherency=np.transpose(flux_here[:, freqidx], (1, 2, 0)),
                out=apparent_coherency,
            )
        else:
            np.copyto(apparent_buf, beam_evaluations[bi])
            apparent_coherency = np.flip(apparent_buf, axis=0)
            _cpu_beam_evaluator.get_apparent_flux_polarized(
                apparent_coherency, np.transpose(flux_here[:, freqidx], (1, 2, 0))
            )

    elif polarized:
        logger.debug(
            "Using polarized beam. Computing apparent flux for unpolarized sources."
        )
        if is_cross_pair:
            logger.debug("Processing cross pair")
            apparent_buf[:] = 0
            apparent_coherency = apparent_buf
            _cpu_beam_evaluator.get_apparent_flux_polarized_beam_pair(
                beam_i=beam_evaluations[bi],
                beam_j=beam_evaluations[bj],
                flux=flux_here[:, freqidx],
                out=apparent_coherency,
            )
        else:
            np.copyto(apparent_buf, beam_evaluations[bi])
            apparent_coherency = apparent_buf
            _cpu_beam_evaluator.get_apparent_flux_polarized_beam(
                apparent_coherency, flux_here[:, freqidx]
            )

    else:
        logger.debug(
            "Using unpolarized beam. Computing apparent flux for unpolarized sources."
        )
        np.multiply(beam_evaluations[bi], beam_evaluations[bj], out=apparent_buf)
        np.sqrt(apparent_buf, out=apparent_buf)

        apparent_buf *= flux_here[:, freqidx]
        apparent_coherency = apparent_buf

    # Reshape to (nfeeds**2, nsrc) for the NUFFT
    try:
        apparent_coherency = np.reshape(apparent_coherency, (nfeeds**2, nsim_sources))
    except ValueError:  # pragma: no cover
        logger.error(
            f"Cannot reshape apparent_coherency with shape {apparent_coherency.shape} "
            f"to {(nfeeds**2, nsim_sources)}"
        )
        return None

    if apparent_coherency.dtype != complex_dtype:
        apparent_coherency = apparent_coherency.astype(complex_dtype)

    return apparent_coherency


def _run_nufft(
    apparent_coherency: np.ndarray,
    topo: np.ndarray,
    uvw: np.ndarray,
    bls: np.ndarray,
    flipped: np.ndarray,
    bls_idxs: np.ndarray,
    use_type1: bool,
    is_coplanar: bool,
    tx: np.ndarray,
    ty: np.ndarray,
    type1_n_modes: int,
    eps: float,
    n_threads: int,
    upsample_factor: float,
    nfeeds: int,
) -> np.ndarray:
    """Dispatch to the appropriate NUFFT and return shaped visibilities.

    Parameters
    ----------
    apparent_coherency : np.ndarray
        Beam-weighted sky coherency, shape (nfeeds**2, nsrc).
    topo : np.ndarray
        Topocentric source coordinates, shape (3, nsrc).
    uvw : np.ndarray
        Frequency-scaled baseline vectors, shape (3, nbls_here).
    bls : np.ndarray
        Integer gridded baseline indices (type-1 path only).
    flipped : np.ndarray
        Boolean mask of baselines whose UVW was negated.
    bls_idxs : np.ndarray
        Indices of the baselines this beam pair contributes to.
    use_type1 : bool
        Use type-1 NUFFT (gridded array).
    is_coplanar : bool
        Use 2D instead of 3D NUFFT.
    tx, ty : np.ndarray
        Frequency-scaled topo[0/1] (type-1 path only).
    type1_n_modes : int
        Grid size for type-1 transform.
    eps, n_threads, upsample_factor : float / int
        NUFFT accuracy and performance parameters.
    nfeeds : int
        Number of feed dimensions.

    Returns
    -------
    np.ndarray
        Visibilities shaped (nbls_here, nfeeds, nfeeds).
    """
    nbls_here = len(bls_idxs)

    if use_type1:
        bls_here = np.where(flipped, -bls[:, bls_idxs], bls[:, bls_idxs])
        _vis_here = cpu_nufft2d_type1(
            tx,
            ty,
            apparent_coherency,
            n_modes=type1_n_modes,
            index=bls_here,
            eps=eps,
            n_threads=n_threads,
            upsample_factor=upsample_factor,
        )
    else:
        _uvw = np.where(flipped, -uvw[:, bls_idxs], uvw[:, bls_idxs])
        if is_coplanar:
            _vis_here = cpu_nufft2d(
                topo[0],
                topo[1],
                apparent_coherency,
                _uvw[0],
                _uvw[1],
                eps=eps,
                n_threads=n_threads,
                upsample_factor=upsample_factor,
            )
        else:
            _vis_here = cpu_nufft3d(
                topo[0],
                topo[1],
                topo[2],
                apparent_coherency,
                _uvw[0],
                _uvw[1],
                _uvw[2],
                eps=eps,
                n_threads=n_threads,
                upsample_factor=upsample_factor,
            )

    # Conjugate visibilities for baselines whose UVW was flipped
    _vis_here = np.where(flipped, np.conj(_vis_here), _vis_here)

    return np.swapaxes(_vis_here.reshape(nfeeds, nfeeds, nbls_here), 2, 0)


def _compute_basis_visibilities(
    beam_evaluations: list,
    flux_here: np.ndarray,
    ant1_idxs: np.ndarray,
    ant2_idxs: np.ndarray,
    beam_coefs: np.ndarray,
    freqidx: int,
    topo: np.ndarray,
    uvw: np.ndarray,
    bls: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    nbls: int,
    nfeeds: int,
    nsim_sources: int,
    complex_dtype: np.dtype,
    use_type1: bool,
    is_coplanar: bool,
    type1_n_modes: int,
    eps: float,
    n_threads: int,
    upsample_factor: float,
    polarized: bool = False,
    polarized_sky_model: bool = False,
) -> np.ndarray:
    """Compute the basis visibility tensor V_tilde[k, l] for all basis pairs.

    For each pair of basis beams (phi_k, phi_l), computes the NUFFT of
    phi_k * phi_l * flux over all baselines, accumulating into a tensor
    of shape (nbasis, nbasis, nbls, nfeeds, nfeeds).

    The apparent coherency phi_kl passed to the NUFFT depends on polarization:

    - Unpolarized beams::

        phi_kl[s] = beam_k[s] * beam_l[s]^* * flux[s]

    - Polarized beams, unpolarized sky::

        phi_kl[p, r, s] = sum_q  beam_k[p,q,s] * beam_l[r,q,s]^* * flux[s]

    - Polarized beams, polarized sky::

        phi_kl[p, r, s] = sum_{q,q'}  beam_k[p,q,s] * C[q,q',s] * beam_l[r,q',s]^*

    Parameters
    ----------
    beam_evaluations : list of np.ndarray
        Evaluated basis beams, length nbasis. Each has shape ``(nsrc,)`` for
        unpolarized or ``(nfeeds, nfeeds, nsrc)`` for polarized.
    flux_here : np.ndarray
        Source flux, shape ``(nsrc, nfreqs)`` for unpolarized sky or
        ``(nsrc, nfreqs, nfeeds, nfeeds)`` for polarized sky.
    ant1_idxs, ant2_idxs : np.ndarray
        Antenna indices for each baseline, each shape ``(nbls,)``.
    beam_coefs : np.ndarray
        Per-antenna basis coefficients, shape ``(nant, nbasis, nfreqs)``. Each baseline
        uses rows ``ant1_idxs``/``ant2_idxs`` at ``freqidx`` to form weights.
        Frequency index into flux_here.
    topo : np.ndarray
        Topocentric source coordinates, shape ``(3, nsrc)``.
    uvw : np.ndarray
        Frequency-scaled baseline vectors, shape ``(3, nbls)``.
    bls : np.ndarray
        Integer gridded baselines (type-1 path), shape ``(2, nbls)``.
    tx, ty : np.ndarray
        Frequency-scaled topo coordinates (type-1 path).
    nbls : int
        Total number of baselines.
    nfeeds : int
        Number of feed dimensions (1 or 2).
    nsim_sources : int
        Number of simulated sources.
    complex_dtype : np.dtype
        Complex dtype for accumulation.
    use_type1, is_coplanar : bool
        NUFFT mode flags.
    type1_n_modes : int
        Grid size for type-1 transform.
    eps, n_threads, upsample_factor : float / int
        NUFFT accuracy and performance parameters.
    polarized : bool
        Whether basis beams have polarization structure, i.e. shape
        ``(nfeeds, nfeeds, nsrc)``. Default False.
    polarized_sky_model : bool
        Whether the sky model carries full coherency, i.e. flux_here has
        shape ``(nsrc, nfreqs, nfeeds, nfeeds)``. Only used when
        ``polarized=True``. Default False.

    Returns
    -------
    np.ndarray
        Basis visibility tensor, shape ``(nbasis, nbasis, nbls, nfeeds, nfeeds)``.
    """
    nbasis = len(beam_evaluations)
    
    # Output accumulator — only (nbls, nfeeds, nfeeds) instead of (K, K, nbls, nfeeds, nfeeds)
    vis_out = np.zeros((nbls, nfeeds, nfeeds), dtype=complex_dtype)

    # No baseline flipping in the basis path — we run all baselines at once.
    flipped = np.zeros(nbls, dtype=bool)
    bls_idxs = np.arange(nbls)

    # Pre-allocate the apparent coherency work buffer, reused across all (k, l) pairs.
    # This mirrors the buffer allocation in the standard beam-pair path.
    if polarized:
        _apparent_buf = np.empty((nfeeds, nfeeds, nsim_sources), dtype=complex_dtype)
    else:
        _apparent_buf = np.empty(nsim_sources, dtype=complex_dtype)

    # Gather coefficients once, outside the loop.
    # The measurement equation is V_ij = A_i^H C A_j, so the left (ant1)
    # coefficients are conjugated and the right (ant2) are not.
    ant1_c = beam_coefs[ant1_idxs, :, freqidx].conj()  # C_ik^*  (nbls, K)
    ant2_c = beam_coefs[ant2_idxs, :, freqidx]          # C_jl    (nbls, K)

    # Only iterate over the upper triangle (k <= l) and use the conjugate
    # symmetry V_tilde[l, k] = V_tilde[k, l]^* to handle the lower triangle
    # without an extra NUFFT.  This halves the number of NUFFTs from K^2 to
    # K*(K+1)/2 at no cost to accuracy.
    for k in range(nbasis):
        for l in range(k, nbasis):
            phi_kl = _compute_apparent_coherency(
                beam_evaluations=beam_evaluations,
                bi=k,
                bj=l,
                flux_here=flux_here,
                freqidx=freqidx,
                polarized=polarized,
                polarized_sky_model=polarized_sky_model,
                nfeeds=nfeeds,
                nsim_sources=nsim_sources,
                complex_dtype=complex_dtype,
                apparent_buf=_apparent_buf,
            )

            if phi_kl is None:  # pragma: no cover (polarized path only)
                continue
            
            vis_kl = _run_nufft(
                apparent_coherency=phi_kl,
                topo=topo,
                uvw=uvw,
                bls=bls,
                flipped=flipped,
                bls_idxs=bls_idxs,
                use_type1=use_type1,
                is_coplanar=is_coplanar,
                tx=tx,
                ty=ty,
                type1_n_modes=type1_n_modes,
                eps=eps,
                n_threads=n_threads,
                upsample_factor=upsample_factor,
                nfeeds=nfeeds,
            )  # (nbls, nfeeds, nfeeds)

            # (k, l) contribution: weight = C1[b,k] * C2[b,l]^*
            w_kl = ant1_c[:, k] * ant2_c[:, l]   # (nbls,)
            vis_out += w_kl[:, None, None] * vis_kl

            if l != k:
                # (l, k) contribution: V_tilde[l,k] = V_tilde[k,l]^*
                # but the weights are different since ant1 != ant2 in general
                w_lk = ant1_c[:, l] * ant2_c[:, k]  # (nbls,)
                vis_out += w_lk[:, None, None] * vis_kl.swapaxes(1, 2)

    return vis_out


@ray.remote
def _evaluate_vis_chunk_remote(
    time_idx: slice,
    freq_idx: slice,
    beam_list: list[Union[UVBeam, BeamInterface]],
    coord_mgr,
    rotation_matrix: np.ndarray,
    antnums: list,
    bls: np.ndarray,
    baselines: list[tuple],
    freqs: np.ndarray,
    complex_dtype: np.dtype,
    nfeeds: int,
    beam_idx: np.ndarray = None,
    polarized: bool = False,
    polarized_sky_model: bool = False,
    eps: float = None,
    upsample_factor: Literal[1.25, 2] = 2,
    beam_spline_opts: dict = None,
    interpolation_function: str = "az_za_map_coordinates",
    n_threads: int = 1,
    is_coplanar: bool = False,
    use_type1: bool = False,
    basis_matrix: np.ndarray = None,
    type1_n_modes: int = None,
    trace_mem: bool = False,
    nchunks: int = 1,
    beam_coefs: np.ndarray = None,
):
    """Ray-compatible remote version of _evaluate_vis_chunk."""
    engine = CPUSimulationEngine()  # pragma: no cover
    return engine._evaluate_vis_chunk(  # pragma: no cover
        time_idx=time_idx,
        freq_idx=freq_idx,
        beam_list=beam_list,
        beam_idx=beam_idx,
        coord_mgr=coord_mgr,
        rotation_matrix=rotation_matrix,
        antnums=antnums,
        bls=bls,
        baselines=baselines,
        freqs=freqs,
        complex_dtype=complex_dtype,
        nfeeds=nfeeds,
        polarized=polarized,
        polarized_sky_model=polarized_sky_model,
        eps=eps,
        upsample_factor=upsample_factor,
        beam_spline_opts=beam_spline_opts,
        interpolation_function=interpolation_function,
        n_threads=n_threads,
        is_coplanar=is_coplanar,
        use_type1=use_type1,
        basis_matrix=basis_matrix,
        type1_n_modes=type1_n_modes,
        trace_mem=trace_mem,
        nchunks=nchunks,
        beam_coefs=beam_coefs,
    )


class CPUSimulationEngine(SimulationEngine):
    """CPU implementation of the simulation engine."""

    def simulate(
        self,
        ants: dict,
        freqs: np.ndarray,
        fluxes: np.ndarray,
        beam_list: list[Union[UVBeam, BeamInterface]],
        ra: np.ndarray,
        dec: np.ndarray,
        times: Union[np.ndarray, Time],
        telescope_loc: EarthLocation,
        baselines: list[tuple] = None,
        beam_idx: np.ndarray = None,
        precision: int = 2,
        polarized: bool = False,
        eps: float = None,
        upsample_factor: Literal[1.25, 2] = 2,
        beam_spline_opts: dict = None,
        flat_array_tol: float = 1e-6,
        interpolation_function: str = "az_za_map_coordinates",
        nprocesses: int | None = 1,
        nthreads: int | None = None,
        coord_method: Literal[
            "CoordinateRotationAstropy", "CoordinateRotationERFA"
        ] = "CoordinateRotationERFA",
        coord_method_params: dict | None = None,
        force_use_ray: bool = False,
        force_use_type3: bool = False,
        trace_mem: bool = False,
        enable_memory_monitor: bool = False,
        nchunks: int = 1,
        source_buffer=1.0,
        beam_coefs: np.ndarray = None,
    ) -> np.ndarray:
        """
        Simulate visibilities using CPU implementation.

        Parameters
        ----------
        beam_coefs : np.ndarray, optional
            Per-antenna basis coefficients, shape ``(nant, K, nfreqs)``. When provided,
            beam_list is interpreted as K basis beams rather than per-antenna beams.
            Visibilities are computed for basis-beam pairs and contracted using these
            coefficients at each frequency.

        See base class for all other parameter descriptions.
        """
        # Get sizes of inputs
        nfreqs = np.size(freqs)
        ntimes = len(times)
        nbeam = len(beam_list)
        nant = len(ants)

        nax = nfeeds = 2 if polarized else 1

        if precision == 1:
            real_dtype = np.float32
            complex_dtype = np.complex64
        else:
            real_dtype = np.float64
            complex_dtype = np.complex128

        if eps is None:
            eps = default_accuracy_dict[precision]

        if ra.dtype != real_dtype:
            ra = ra.astype(real_dtype)
        if dec.dtype != real_dtype:
            dec = dec.astype(real_dtype)
        if freqs.dtype != real_dtype:
            freqs = freqs.astype(real_dtype)

        # Check beam_idx validity against input beams and antennas.
        if beam_idx is None:
            if nbeam == nant:
                beam_idx = np.arange(nant)
            elif nbeam != 1:
                raise ValueError(
                    "If number of beams provided is not 1 or nant, beam_idx must be provided."
                )
        if beam_idx is not None:
            if beam_idx.shape != (nant,):
                raise ValueError("beam_idx must be length nant")
            if not all(0 <= i < nbeam for i in beam_idx):
                raise ValueError(
                    "beam_idx contains indices greater than the number of beams"
                )

        # Get the redundant groups
        if baselines is None:
            reds = utils.get_pos_reds(ants, include_autos=True)
            baselines = [red[0] for red in reds]

        # Get number of baselines
        nbls = len(baselines)

        # Prepare source catalog for the given fluxes
        coherency, polarized_sky_model = cpu_utils.prepare_source_catalog(
            fluxes, polarized_beam=polarized
        )
        if coherency.dtype != complex_dtype:
            coherency = coherency.astype(complex_dtype)

        # Flatten antenna positions
        antnums = list(ants.keys())
        antkey_to_idx = dict(zip(ants.keys(), range(len(ants))))
        antvecs = np.array([ants[ant] for ant in ants], dtype=real_dtype)

        # If the array is flat within tolerance, we can check for griddability
        if np.abs(antvecs[:, -1]).max() > flat_array_tol or force_use_type3:
            is_gridded = False
        else:
            is_gridded, gridded_antpos, basis_matrix = check_antpos_griddability(ants)

        # Rotate antenna positions to XY plane if not gridded
        if not is_gridded:
            # Get the rotation matrix to rotate the array to the XY plane
            rotation_matrix = utils.get_plane_to_xy_rotation_matrix(antvecs)
            rotation_matrix = np.ascontiguousarray(rotation_matrix.T)
            rotated_antvecs = np.dot(rotation_matrix, antvecs.T)
            rotated_ants = {
                ant: rotated_antvecs[:, antkey_to_idx[ant]] for ant in ants
            }
            rotation_matrix = rotation_matrix.astype(real_dtype)

            bls = np.array([rotated_ants[bl[1]] - rotated_ants[bl[0]] for bl in baselines])[
                :, :
            ].T
            bls /= utils.speed_of_light
            bls = bls.astype(real_dtype)

            # Check if the array is flat within tolerance
            is_coplanar = np.all(np.less_equal(np.abs(bls[2]), flat_array_tol))
        else:
            logger.info(
                "Using gridded coordinates for the array. Type 1 transform will be used."
            )
            # Compute the baseline vectors in the gridded coordinate system
            bls = np.array([
                gridded_antpos[bl[1]] - gridded_antpos[bl[0]]
                for bl in baselines]
            ).T
            bls = np.round(bls).astype(int)

            # Find the maximum extent of the array in gridded coordinates
            n_modes = 2 * int(np.round(np.max(np.abs(bls)))) + 1

            # Get the maximum baseline length for proper coordinate scaling
            basis_matrix *= 1 / utils.speed_of_light
            basis_matrix = basis_matrix.astype(real_dtype)

            # Assume the array is coplanar for gridded coordinates
            is_coplanar = True
            rotation_matrix = np.eye(3, dtype=real_dtype)

        # Get number of processes for multiprocessing
        if nprocesses is None:
            nprocesses = cpu_count()  # pragma: no cover

        # Check if the times array is a numpy array
        if isinstance(times, np.ndarray):
            times = Time(times, format="jd")

        chunk_size = int(np.ceil(dec.size / nchunks))

        coord_method = CoordinateRotation._methods[coord_method]
        coord_method_params = coord_method_params or {}
        coord_mgr = coord_method(
            flux=coherency,
            times=times,
            telescope_loc=telescope_loc,
            skycoords=SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame="icrs"),
            precision=precision,
            source_buffer=source_buffer,
            chunk_size=chunk_size,
            **coord_method_params,
        )

        if getattr(coord_mgr, "update_bcrs_every", 0) > (times[-1] - times[0]).to(un.s):
            # We don't need to ever update BCRS, so we get it now before sending
            # out the jobs to multiple processes.
            coord_mgr._set_bcrs(0)  # pragma: no cover

        nprocesses, freq_chunks, time_chunks, nf, nt = utils.get_task_chunks(
            nprocesses, nfreqs, ntimes
        )
        use_ray = nprocesses > 1 or force_use_ray

        if use_ray:  # pragma: no cover
            # Try to estimate how much shared memory will be required.
            required_shm = bls.nbytes + rotation_matrix.nbytes + freqs.nbytes

            for key, val in coord_mgr.__dict__.items():
                if isinstance(val, np.ndarray):
                    required_shm += val.nbytes

            for beam in beam_list:
                if isinstance(beam, BeamInterface) and getattr(beam, "_isuvbeam", False):
                    required_shm += beam.beam.data_array.nbytes

            required_shm += (ntimes * nfreqs * nbls * nax * nfeeds) * np.dtype(
                complex_dtype
            ).itemsize

            logger.info(
                f"Initializing with {2*required_shm/1024**3:.2f} GB of shared memory"
            )
            if not ray.is_initialized():
                if trace_mem:
                    os.environ["RAY_record_ref_creation_sites"] = "1"

                if not enable_memory_monitor:
                    os.environ["RAY_memory_monitor_refresh_ms"] = "0"

                # Only spill shared memory objects to disk if the Store is totally full.
                # If we don't do this, then since we need a relatively small amount of
                # SHM, it starts writing to disk even though we never needed to.
                os.environ["RAY_object_spilling_threshold"] = "1.0"

                try:
                    ray.init(
                        num_cpus=nprocesses,
                        object_store_memory=2 * required_shm,
                        include_dashboard=False,
                    )
                except ValueError:
                    ray.init()

            if trace_mem:
                os.system("ray memory --units MB > before-puts.txt")

            antnums = ray.put(antnums)
            baselines = ray.put(baselines)
            bls = ray.put(bls)
            rotation_matrix = ray.put(rotation_matrix)
            freqs = ray.put(freqs)
            beam_list = ray.put(beam_list)
            coord_mgr = ray.put(coord_mgr)
            if beam_coefs is not None:
                beam_coefs = ray.put(beam_coefs)
            if trace_mem:
                os.system("ray memory --units MB > after-puts.txt")

        ncpus = nthreads or cpu_count()
        nthreads_per_proc = [
            ncpus // nprocesses + (i < ncpus % nprocesses) for i in range(nprocesses)
        ]
        _nbig = nthreads_per_proc.count(nthreads_per_proc[0])
        if _nbig != nprocesses:
            logger.info(
                f"Splitting calculation into {nprocesses} processes. {_nbig} processes will"
                f" use {nthreads_per_proc[0]} threads each, and {nprocesses-_nbig} will use"
                f" {nthreads_per_proc[-1]} threads. Each process will compute "
                f"{nf} frequencies and {nt} times."
            )
        else:
            logger.info(
                f"Splitting calculation into {nprocesses} processes with "
                f"{nthreads_per_proc[0]} threads per-process. Each process will compute "
                f"{nf} frequencies and {nt} times."
            )

        futures = []
        init_time = time.time()

        # Create a remote version of _evaluate_vis_chunk if needed
        if use_ray:
            fnc = _evaluate_vis_chunk_remote.remote
        else:
            fnc = self._evaluate_vis_chunk

        # Create workers for each chunk
        for nthi, fc, tc in zip(nthreads_per_proc, freq_chunks, time_chunks):
            futures.append(
                fnc(
                    time_idx=tc,
                    freq_idx=fc,
                    beam_list=beam_list,
                    coord_mgr=coord_mgr,
                    rotation_matrix=rotation_matrix,
                    antnums=antnums,
                    baselines=baselines,
                    bls=bls,
                    freqs=freqs,
                    complex_dtype=complex_dtype,
                    nfeeds=nfeeds,
                    beam_idx=beam_idx,
                    polarized=polarized,
                    polarized_sky_model=polarized_sky_model,
                    eps=eps,
                    upsample_factor=upsample_factor,
                    beam_spline_opts=beam_spline_opts,
                    interpolation_function=interpolation_function,
                    n_threads=nthi,
                    is_coplanar=is_coplanar,
                    use_type1=is_gridded,
                    basis_matrix=basis_matrix if is_gridded else None,
                    type1_n_modes=n_modes if is_gridded else None,
                    trace_mem=(nprocesses > 1 or force_use_ray) and trace_mem,
                    nchunks=nchunks,
                    beam_coefs=beam_coefs,
                )
            )
            if trace_mem:
                os.system("ray memory --units MB > after-futures.txt")

        if use_ray:
            futures = ray.get(futures)
            if trace_mem:
                os.system("ray memory --units MB > got-all.txt")

        end_time = time.time()
        logger.info(f"Main loop evaluation time: {end_time - init_time}")
        
        # Combine results from all workers
        vis = np.zeros(
            dtype=complex_dtype, shape=(ntimes, nbls, nfeeds, nfeeds, nfreqs)
        )
        for fc, tc, future in zip(freq_chunks, time_chunks, futures):
            vis[tc][..., fc] = future

        # Reshape to expected output format
        return (
            np.transpose(vis, (4, 0, 2, 3, 1))
            if polarized
            else np.moveaxis(vis[..., 0, 0, :], 2, 0)
        )

    def _evaluate_vis_chunk(
        self,
        time_idx: slice,
        freq_idx: slice,
        beam_list: list[Union[UVBeam, BeamInterface]],
        coord_mgr: CoordinateRotation,
        rotation_matrix: np.ndarray,
        antnums: list,
        baselines: list[tuple],
        bls: np.ndarray,
        freqs: np.ndarray,
        complex_dtype: np.dtype,
        nfeeds: int,
        beam_idx: np.ndarray = None,
        polarized: bool = False,
        polarized_sky_model: bool = False,
        eps: float = None,
        upsample_factor: Literal[1.25, 2] = 2,
        beam_spline_opts: dict = None,
        interpolation_function: str = "az_za_map_coordinates",
        n_threads: int = 1,
        is_coplanar: bool = False,
        use_type1: bool = False,
        basis_matrix: float = None,
        type1_n_modes: int = None,
        trace_mem: bool = False,
        nchunks: int = 1,
        beam_coefs: np.ndarray = None,
    ) -> np.ndarray:
        """
        Evaluate a chunk of visibility data using CPU.

        Parameters
        ----------
        beam_coefs : np.ndarray, optional
            Per-antenna SVD coefficients, shape (N_ant, K). When provided,
            beam_list contains the K basis beams and the standard beam-pair
            loop is replaced by the basis visibility path.

        See base class for all other parameter descriptions.
        """
        pid = os.getpid()
        pr = psutil.Process(pid)

        if trace_mem:
            memray.Tracker(f"memray-{time.time()}_{pid}.bin").__enter__()

        nbls = bls.shape[1]
        ntimes = len(coord_mgr.times)
        nfreqs = len(freqs)

        nt_here = len(coord_mgr.times[time_idx])
        nf_here = len(freqs[freq_idx])
        vis = np.zeros(
            dtype=complex_dtype, shape=(nt_here, nbls, nfeeds, nfeeds, nf_here)
        )

        coord_mgr.setup()

        use_basis = beam_coefs is not None

        if use_basis:
            # Pre-compute the per-baseline antenna index arrays once.
            # These are used in post-processing to gather the right rows of beam_coefs.
            ant1_idxs = np.array([antnums.index(bl[0]) for bl in baselines])
            ant2_idxs = np.array([antnums.index(bl[1]) for bl in baselines])
        else:
            (
                unique_beam_pairs,
                beam_pair_to_bls_idxs,
                beam_pair_to_flipped,
            ) = _cpu_beam_evaluator.prepare_beam_evaluation(
                antnums=antnums,
                baselines=baselines,
                beam_idx=beam_idx,
            )

        is_rotation_identity = np.allclose(rotation_matrix, np.eye(3))

        with threadpool_limits(limits=n_threads, user_api="blas"):
            for time_index, ti in enumerate(range(ntimes)[time_idx]):
                coord_mgr.rotate(ti)

                for chunk in range(nchunks):
                    topo, flux, nsim_sources = coord_mgr.select_chunk(chunk, ti)

                    topo = topo[:, :nsim_sources]
                    flux = flux[:nsim_sources]

                    if nsim_sources == 0:
                        continue

                    # Pre-allocate apparent coherency work buffer (standard path only)
                    if not use_basis:
                        if polarized:
                            _apparent_buf = np.empty(
                                (nfeeds, nfeeds, nsim_sources), dtype=complex_dtype
                            )
                        else:
                            _apparent_buf = np.empty(nsim_sources, dtype=complex_dtype)

                    az, za = coordinates.enu_to_az_za(
                        enu_e=topo[0], enu_n=topo[1], orientation="uvbeam"
                    )

                    if not is_rotation_identity:
                        cpu_utils.inplace_rot(rotation_matrix, topo)

                    if basis_matrix is not None:
                        cpu_utils.inplace_rot(basis_matrix.T, topo)

                    topo *= 2 * np.pi

                    for freqidx in range(nfreqs)[freq_idx]:
                        freq = freqs[freqidx]

                        if not use_type1:
                            uvw = bls * freq

                        beam_evaluations = _evaluate_beam_list(
                            beam_list=beam_list,
                            az=az,
                            za=za,
                            polarized=polarized,
                            freq=freq,
                            beam_spline_opts=beam_spline_opts,
                            interpolation_function=interpolation_function,
                            complex_dtype=complex_dtype,
                        )

                        # Pre-compute frequency-scaled source coordinates once per
                        # frequency. topo and freq are identical for every beam pair,
                        # so computing these inside the pair loop would allocate two
                        # redundant (nsrc,) arrays per pair.
                        if use_type1:
                            tx = topo[0] * freq
                            ty = topo[1] * freq

                        # -------------------------------------------------------
                        # Basis visibility path
                        # -------------------------------------------------------
                        if use_basis:
                            vis_basis = _compute_basis_visibilities(
                                beam_evaluations=beam_evaluations,
                                flux_here=flux,
                                ant1_idxs=ant1_idxs,
                                ant2_idxs=ant2_idxs,
                                beam_coefs=beam_coefs,
                                freqidx=freqidx,
                                topo=topo,
                                uvw=uvw if not use_type1 else None,
                                bls=bls,
                                tx=tx if use_type1 else None,
                                ty=ty if use_type1 else None,
                                nbls=nbls,
                                nfeeds=nfeeds,
                                nsim_sources=nsim_sources,
                                complex_dtype=complex_dtype,
                                use_type1=use_type1,
                                is_coplanar=is_coplanar,
                                type1_n_modes=type1_n_modes,
                                eps=eps,
                                n_threads=n_threads,
                                upsample_factor=upsample_factor,
                                polarized=polarized,
                                polarized_sky_model=polarized_sky_model,
                            )

                            vis[time_index, :, :, :, freqidx] += vis_basis

                        # -------------------------------------------------------
                        # Standard beam-pair path
                        # -------------------------------------------------------
                        else:
                            for (bi, bj) in unique_beam_pairs:
                                bls_idxs = beam_pair_to_bls_idxs[(bi, bj)]
                                flipped = beam_pair_to_flipped[(bi, bj)]

                                apparent_coherency = _compute_apparent_coherency(
                                    beam_evaluations=beam_evaluations,
                                    bi=bi,
                                    bj=bj,
                                    flux_here=flux,
                                    freqidx=freqidx,
                                    polarized=polarized,
                                    polarized_sky_model=polarized_sky_model,
                                    nfeeds=nfeeds,
                                    nsim_sources=nsim_sources,
                                    complex_dtype=complex_dtype,
                                    apparent_buf=_apparent_buf,
                                )

                                if apparent_coherency is None:  # pragma: no cover
                                    continue

                                _vis_here = _run_nufft(
                                    apparent_coherency=apparent_coherency,
                                    topo=topo,
                                    uvw=uvw if not use_type1 else None,
                                    bls=bls,
                                    flipped=flipped,
                                    bls_idxs=bls_idxs,
                                    use_type1=use_type1,
                                    is_coplanar=is_coplanar,
                                    tx=tx if use_type1 else None,
                                    ty=ty if use_type1 else None,
                                    type1_n_modes=type1_n_modes,
                                    eps=eps,
                                    n_threads=n_threads,
                                    upsample_factor=upsample_factor,
                                    nfeeds=nfeeds,
                                )

                                vis[time_index, bls_idxs, ..., freqidx] += _vis_here

        return vis
