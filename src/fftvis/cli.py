import typer
from matvis.cli import (
    get_standard_sim_params,
    get_label,
)
from pathlib import Path
import logging
from rich.console import Console
from rich.rule import Rule
from line_profiler import LineProfiler
import time
import cProfile
import pstats
import os
import numpy as np
from hera_sim.antpos import hex_array

# Import from the new structure
from .wrapper import simulate_vis
from .cpu.simulate import CPUSimulationEngine

cns = Console()

logger = logging.getLogger("fftvis")
profiler = LineProfiler()

app = typer.Typer()


@app.command()
def run_profile(
    analytic_beam: bool = True,
    nfreq: int = 1,
    ntimes: int = 1,
    nants: int = 1,
    nsource: int = 1,
    double_precision: bool = True,
    outdir: Path = Path(".").absolute(),
    verbose: bool = True,
    log_level: str = "INFO",
    coord_method: str = "CoordinateRotationERFA",
    naz: int = 360,
    nza: int = 180,
    nprocesses: int = 1,
    update_bcrs_every: float = np.inf,
    hera: int = 0,
    nside: int = 0,
    force_use_ray: bool = False,
    trace_mem: bool = False,
    beam_spline_order: int = 3,
    freq_min: float = 100,  # MHz
    backend: str = "cpu",
):
    """Run the script."""
    logger.setLevel(log_level.upper())

    if nside > 0:
        nsource = 12 * nside**2

    (
        ants,
        flux,
        ra,
        dec,
        freqs,
        times,
        cpu_beams,
        beam_idx,
    ) = get_standard_sim_params(
        analytic_beam,
        nfreq,
        ntimes,
        nants,
        nsource,
        nbeams=1,
        naz=naz,
        nza=nza,
        freq_min=freq_min * 1e6,
    )
    if hera > 0:
        ants = hex_array(hera)

    ants = {k: list(v) for k, v in ants.items()}
    nants = len(ants)

    cns.print(Rule("Running fftvis profile"))
    cns.print(f"  NANTS:            {nants:>7}")
    cns.print(f"  NTIMES:           {ntimes:>7}")
    cns.print(f"  NFREQ:            {nfreq:>7}")
    cns.print(f"  NSOURCE:          {nsource:>7}")
    cns.print(f"  DOUBLE-PRECISION: {double_precision:>7}")
    cns.print(f"  ANALYTIC-BEAM:    {analytic_beam:>7}")
    cns.print(f"  COORDROT METHOD:  {coord_method:>7}")
    cns.print(f"  NAZ:              {naz:>7}")
    cns.print(f"  NZA:              {nza:>7}")
    cns.print(f"  NPROCESSES:       {nprocesses:>7}")
    cns.print(f"  INTERP ORDER:     {beam_spline_order:>7}")
    cns.print(f"  BACKEND:          {backend:>7}")

    if coord_method == "CoordinateRotationERFA":
        cns.print(f"  BCRS UPDATE:       {update_bcrs_every:>7}")
        coord_params = {"update_bcrs_every": update_bcrs_every}
    else:
        coord_params = {}

    cns.print(Rule())

    # Add profiling to the simulate_vis function and the CPU implementation's _evaluate_vis_chunk
    profiler.add_function(simulate_vis)
    cpu_engine = CPUSimulationEngine()
    profiler.add_function(cpu_engine._evaluate_vis_chunk)

    init_time = time.time()

    str_id = get_label(
        analytic_beam=analytic_beam,
        nfreq=nfreq,
        ntimes=ntimes,
        nants=nants,
        nsource=nsource,
        nbeams=1,
        matprod_method="",
        gpu=backend == "gpu",
        double_precision=double_precision,
        coord_method=coord_method,
        naz=naz,
        nza=nza,
    )

    cProfile.runctx(
        """simulate_vis(
    ants=ants,
    fluxes=flux,
    ra=ra,
    dec=dec,
    freqs=freqs,
    times=times.jd,
    beam=cpu_beams[0],
    polarized=True,
    precision=2 if double_precision else 1,
    telescope_loc=get_telescope("hera").location,
    nprocesses=nprocesses,
    coord_method_params=coord_params,
    force_use_ray=force_use_ray,
    trace_mem=trace_mem,
    beam_spline_opts={'order': beam_spline_order},
    backend=backend,
        )""",
        globals(),
        locals(),
        str_id,
    )

    out_time = time.time()
    cns.print("TOTAL TIME: ", out_time - init_time)

    p = pstats.Stats(str_id)
    p.sort_stats("cumulative").print_stats(50)
    os.system(f"flameprof --format=log {str_id} > {str_id}.flame")


if __name__ == "__main__":
    app()
