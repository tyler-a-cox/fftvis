import typer
from matvis.cli import get_standard_sim_params, get_line_based_stats, get_summary_stats, get_label
from .simulate import simulate_vis, _evaluate_vis_chunk
from pathlib import Path
from typing import Optional
import logging
from rich.console import Console
from rich.rule import Rule
from line_profiler import LineProfiler
import time
from pyuvdata.telescopes import get_telescope

cns = Console()

logger = logging.getLogger(__name__)
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
):
    """Run the script."""
    logger.setLevel(log_level.upper())

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
        analytic_beam, nfreq, ntimes, nants, nsource, nbeams=1, naz=naz, nza=nza
    )
    ants = {k: list(v) for k, v in ants.items()}
    
    cns.print(Rule("Running fftvis profile"))
    cns.print(f"  NANTS:            {nants:>7}")
    cns.print(f"  NTIMES:           {ntimes:>7}")
    cns.print(f"  NFREQ:            {nfreq:>7}")
    cns.print(f"  NSOURCE:          {nsource:>7}")
    cns.print(f"  DOUBLE-PRECISION: {double_precision:>7}")
    cns.print(f"  ANALYTIC-BEAM:    {analytic_beam:>7}")
    cns.print(f"  COORDROT METHOD:  {coord_method:>7}")
    #cns.print(f"  NPAIRS:           {len(pairs) if pairs is not None else nants**2:>7}")
    cns.print(f"  NAZ:              {naz:>7}")
    cns.print(f"  NZA:              {nza:>7}")
    cns.print(f"  NPROCESSES:       {nprocesses:>7}")
    
    cns.print(Rule())

    profiler.add_function(simulate_vis)
    profiler.add_function(_evaluate_vis_chunk)

    init_time = time.time()
    profiler.runcall(
        simulate_vis,
        ants=ants,
        fluxes=flux,
        ra=ra,
        dec=dec,
        freqs=freqs,
        times=times,
        beam=cpu_beams[0],
        polarized=True,
        precision=2 if double_precision else 1,
        telescope_loc=get_telescope("hera").location,
        nprocesses=nprocesses,
        #coord_method=coord_method,
        #baselines=pairs,
    )
    out_time = time.time()

    outdir = Path(outdir).expanduser().absolute()

    str_id = get_label(
        analytic_beam=analytic_beam,
        nfreq=nfreq,
        ntimes=ntimes,
        nants=nants,
        nsource=nsource,
        nbeams=1,
        matprod_method='',
        gpu=False,
        double_precision=double_precision,
        coord_method=coord_method,
        naz=naz,
        nza=nza,
    )

    with open(f"{outdir}/full-stats-{str_id}.txt", "w") as fl:
        profiler.print_stats(stream=fl, stripzeros=True)

    if verbose:
        profiler.print_stats()

    # line_stats = get_line_based_stats(profiler.get_stats())
    # thing_stats = get_summary_stats(line_stats, STEPS)

    # cns.print()
    # cns.print(Rule("Summary of timings"))
    # cns.print(f"         Total Time:            {out_time - init_time:.3e} seconds")
    # for thing, (hits, _time, time_per_hit, percent, nlines) in thing_stats.items():
    #     cns.print(
    #         f"{thing:>19}: {hits:>4} hits, {_time:.3e} seconds, {time_per_hit:.3e} sec/hit, {percent:4.2f}%, {nlines} lines"
    #     )
    # cns.print(Rule())

    # with open(f"{outdir}/summary-stats-{str_id}.pkl", "wb") as fl:
    #     pickle.dump(thing_stats, fl)
        
        
if __name__=="__main__":
    app()