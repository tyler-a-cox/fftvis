import typer
from matvis.cli import get_standard_sim_params, get_line_based_stats, get_summary_stats, get_label, get_redundancies
from .simulate import simulate_vis, _evaluate_vis_chunk
from pathlib import Path
from typing import Optional
import logging
from rich.console import Console
from rich.rule import Rule
from line_profiler import LineProfiler
import time
from pyuvdata.telescopes import get_telescope
import cProfile
import pstats
import os
import numpy as np
from hera_sim.antpos import hex_array

cns = Console()

logger = logging.getLogger('fftvis')
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
):
    """Run the script."""
    logger.setLevel(log_level.upper())

    if nside > 0:
        nsource = 12*nside**2
        
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
    #cns.print(f"  NPAIRS:           {len(pairs) if pairs is not None else nants**2:>7}")
    cns.print(f"  NAZ:              {naz:>7}")
    cns.print(f"  NZA:              {nza:>7}")
    cns.print(f"  NPROCESSES:       {nprocesses:>7}")
    
    if coord_method == "CoordinateRotationERFA":
        cns.print(f"  BCRS UPDATE:       {update_bcrs_every:>7}")
        coord_params = {"update_bcrs_every": update_bcrs_every}
    else:
        coord_params = {}
        
    cns.print(Rule())

    profiler.add_function(simulate_vis)
    profiler.add_function(_evaluate_vis_chunk)

    init_time = time.time()

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
    
    cProfile.runctx(
        """simulate_vis(
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
    coord_method_params=coord_params,
    force_use_ray=force_use_ray,
        )""", 
        globals(),
        locals(), 
        str_id
    )
    
    out_time = time.time()
    cns.print("TOTAL TIME: ", out_time - init_time)
    
    p = pstats.Stats(str_id)
    p.sort_stats("cumulative").print_stats(50)
    os.system(f"flameprof --format=log {str_id} > {str_id}.flame")
    
                
if __name__=="__main__":
    app()