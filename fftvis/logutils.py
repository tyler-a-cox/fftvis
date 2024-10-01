"""Utilities for logging / output during the simulation."""

import datetime
import logging
import time
import tracemalloc as tm


logger = logging.getLogger(__name__)


def human_readable_size(size, decimal_places=2, indicate_sign=False):
    """Get a human-readable data size.

    From: https://stackoverflow.com/a/43690506/1467820
    """
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if abs(size) < 1024.0:
            break
        if unit != "PiB":
            size /= 1024.0

    if indicate_sign:
        return f"{size:+.{decimal_places}f} {unit}"
    else:
        return f"{size:.{decimal_places}f} {unit}"


def memtrace(highest_peak) -> int:
    if logger.isEnabledFor(logging.INFO):
        cm, pm = tm.get_traced_memory()
        logger.info(f"Starting Memory usage  : {cm / 1024**3:.3f} GB")
        logger.info(f"Starting Peak Mem usage: {pm / 1024**3:.3f} GB")
        logger.info(f"Traemalloc Peak Memory (tot)(GB): {highest_peak / 1024**3:.2f}")
        tm.reset_peak()
        return max(pm, highest_peak)


def log_progress(start_time, prev_time, iters, niters, pr, last_mem):
    """Logging of progress."""
    if not logger.isEnabledFor(logging.INFO):
        return prev_time, last_mem

    t = time.time()
    lapsed = datetime.timedelta(seconds=(t - prev_time))
    total = datetime.timedelta(seconds=(t - start_time))
    per_iter = total / iters
    expected = per_iter * niters

    rss = pr.memory_info().rss
    mem = human_readable_size(rss)
    memdiff = human_readable_size(rss - last_mem, indicate_sign=True)

    logger.info(
        f"""
        Progress Info   [{iters}/{niters} times ({100 * iters / niters:.1f}%)]
            -> Update Time:   {lapsed}
            -> Total Time:    {total} [{per_iter} per integration]
            -> Expected Time: {expected} [{expected - total} remaining]
            -> Memory Usage:  {mem}  [{memdiff}]
        """
    )

    return t, rss
