import cupy as cp
import logging
import gc
from typing import Callable, Any, Tuple

logger = logging.getLogger(__name__)


def cleanup_gpu_memory():
    """
    Release cached GPU memory back to the system.

    CuPy's memory pool caches freed allocations rather than returning them to
    CUDA.  This function flushes those pools so that a subsequent allocation
    attempt can use the freed memory.  It is called by ``execute_with_retry``
    between OOM retry attempts.
    """
    try:
        cp.cuda.runtime.deviceSynchronize()
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        gc.collect()
    except Exception as e:
        logger.debug(f"GPU memory cleanup warning: {e}")


def execute_with_retry(
    nufft_func: Callable,
    func_args: tuple,
    func_kwargs: dict,
    initial_batch_size: int,
    max_retries: int = 3,
    logger_context: str = "NUFFT"
) -> Tuple[Any, int]:
    """
    Execute a NUFFT function with automatic retry and batch size reduction on memory errors.

    This helper encapsulates the retry logic for GPU NUFFT operations, handling
    CUDA out-of-memory errors by reducing batch size and cleaning up GPU memory.

    Parameters
    ----------
    nufft_func : Callable
        The NUFFT function to execute (e.g., gpu_nufft2d_batch, gpu_nufft3d_batch)
    func_args : tuple
        Positional arguments to pass to nufft_func
    func_kwargs : dict
        Keyword arguments to pass to nufft_func
    initial_batch_size : int
        The initial batch size to try
    max_retries : int, optional
        Maximum number of retry attempts (default: 3)
    logger_context : str, optional
        Context string for log messages (default: "NUFFT")

    Returns
    -------
    result : Any
        The result from the NUFFT function, or None if all retries failed
    final_batch_size : int
        The batch size that succeeded (may be smaller than initial)

    Raises
    ------
    RuntimeError
        If a critical CUDA error occurs (e.g., cudaErrorInvalidValue)
    Exception
        If all retry attempts fail
    """
    current_batch_size = initial_batch_size
    retry_count = 0
    result = None

    while retry_count < max_retries:
        try:
            result = nufft_func(*func_args, **func_kwargs)
            break  # Success, exit retry loop

        except (RuntimeError, cp.cuda.memory.OutOfMemoryError,
                cp.cuda.runtime.CUDARuntimeError) as e:
            retry_count += 1
            error_str = str(e)

            # Check for critical CUDA errors that indicate corrupted GPU state
            if "cudaErrorInvalidValue" in error_str:
                logger.error(
                    f"Critical CUDA error during {logger_context}: {e}"
                )
                try:
                    cp.cuda.runtime.deviceReset()
                except Exception:
                    pass
                raise RuntimeError(
                    f"CUDA error during {logger_context}: {e}"
                )

            # Check if we've exhausted retries
            if retry_count >= max_retries:
                logger.error(
                    f"GPU memory exhausted after {max_retries} retries "
                    f"during {logger_context}."
                )
                raise

            # Reduce batch size and retry
            new_batch_size = max(1, current_batch_size // 2)
            logger.warning(
                f"GPU memory error during {logger_context} with batch size {current_batch_size}. "
                f"Retrying with reduced batch size: {new_batch_size}"
            )
            current_batch_size = new_batch_size

            cleanup_gpu_memory()

    return result, current_batch_size


def inplace_rot(rot: cp.ndarray, b: cp.ndarray):
    """
    GPU implementation of in-place rotation of coordinates using CuPy.

    Parameters
    ----------
    rot : cp.ndarray
        3x3 rotation matrix (on GPU)
    b : cp.ndarray
        Array of shape (3, n) containing coordinates to rotate (on GPU)
    """
    cp.matmul(rot, b, out=b)
