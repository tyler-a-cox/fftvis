import cupy as cp
import logging
import gc
import time
from typing import Callable, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def cleanup_gpu_memory():
    """
    Perform aggressive GPU memory cleanup.

    This function clears all cached memory pools and forces garbage collection.
    """
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        gc.collect()
        # Small delay to let GPU recover
        time.sleep(0.1)
    except Exception:
        pass


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
                    f"Critical GPU error detected during {logger_context}: {e}\n"
                    f"GPU memory is exhausted and cannot allocate more.\n"
                    f"Please switch to CPU backend by using backend='cpu' in your simulate_vis() call."
                )
                # Try to reset GPU state
                try:
                    cp.cuda.runtime.deviceReset()
                except Exception:
                    pass
                raise RuntimeError(
                    "GPU out of memory - cannot process this dataset on GPU. "
                    "Please use backend='cpu' to continue processing."
                )

            # Check if we've exhausted retries
            if retry_count >= max_retries:
                logger.error(
                    f"GPU memory exhausted after {max_retries} retries with reduced batch sizes.\n"
                    f"This dataset is too large for your GPU memory.\n"
                    f"Please switch to CPU backend by using backend='cpu' in your simulate_vis() call."
                )
                raise

            # Reduce batch size and retry
            new_batch_size = max(1, current_batch_size // 2)
            logger.warning(
                f"GPU memory error during {logger_context} with batch size {current_batch_size}. "
                f"Retrying with reduced batch size: {new_batch_size}"
            )
            current_batch_size = new_batch_size

            # Aggressive GPU memory cleanup
            cleanup_gpu_memory()

    return result, current_batch_size


def inplace_rot(rot: cp.ndarray, b: cp.ndarray):
    """
    GPU implementation of in-place rotation of coordinates using CuPy.

    Parameters:
    ----------
    rot : cp.ndarray
        3x3 rotation matrix (on GPU)
    b : cp.ndarray
        Array of shape (3, n) containing coordinates to rotate (on GPU)
    """
    # Ensure inputs are cupy arrays
    if not isinstance(rot, cp.ndarray):
        rot = cp.asarray(rot)
    if not isinstance(b, cp.ndarray):
        b = cp.asarray(b)

    # Perform matrix multiplication in-place
    cp.matmul(rot, b, out=b)
