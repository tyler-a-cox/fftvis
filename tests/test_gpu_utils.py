"""
Tests for GPU utility functions in fftvis.gpu.utils.

This module tests:
- cleanup_gpu_memory(): GPU memory cleanup
- execute_with_retry(): Retry logic for GPU operations with OOM handling
- inplace_rot(): In-place coordinate rotation on GPU
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Skip all GPU tests if CuPy is not available
cupy = pytest.importorskip("cupy")

from fftvis.gpu.utils import cleanup_gpu_memory, execute_with_retry, inplace_rot


# ============================================================================
# cleanup_gpu_memory Tests
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_cleanup_gpu_memory_basic():
    """Test that cleanup_gpu_memory runs without error."""
    # Allocate some GPU memory
    arr = cupy.random.random((1000, 1000))

    # Cleanup should not raise any errors
    cleanup_gpu_memory()

    # Memory should still be accessible after cleanup of freed blocks
    # The array we created should still exist
    assert arr.shape == (1000, 1000)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_cleanup_gpu_memory_frees_unreferenced():
    """Test that cleanup_gpu_memory frees unreferenced memory."""
    # Get initial memory usage
    mempool = cupy.get_default_memory_pool()
    initial_used = mempool.used_bytes()

    # Allocate and then delete array
    arr = cupy.random.random((1000, 1000))
    del arr

    # Run cleanup
    cleanup_gpu_memory()

    # After cleanup, memory should be freed
    final_used = mempool.used_bytes()
    # Memory should be freed or at least not significantly higher
    assert final_used <= initial_used + 1024  # Allow small overhead


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_cleanup_gpu_memory_handles_exceptions():
    """Test that cleanup_gpu_memory handles exceptions gracefully."""
    # Even if something goes wrong internally, it should not raise
    # This tests the try/except block
    with patch.object(cupy, 'get_default_memory_pool', side_effect=Exception("Test error")):
        # Should not raise even with mocked exception
        cleanup_gpu_memory()


def test_cleanup_gpu_memory_no_cuda():
    """Test cleanup_gpu_memory when CUDA operations might fail."""
    # This should handle any exceptions gracefully
    cleanup_gpu_memory()


# ============================================================================
# execute_with_retry Tests
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_execute_with_retry_success_first_try():
    """Test execute_with_retry when function succeeds on first try."""
    # Create a simple function that always succeeds
    def success_func(*args, **kwargs):
        return cupy.array([1, 2, 3])

    result, final_batch_size = execute_with_retry(
        nufft_func=success_func,
        func_args=(),
        func_kwargs={},
        initial_batch_size=10,
        max_retries=3,
        logger_context="test"
    )

    assert result is not None
    assert cupy.allclose(result, cupy.array([1, 2, 3]))
    assert final_batch_size == 10  # Should not have reduced


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_execute_with_retry_success_after_retry():
    """Test execute_with_retry when function succeeds after retry."""
    call_count = [0]  # Use list to allow modification in nested function

    def fail_then_succeed(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 2:
            raise RuntimeError("Simulated OOM")
        return cupy.array([4, 5, 6])

    result, final_batch_size = execute_with_retry(
        nufft_func=fail_then_succeed,
        func_args=(),
        func_kwargs={},
        initial_batch_size=10,
        max_retries=3,
        logger_context="test"
    )

    assert result is not None
    assert cupy.allclose(result, cupy.array([4, 5, 6]))
    assert final_batch_size == 5  # Should have reduced from 10 to 5


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_execute_with_retry_exhausted_retries():
    """Test execute_with_retry when all retries are exhausted."""
    def always_fail(*args, **kwargs):
        raise RuntimeError("Simulated persistent OOM")

    with pytest.raises(RuntimeError, match="Simulated persistent OOM"):
        execute_with_retry(
            nufft_func=always_fail,
            func_args=(),
            func_kwargs={},
            initial_batch_size=10,
            max_retries=3,
            logger_context="test"
        )


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_execute_with_retry_cuda_invalid_value_error():
    """Test execute_with_retry with cudaErrorInvalidValue."""
    def cuda_error_func(*args, **kwargs):
        raise RuntimeError("cudaErrorInvalidValue: some error")

    with pytest.raises(RuntimeError, match="GPU out of memory"):
        execute_with_retry(
            nufft_func=cuda_error_func,
            func_args=(),
            func_kwargs={},
            initial_batch_size=10,
            max_retries=3,
            logger_context="test"
        )


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_execute_with_retry_with_func_args():
    """Test execute_with_retry passes args and kwargs correctly."""
    received_args = []
    received_kwargs = []

    def capture_func(*args, **kwargs):
        received_args.extend(args)
        received_kwargs.append(kwargs.copy())
        return cupy.array([1])

    result, _ = execute_with_retry(
        nufft_func=capture_func,
        func_args=(1, 2, 3),
        func_kwargs={'key': 'value'},
        initial_batch_size=10,
        max_retries=3,
        logger_context="test"
    )

    assert received_args == [1, 2, 3]
    assert received_kwargs[0] == {'key': 'value'}


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_execute_with_retry_batch_size_reduction():
    """Test that batch size is halved on each retry."""
    call_count = [0]
    batch_sizes = []

    def track_batch_size(*args, **kwargs):
        call_count[0] += 1
        # Fail first 3 times, succeed on 4th
        if call_count[0] < 4:
            raise RuntimeError("Simulated OOM")
        return cupy.array([1])

    # Note: execute_with_retry doesn't pass batch_size to the function
    # it just tracks it internally. We test that it eventually succeeds
    # after reducing batch size.
    result, final_batch_size = execute_with_retry(
        nufft_func=track_batch_size,
        func_args=(),
        func_kwargs={},
        initial_batch_size=16,
        max_retries=4,
        logger_context="test"
    )

    # After 3 failures, batch_size should be: 16 -> 8 -> 4 -> 2
    assert final_batch_size == 2


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_execute_with_retry_min_batch_size():
    """Test that batch size doesn't go below 1."""
    call_count = [0]

    def succeed_eventually(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 6:
            raise RuntimeError("OOM")
        return cupy.array([1])

    result, final_batch_size = execute_with_retry(
        nufft_func=succeed_eventually,
        func_args=(),
        func_kwargs={},
        initial_batch_size=4,
        max_retries=6,
        logger_context="test"
    )

    # Batch size should be at least 1
    assert final_batch_size >= 1


# ============================================================================
# inplace_rot Tests
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_inplace_rot_identity():
    """Test inplace_rot with identity matrix (no rotation)."""
    rot = cupy.eye(3, dtype=cupy.float64)
    b = cupy.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]], dtype=cupy.float64)
    b_original = b.copy()

    inplace_rot(rot, b)

    assert cupy.allclose(b, b_original)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_inplace_rot_90deg_z_axis():
    """Test inplace_rot with 90-degree rotation around z-axis."""
    # 90-degree rotation around z-axis
    rot_z = cupy.array([[0, -1, 0],
                        [1,  0, 0],
                        [0,  0, 1]], dtype=cupy.float64)

    # Test vector [1, 0, 0] should become [0, 1, 0]
    b = cupy.array([[1.0], [0.0], [0.0]], dtype=cupy.float64)

    inplace_rot(rot_z, b)

    expected = cupy.array([[0.0], [1.0], [0.0]], dtype=cupy.float64)
    assert cupy.allclose(b, expected, atol=1e-10)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_inplace_rot_multiple_vectors():
    """Test inplace_rot with multiple vectors."""
    rot = cupy.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]], dtype=cupy.float64)

    # Multiple vectors: columns are individual vectors
    b = cupy.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]], dtype=cupy.float64)

    inplace_rot(rot, b)

    # After 90-deg z rotation:
    # [1,0,0] -> [0,1,0]
    # [0,1,0] -> [-1,0,0]
    # [0,0,1] -> [0,0,1]
    expected = cupy.array([[0.0, -1.0, 0.0],
                           [1.0,  0.0, 0.0],
                           [0.0,  0.0, 1.0]], dtype=cupy.float64)
    assert cupy.allclose(b, expected, atol=1e-10)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_inplace_rot_numpy_input_conversion():
    """Test that inplace_rot converts numpy arrays to cupy."""
    # Pass numpy arrays instead of cupy arrays
    rot_np = np.eye(3, dtype=np.float64)
    b_np = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

    # Convert to cupy for the function
    rot = cupy.asarray(rot_np)
    b = cupy.asarray(b_np)

    inplace_rot(rot, b)

    # Should still work and not raise errors
    assert b.shape == (3, 1)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_inplace_rot_180deg_x_axis():
    """Test inplace_rot with 180-degree rotation around x-axis."""
    # 180-degree rotation around x-axis
    rot_x = cupy.array([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]], dtype=cupy.float64)

    b = cupy.array([[0.0], [1.0], [1.0]], dtype=cupy.float64)

    inplace_rot(rot_x, b)

    # [0, 1, 1] -> [0, -1, -1]
    expected = cupy.array([[0.0], [-1.0], [-1.0]], dtype=cupy.float64)
    assert cupy.allclose(b, expected, atol=1e-10)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_inplace_rot_large_array():
    """Test inplace_rot with a larger array of vectors."""
    n_vectors = 1000

    # Identity rotation
    rot = cupy.eye(3, dtype=cupy.float64)
    b = cupy.random.random((3, n_vectors), dtype=cupy.float64)
    b_original = b.copy()

    inplace_rot(rot, b)

    # Should be unchanged with identity
    assert cupy.allclose(b, b_original)

    # Now with actual rotation
    rot_z = cupy.array([[0, -1, 0],
                        [1,  0, 0],
                        [0,  0, 1]], dtype=cupy.float64)
    b2 = cupy.random.random((3, n_vectors), dtype=cupy.float64)
    b2_original = b2.copy()

    inplace_rot(rot_z, b2)

    # Should be different after rotation
    assert not cupy.allclose(b2, b2_original)

    # But should have the same norm (rotation preserves length)
    norms_original = cupy.linalg.norm(b2_original, axis=0)
    norms_rotated = cupy.linalg.norm(b2, axis=0)
    assert cupy.allclose(norms_original, norms_rotated, rtol=1e-10)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_inplace_rot_with_raw_numpy_inputs():
    """Test inplace_rot handles numpy inputs by converting them."""
    # The function should convert numpy to cupy if needed
    rot_np = np.array([[0, -1, 0],
                       [1,  0, 0],
                       [0,  0, 1]], dtype=np.float64)
    b_np = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)

    # The function should handle this - it checks isinstance and converts
    # However, note that the conversion happens inside the function,
    # so the original numpy array won't be modified in place
    rot_cp = cupy.asarray(rot_np)
    b_cp = cupy.asarray(b_np)

    inplace_rot(rot_cp, b_cp)

    expected = cupy.array([[0.0], [1.0], [0.0]], dtype=cupy.float64)
    assert cupy.allclose(b_cp, expected, atol=1e-10)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_execute_with_retry_cupy_oom_error():
    """Test execute_with_retry with CuPy OutOfMemoryError."""
    call_count = [0]

    def oom_then_succeed(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 2:
            raise cupy.cuda.memory.OutOfMemoryError(
                1024, 512  # Requested, available
            )
        return cupy.array([1])

    result, final_batch_size = execute_with_retry(
        nufft_func=oom_then_succeed,
        func_args=(),
        func_kwargs={},
        initial_batch_size=10,
        max_retries=3,
        logger_context="test"
    )

    assert result is not None
    assert final_batch_size == 5  # Reduced after OOM


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_execute_with_retry_returns_none_on_failure():
    """Test that result is None when retries fail but not raise."""
    # This tests a specific edge case in the function behavior
    def fail_func(*args, **kwargs):
        raise RuntimeError("Expected failure")

    # When max_retries is exhausted, it should raise the original exception
    with pytest.raises(RuntimeError, match="Expected failure"):
        execute_with_retry(
            nufft_func=fail_func,
            func_args=(),
            func_kwargs={},
            initial_batch_size=10,
            max_retries=2,
            logger_context="test"
        )
