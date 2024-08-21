import numpy as np

def simulate(
    flux: np.ndarray,
    crd_eq: np.ndarray,
    eq2top: np.ndarray,
    chunk_size: int | None = None,
    source_buffer: float = 0.55,
    precision: int = 1,
    gpu: bool = False,
):
    pass