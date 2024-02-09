from . import cpu
from . import gpu


def simulate_cpu(sources, freqs**kwargs):
    return cpu.simulate(sources, **kwargs)

def simulate_gpu(sources, **kwargs):
    return gpu.simulate(sources, **kwargs)

def simulate_vis(sources, , **kwargs):
    return simulate_cpu(sources, **kwargs)