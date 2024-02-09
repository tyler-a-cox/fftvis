from . import cpu
from . import gpu

def simulate_cpu(data, **kwargs):
    return cpu.simulate(data, **kwargs)

def simulate_gpu(data, **kwargs):
    return gpu.simulate(data, **kwargs)

def simulate_vis(data, **kwargs):
    return simulate_cpu(data, **kwargs)