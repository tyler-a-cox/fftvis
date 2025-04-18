# fftvis: A Non-Uniform Fast Fourier Transform-based Visibility Simulator

![Tests](https://github.com/tyler-a-cox/fftvis/actions/workflows/ci.yml/badge.svg)
![codecov](https://codecov.io/gh/tyler-a-cox/fftvis/branch/main/graph/badge.svg)
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)

`fftvis` is a fast Python package designed for simulating interferometric visibilities using the Non-Uniform Fast Fourier Transform (NUFFT). It provides a convenient and efficient way to generate simulated visibilities.

## Features

- Utilizes the Flatiron Institute NUFFT (finufft) [algorithm](https://arxiv.org/abs/1808.06736) for fast visibility simulations that agree with similar methods ([`matvis`](https://github.com/HERA-team/matvis)) to high precision.
- Designed to be a near drop-in replacement to `matvis` with a ~10x improvement in runtime
- Modular architecture with separate core, CPU, and GPU implementations
- Extensible design that allows for easy addition of new backends
- Support for polarized beam patterns

## Current Limitations
- No support for per-antenna beams
- No support for polarized sky emission 
- GPU backend exists only as a stub implementation (coming soon!)
- Diffuse sky models must be pixelized

## Installation

You can install `fftvis` via pip:

```bash
pip install fftvis
```

## Basic Usage

```python
from fftvis import simulate_vis

# Simulate visibilities with the CPU backend (default)
vis = simulate_vis(
    ants=antenna_positions,
    fluxes=source_fluxes,
    ra=source_ra,
    dec=source_dec,
    freqs=frequencies,
    times=observation_times,
    beam=beam_model,
    polarized=True,
    backend="cpu"  # Use "gpu" for GPU acceleration when implemented
)
```

## Architecture

`fftvis` is structured with a modular design:

- **Core**: Contains abstract interfaces and base classes that define the API
- **CPU**: Contains the CPU-specific implementation
- **GPU**: Contains the GPU implementation (currently stubbed for future development)
- **Wrapper**: Provides a high-level API for backward compatibility

This modular design makes the package more maintainable and extensible, allowing for the addition of new backends and optimizations without affecting the user API.

## Contributing
Contributions to `fftvis` are welcome! If you find any issues, have feature requests, or want to contribute improvements, please open an issue or submit a pull request on the GitHub repository: `fftvis` on GitHub

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This package relies on the `finufft` implementation provided by [finufft](https://github.com/flatironinstitute/finufft) library. Special thanks to the contributors and maintainers of open-source libraries used in this project.
