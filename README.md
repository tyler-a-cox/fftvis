# fftvis: A Non-Uniform Fast Fourier Transform-based Visibility Simulator

![Tests](https://github.com/tyler-a-cox/fftvis/actions/workflows/ci.yml/badge.svg)
![codecov](https://codecov.io/gh/tyler-a-cox/fftvis/branch/main/graph/badge.svg)
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)

`fftvis` is a fast Python package designed for simulating interferometric visibilities using the Non-Uniform Fast Fourier Transform (NUFFT). It provides a convenient and efficient way to generate simulated visibilities.

## Features

- Utilizes the Flatiron Institute NUFFT (finufft) [algorithm](https://arxiv.org/abs/1808.06736) for fast visibility simulations that agree with similar methods ([`matvis`](https://github.com/HERA-team/matvis)) to high precision.
- Designed to be a near drop-in replacement to `matvis` with a ~10x improvement in runtime

## Limitations
- Currently no support for per-antenna beams
- Currently no support for polarized sky emission
- Diffuse sky models must be pixelized

## Installation

You can install `fftvis` via pip:

```bash
pip install fftvis
```

### Installation for GPU

Clone `finufft` repository
```
git clone https://github.com/flatironinstitute/finufft
cd finufft
```

Build `finufft` for `CUDA`
```bash
mkdir build
cd build
cmake -D FINUFFT_USE_CUDA=ON ..
cmake --build .
```

Install `Python` installation
```bash
pip install python/cufinufft
```

## Contributing
Contributions to `fftvis` are welcome! If you find any issues, have feature requests, or want to contribute improvements, please open an issue or submit a pull request on the GitHub repository: `fftvis` on GitHub

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This package relies on the `finufft` implementation provided by [finufft](https://github.com/flatironinstitute/finufft) library. Special thanks to the contributors and maintainers of open-source libraries used in this project.
