# fftvis: A Non-Uniform Fast Fourier Transform-based Visibility Simulator

![Tests](https://github.com/tyler-a-cox/fftvis/actions/workflows/ci.yml/badge.svg)
![codecov](https://codecov.io/gh/tyler-a-cox/fftvis/branch/main/graph/badge.svg)
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)

`fftvis` is a fast Python package designed for simulating interferometric visibilities using the Non-Uniform Fast Fourier Transform (NUFFT). It provides a convenient and efficient way to generate simulated visibilities.

## Features

- Utilizes the Flatiron Institute NUFFT (finufft) [algorithm](https://arxiv.org/abs/1808.06736) for fast visibility simulations that agree with similar methods ([matvis](https://github.com/HERA-team/matvis)) to nearly machine precision.

## Limitations
- Currently no support for per-antenna beams
- Currently no support for polarized sky emission 
- Currently no GPU support
- Diffuse sky models must be pixelized

## Installation

You can install `fftvis` via pip:

```bash
git clone https://github.com/tyler-a-cox/fftvis
cd fftvis
pip install .
```

## Contributing
Contributions to `fftvis` are welcome! If you find any issues, have feature requests, or want to contribute improvements, please open an issue or submit a pull request on the GitHub repository: `fftvis` on GitHub

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This package relies on the `finufft` implementation provided by [finufft](https://github.com/flatironinstitute/finufft) library. Special thanks to the contributors and maintainers of open-source libraries used in this project.
