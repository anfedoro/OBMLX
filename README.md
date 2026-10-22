# OBMLX

## Overview

OBMLX is an experimental implementation of the Oriented Basis Transform using Apple's MLX framework. The project explores the transform's numerical properties and compares its behavior with established transforms on synthetic signals. The experiments identify several interesting structural properties but do not demonstrate a general practical advantage.

The project began by exploring OB as a basis transformation within deep neural networks. No compelling application was established. The repository now documents the implementation, numerical experiments and their limitations. The legacy activation modules in `obf/obaf.py` are retained as experimental code; they are not a validated neural architecture.

## Reference

The implementation follows the mathematical description in:

Julia Yamnenko, Tetyana Tereshchenko, Yuriy Khokhlov, Oleksandr Bondarenko, Artem Polishuk. *Data Transmission with Code Division Based on Transform in Rotating Coordinates.* 2017 IEEE 58th International Scientific Conference on Power and Electrical Engineering of Riga Technical University (RTUCON). [DOI: 10.1109/RTUCON.2017.8124839](https://doi.org/10.1109/RTUCON.2017.8124839).

The OB transform is not introduced by this project. The reference article is not distributed with the repository.

## Implementation

`obf.core.OBMatrix` applies saved analysis and synthesis matrices at ternary dimensions `N = 3**m`. For row-oriented signals, forward is `X @ forward / N` and inverse is `C @ inverse`, without another scale factor. This is a biorthogonal transform, not an orthonormal one.

Included matrix sizes are 3, 9, 27, 81, 243, 729 and 2187. Analysis entries are 1 or −2; synthesis entries are 0 or ±1. Existing assets retain float32 storage except the recovered 243 pair, stored as int8. The deterministic MLX generator writes int8 SafeTensors; application casts weights to the input dtype. The historical misnamed 81×81 asset for size 243 has been replaced with its verified 243×243 copy.

## Installation

MLX execution requires **Apple Silicon and macOS**. The locked environment uses Python **3.13.15**. Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```sh
uv sync --frozen
uv run pytest
```

The environment includes MLX, NumPy, SciPy, Matplotlib, Jupyter and experiment/test dependencies. No PyTorch installation is needed.

## Usage

Run from the repository root, where the default matrix directory is resolved:

```python
import mlx.core as mx
from obf.core import OBMatrix

transform = OBMatrix(size=27)
x = mx.arange(54, dtype=mx.float32).reshape(2, 27) / 54
coefficients = transform(x)
restored = transform.inverse(coefficients)
error = mx.max(mx.abs(restored - x))
mx.eval(error)
assert error.item() < 1e-5
```

Generate a pair of size `3**6 = 729` in a separate directory:

```sh
uv run python ob.py --order 6 --output-dir generated_matrices --device gpu
```

The command refuses to replace an existing file unless `--overwrite` is supplied. To load the generated pair, pass `matrices_dir="generated_matrices"` to `OBMatrix`. Generation is dense and needs quadratic memory; arbitrary large orders are not practical.

## Experiments

| Notebook | Purpose | Saved results |
| --- | --- | --- |
| [Distribution experiment](core_distribution_experiment.ipynb) | Forward/inverse effects on Gaussian and Uniform inputs | Figures and statistics in the notebook |
| [Transform comparison](ob_transform_comparison.ipynb) | OB versus real Fourier, DCT-II and Ternary Haar; sparse reconstruction and denoising | [CSV summaries and figures](ob_transform_comparison_results/) |
| [Piecewise-linear experiment](ob_piecewise_linear_experiment.ipynb) | Boundary alignment, direct fitting, validation-selected denoising and storage costs | [CSV summaries and figures](ob_piecewise_linear_results/) |

Open with `uv run jupyter lab`, or execute from a clean kernel in this order:

```sh
uv run jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 core_distribution_experiment.ipynb
uv run jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 ob_transform_comparison.ipynb
uv run jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 ob_piecewise_linear_experiment.ipynb
```

Configuration cells control seeds, dimensions and budgets. The last notebook reuses selected definitions from the comparison notebook and its recovered 243 asset. Runs regenerate CSVs, figures and metadata; timing and allocator measurements vary by machine and system load.

## Findings

- Linear trends have compact OB representations. A nonconstant affine signal with nonzero mean at `N=3**m` uses `m+1` coefficients; centered trends use `m`. The piecewise-linear notebook derives the formula and checks exact integer products and float64 residuals.
- Piecewise-linear compactness strongly depends on ternary-aligned boundaries. At N=729 and K=146, mean OB error for aligned discontinuous signals is about 0.00578; shifting the boundaries by 1–5 samples raises it to about 0.11184. Ternary Haar is more robust to these shifts, while DCT-II is better on average for continuous signals.
- Comparisons with real Fourier, DCT-II and Ternary Haar show no general OB superiority in reconstruction or denoising. Direct piecewise-linear fitting is a strong baseline for this known signal family. Gaussian-looking coordinate marginals after mixing do not establish joint normality or independence.
- Compact matrix storage is different from computational performance. Int8 cuts matrix file payload by four relative to float32; current float32 matrix multiplication still requires casts and dense work. Two-bit storage and faster factorization are theoretical estimates, not implemented kernels or measured speedups.

## Limitations and license

These are synthetic experiments with one seed and specific coefficient-selection policies, not evidence of a neural-network benefit. OB coefficients are ranked by synthesis-component norm; this is not globally optimal biorthogonal sparse approximation. The current dense implementation has quadratic matrix storage and computation per signal. The windowing and legacy activation utilities have less validation than the matrix interface; use the latter for the documented experiments.

See [publication checks](VALIDATION.md) for validation details. Original project code, generated matrices and experiment artifacts are provided under the [MIT license](LICENSE). The cited publication and dependencies retain their own rights and licenses.
