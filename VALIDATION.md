# Publication checks

Checked on 2026-10-22 on Apple M4 Max, macOS, Python 3.13.15 and MLX 0.32.4.

- `uv sync --frozen` succeeded in a new environment. An export containing only publication files also installed successfully, without the original local environment, reference PDFs or redundant Torch assets.
- The README Python example and matrix-generation command passed in that clean export.
- All 21 pytest checks passed, including notebook schema, execution and English-output checks.
- All seven stored matrix pairs (3 through 2187) matched the deterministic generator exactly. Dimensions, coefficient alphabets, float32 round trips, constant signals and small-matrix inverse products were checked.
- Real Fourier, DCT-II and Ternary Haar passed reconstruction and norm-preservation checks. The notebooks also check full-budget reconstruction, matrix orientations and small explicit orthonormal bases.
- All three notebooks executed from fresh kernels using the locked environment. The comparison and piecewise-linear experiments include N=729 and N=2187; the latter also covers 81 and 243. Saved figures and visible text are English.
- All 32 deterministic numerical CSVs were byte-identical between two full runs. Loading, timing, allocator and process-memory measurements are excluded from equality claims because they depend on system state.
- Local Markdown links, notebook schemas, execution counts, saved outputs, publication language and embedded figures were checked. No reference PDF or local credential material is included.

The historical malformed 243 asset was replaced with the verified int8 recovery. Other matrix values and transform formulas were preserved. Tests focus on the matrix interface and experiment baselines; the legacy windowing/activation utilities are not covered comprehensively. Numerical tolerances and methodological limits are stated in the notebooks. No advantage on real data or neural-network workloads was tested.

## Backend dependency update

The environment now selects `mlx`/Metal on macOS arm64 and `mlx[cuda12]` on Linux x86_64/aarch64. Frozen sync succeeded on the Apple Silicon host. Cross-platform frozen dry runs selected `mlx-metal` for macOS and `mlx-cuda-12` plus NVIDIA libraries for both Linux architectures. These checks validate package selection, not CUDA execution. No NVIDIA hardware was available; the published experiment results remain Apple Silicon measurements. OS/architecture markers do not detect installed GPUs or drivers.
