"""Generate oriented-basis matrices with MLX and save core-compatible assets."""

import argparse
import math
from pathlib import Path

import mlx.core as mx

from obf.core import OBMatrix as CoreOBMatrix


DEFAULT_MATRICES_DIR = Path(__file__).resolve().parent / "obf" / "matrices"


class OBMatrixCalculator:
    @staticmethod
    def calculate_matrix(size_order, device=None):
        """Return exact int8 analysis and synthesis matrices of size 3**order.

        Reducing the ternary digit dot product modulo three gives the same
        rounded trigonometric expressions as the original implementation.
        Lookup tables avoid floating-point phase and rounding errors.
        """
        if not isinstance(size_order, int) or isinstance(size_order, bool) or size_order < 1:
            raise ValueError("size_order must be a positive integer.")
        with mx.stream(device if device is not None else mx.default_device()):
            indices = OBMatrixCalculator.generate_indices(3**size_order, device)
            residues = (indices @ indices.T).astype(mx.int32) % 3
            forward = mx.array([1, -2, 1], dtype=mx.int8)[residues]
            inverse = mx.array([1, -1, 0], dtype=mx.int8)[residues]
            mx.eval(forward, inverse)
        return forward, inverse

    @staticmethod
    def generate_indices(size, device=None):
        """Return base-three digits, most significant first, for each index."""
        if not isinstance(size, int) or isinstance(size, bool) or size < 3:
            raise ValueError("size must be a positive power of three, at least 3.")
        remaining, order = size, 0
        while remaining > 1 and remaining % 3 == 0:
            remaining //= 3
            order += 1
        if remaining != 1:
            raise ValueError(f"size must be a power of three, got {size}.")
        with mx.stream(device if device is not None else mx.default_device()):
            values = mx.arange(size, dtype=mx.int32)
            digits = [(values // (3**i)) % 3 for i in range(order - 1, -1, -1)]
            return mx.stack(digits, axis=-1).astype(mx.float32)

    @staticmethod
    def sin_cos_part(sum_product):
        """Evaluate the original trigonometric factors on integer dot products."""
        angles = (sum_product % 3) * (2 * math.pi / 3)
        return mx.cos(angles), mx.sin(angles)


def save_ob_matrices(max_order=7, save_dir=DEFAULT_MATRICES_DIR, device=None, overwrite=False):
    """Save int8 SafeTensor pairs in the format consumed by obf.core."""
    if not isinstance(max_order, int) or isinstance(max_order, bool) or max_order < 1:
        raise ValueError("max_order must be a positive integer.")
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for order in range(1, max_order + 1):
        path = destination / f"ob_mx_{3**order}.safetensors"
        if path.exists() and not overwrite:
            print(f"Skipping existing asset: {path}")
            continue
        forward, inverse = OBMatrixCalculator.calculate_matrix(order, device)
        mx.save_safetensors(str(path), {
            "forward": forward,
            "inverse": inverse,
        })
        print(f"Saved {path}: shape={forward.shape}, storage dtype=int8")


class OBMatrix(CoreOBMatrix):
    """Compatibility adapter for the original order-based calculator API."""

    def __init__(self, size_order, matrices_dir=DEFAULT_MATRICES_DIR, device=None):
        if not isinstance(size_order, int) or isinstance(size_order, bool) or size_order < 1:
            raise ValueError("size_order must be a positive integer.")
        self.device = device if device is not None else mx.default_device()
        with mx.stream(self.device):
            super().__init__(size=3**size_order, matrices_dir=str(matrices_dir))

    @property
    def forward_matrix(self):
        return self._matrix["forward"]

    @property
    def inverse_matrix(self):
        return self._matrix["inverse"]

    def __call__(self, x: mx.array):
        with mx.stream(self.device):
            return super().__call__(x)

    def forward(self, x: mx.array):
        return self(x)

    def inverse(self, spectrum: mx.array):
        with mx.stream(self.device):
            return super().inverse(spectrum)

    def inverse_transform(self, spectrum: mx.array):
        return self.inverse(spectrum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, required=True, help="Generate one matrix pair of size 3**order.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MATRICES_DIR)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.output_dir / f"ob_mx_{3**args.order}.safetensors"
    if destination.exists() and not args.overwrite:
        parser.error(f"Asset already exists: {destination}. Use --overwrite to replace it.")
    forward, inverse = OBMatrixCalculator.calculate_matrix(args.order, getattr(mx, args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(destination), {"forward": forward, "inverse": inverse})
    print(f"Saved {destination}: shape={forward.shape}, storage dtype=int8")
