import os
import math
import mlx.core as mx
import mlx.nn as nn

import matplotlib.pyplot as plt


class OBMatrix(nn.Module):
    def __init__(self, size, matrices_dir="obf/matrices"):
        super(OBMatrix, self).__init__()
        self.size = size
        self.rank = math.log(size, 3)
        self._matrix = None
        self.load_matrices(matrices_dir)

    def load_matrices(self, matrices_dir):
        path = os.path.join(matrices_dir, f"ob_mx_{self.size}.safetensors")

        if os.path.exists(path):
            self._matrix = mx.load(path)

        else:
            raise FileNotFoundError(f"Pre-calculated matrices for size {self.size} not found.")

    def __call__(self, x: mx.array):
        out = mx.matmul(x, self._matrix["forward"].astype(x.dtype)) * (1 / self.size)
        return out

    def inverse(self, spectrum: mx.array):
        out = mx.matmul(spectrum, self._matrix["inverse"].astype(spectrum.dtype))
        return out


class OBTransform(nn.Module):
    def __init__(self, size=3, stride=None):
        super(OBTransform, self).__init__()
        self.name = "OBTransform"
        self.obm = OBMatrix(size)
        self.window_size = size
        self.stride = stride if stride is not None else self.window_size // 3

    def unfold(self, x):
        length = x.shape[-1]
        win = self.window_size
        step = self.stride
        self.pad = win - (length - ((length - win) // step + 1) * step)  # leave padding only min required paddings, only to keep the last window with tailing 0
        padding = [(0, 0)] * (len(x.shape) - 1) + [(0, self.pad)]  # pad only the last dimension
        x_padded = mx.pad(x, padding, mode="edge")
        self.new_length = length + self.pad
        num_windows = (self.new_length - win) // step + 1

        if num_windows <= 0:
            raise ValueError(f"Cannot create windows: length={self.new_length}, kernel_size={win}, stride={step}")

        base_indices = mx.arange(win)
        window_starts = mx.arange(num_windows) * step
        self._indices = window_starts[:, None] + base_indices[None, :]

        result = x_padded[..., self._indices]

        return result

    def fold(self, x):
        pad = self.get_pad()
        win = self.window_size
        step = self.stride
        leading_dim = list(x.shape[:-2])
        tailing_dim = list(x.shape[-2:])
        x = x.reshape(-1, *tailing_dim)
        weight = mx.ones([1, win, win])
        x_fold_sum = mx.conv_transpose1d(x, weight, stride=step)
        # 1.	Порядок каналов в input
        # PyTorch: (batch, C_in, length)
        # MLX:     (batch, length, C_in)
        # 	2.	Порядок осей в weight
        # PyTorch: (C_in, C_out, kernel)
        # MLX:     (C_out, kernel, C_in)

        ones = mx.ones_like(x)
        overlap_count = mx.conv_transpose1d(ones, weight, stride=step)
        overlap_count[..., -(win):] = overlap_count[..., -(win):] - pad  # exclude last pads from count

        x_restored = x_fold_sum / overlap_count
        x_restored = x_restored.reshape(*leading_dim, -1)[..., :-pad]
        return x_restored

    def fold_fast(self, x_patches):
        lead_shape = x_patches.shape[:-2]
        patch_shape = x_patches.shape[-2:]  # (P, W)
        num_leads = math.prod(lead_shape)
        P, W = patch_shape
        original_length = self.new_length
        indx = self._indices

        # reshape inputs
        a_flat = x_patches.reshape(num_leads, -1)  # [N, P*W]
        # idx_flat = indx.reshape(1, -1).((num_leads, P * W))  # [N, P*W]
        idx_flat = mx.repeat(indx.reshape(1, -1), num_leads, axis=0)  # [N, P*W]
        # flatten for scatter
        a_vals = a_flat.reshape(-1)
        idx_vals = idx_flat.reshape(-1)

        # prepare accumulators
        accum = mx.zeros((num_leads, original_length), dtype=x_patches.dtype)
        count = mx.zeros((num_leads, original_length), dtype=x_patches.dtype)

        # scatter-add
        accum = accum.at[(mx.arange(a_vals.size) // (P * W), idx_vals)].add(a_vals)
        count = count.at[(mx.arange(a_vals.size) // (P * W), idx_vals)].add(mx.ones_like(a_vals))

        # avoid division by zero
        out = accum / mx.clip(count, 1, None)

        out = out.reshape(*lead_shape, original_length)
        if self.get_pad() > 0:
            out = out[..., : -self.get_pad()]

        return out

    def bq_fold(self, x):
        l_idx = self.window_size // 3
        r_idx = l_idx * 2
        x[..., 1:, l_idx] = (x[..., 1:, l_idx] + x[..., :-1, r_idx]) * 0.5
        targed_shape = list(x.shape[:-2]) + [x.shape[-2] + 2, l_idx]
        target = mx.zeros(targed_shape, dtype=x.dtype)
        target[..., :1, :] = x[..., :1, :l_idx]
        target[..., -1:, :] = x[..., -1:, r_idx:]
        target[..., 1:-1, :] = x[..., :, l_idx:r_idx]
        if self.get_pad() > 0:
            target = target.reshape(*targed_shape[:-2], -1)
            target = target[..., : -self.get_pad()]  # remove last padding
        else:
            target = target.reshape(x.shape[:-2], -1)
        return target

    def get_pad(self) -> int:
        return self.pad

    def __call__(self, x):
        x = self.unfold(x)
        spectr = self.obm(x)
        return spectr

    def inverse(self, spectr):
        x = self.obm.inverse(spectr)
        x = self.fold_fast(x)
        return x


############## Test routines ##############


def main(matrix_size=9):
    # force mlx output in non scientific format

    import matplotlib.pyplot as plt

    obm = OBMatrix(size=matrix_size)
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111)
    ax.matshow(obm._matrix["forward"], cmap="viridis")
    ax.set_title(f"OB matrix for size {matrix_size}")
    ax.set_xlabel("Input dimension")
    ax.set_ylabel("Output dimension")

    plt.show()
    print(f"OB matrix for size {matrix_size}:\n", obm._matrix["forward"])
    print(f"Inverse OB matrix for size {matrix_size}:\n", obm._matrix["inverse"])

    # transform test
    size = matrix_size
    ob_transform = OBTransform(size=size)
    X = mx.sin(mx.random.normal([1, 500]))
    print(f"Original X:\n{X}\nShape: {X.shape}")

    transformed = ob_transform(X)
    print(f"Transformed X:\n{transformed}\nShape: {transformed.shape}")
    restored = ob_transform.inverse(transformed)
    print(f"Restored X:\n{restored}\nShape: {restored.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot the ON matrix")
    parser.add_argument("--size", type=int, default=27, help="Size of the OB matrix")
    args = parser.parse_args()
    main(args.size)
