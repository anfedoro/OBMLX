"""Numerical checks for published matrix assets, generators and notebook baselines."""

import ast
import math
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import nbformat
import numpy as np
import pytest
from scipy import fft as spfft

from ob import OBMatrixCalculator
from obf.core import OBMatrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("order", range(1, 8))
def test_assets_and_round_trip(order):
    n = 3**order
    arrays = mx.load(str(ROOT / "obf/matrices" / f"ob_mx_{n}.safetensors"))
    generated = OBMatrixCalculator.calculate_matrix(order)
    for key, expected, alphabet in zip(
        ["forward", "inverse"], generated, [{-2, 1}, {-1, 0, 1}], strict=True
    ):
        actual = np.asarray(arrays[key])
        assert actual.shape == (n, n)
        assert actual.dtype in [np.dtype("int8"), np.dtype("float32")]
        assert set(np.unique(actual)) <= alphabet
        np.testing.assert_array_equal(actual, np.asarray(expected))
    transform = OBMatrix(n, matrices_dir=str(ROOT / "obf/matrices"))
    x = np.random.default_rng(42 + n).normal(size=(8, n)).astype(np.float32)
    restored = np.asarray(transform.inverse(transform(mx.array(x))))
    assert np.linalg.norm(restored - x) / np.linalg.norm(x) < 3e-5
    dc = np.asarray(transform(mx.ones((1, n))))
    np.testing.assert_allclose(dc, np.eye(1, n), atol=3e-6)
    if n <= 81:
        f, b = [np.asarray(arrays[k]).astype(np.float64) for k in ["forward", "inverse"]]
        np.testing.assert_allclose(f @ b / n, np.eye(n), atol=1e-14)
        np.testing.assert_allclose(b @ f / n, np.eye(n), atol=1e-14)


def baseline_definitions():
    names = {"fourier_forward", "fourier_inverse", "haar_forward", "haar_inverse"}
    namespace = {"np": np, "spfft": spfft, "math": math, "DTYPE": np.dtype("float32")}
    notebook = nbformat.read(ROOT / "ob_transform_comparison.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("n", [27, 81, 243, 729, 2187])
def test_orthonormal_baselines(n):
    functions = baseline_definitions()
    x = np.random.default_rng(n).normal(size=(8, n)).astype(np.float32)
    for forward, inverse in [
        (functions["fourier_forward"], functions["fourier_inverse"]),
        (functions["haar_forward"], functions["haar_inverse"]),
        (lambda z: spfft.dct(z, type=2, norm="ortho"), lambda z: spfft.idct(z, type=2, norm="ortho")),
    ]:
        c = forward(x)
        restored = inverse(c)
        assert c.shape == x.shape and not np.iscomplexobj(restored)
        np.testing.assert_allclose(restored, x, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(np.linalg.norm(c, axis=1), np.linalg.norm(x, axis=1), rtol=5e-6)


def test_documented_example():
    transform = OBMatrix(size=27, matrices_dir=str(ROOT / "obf/matrices"))
    x = mx.arange(54, dtype=mx.float32).reshape(2, 27) / 54
    error = mx.max(mx.abs(transform.inverse(transform(x)) - x))
    mx.eval(error)
    assert error.item() < 1e-5


def test_generator_cli(tmp_path):
    command = [sys.executable, str(ROOT / "ob.py"), "--order", "3", "--output-dir", str(tmp_path), "--device", "cpu"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    path = tmp_path / "ob_mx_27.safetensors"
    assert mx.load(str(path))["forward"].dtype == mx.int8
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "already exists" in result.stderr


@pytest.mark.parametrize("order", [0, -1, True, 1.5])
def test_invalid_generator_order(order):
    with pytest.raises(ValueError):
        OBMatrixCalculator.calculate_matrix(order)


@pytest.mark.parametrize("filename", [
    "core_distribution_experiment.ipynb",
    "ob_transform_comparison.ipynb",
    "ob_piecewise_linear_experiment.ipynb",
])
def test_published_notebook(filename):
    import json
    import re

    notebook = nbformat.read(ROOT / filename, as_version=4)
    nbformat.validate(notebook)
    counts = []
    figures = 0
    for cell in notebook.cells:
        assert not re.search("[А-Яа-яЁё]", cell.source)
        if cell.cell_type == "code":
            counts.append(cell.execution_count)
            for output in cell.outputs:
                assert output.output_type != "error"
                assert not re.search("[А-Яа-яЁё]", json.dumps(output, ensure_ascii=False))
                figures += "image/png" in output.get("data", {})
    assert counts == list(range(1, len(counts) + 1))
    assert figures >= 2
