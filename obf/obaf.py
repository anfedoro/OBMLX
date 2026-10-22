import mlx.core as mx
import mlx.nn as nn
from obf.core import OBTransform


class QubicPolyLayer(nn.Module):
    def __init__(self, features: int, win: int = 9):
        super().__init__()
        self.features = features
        self.win = win
        # Create weights using proper MLX parameter initialization
        self.weights = mx.random.normal((1, self.features, 1, win, 4)) * 0.1
        self.normalize = nn.LayerNorm((self.features), bias=False)

    def __call__(self, x: mx.array):
        """
        x: Tensor of shape (batch, features, patches, win)
        """
        w = self.weights
        A = w[..., 3]  # cubic term
        B = w[..., 2]
        C = w[..., 1]
        D = w[..., 0]
        y = ((A * x + B) * x + C) * x + D
        y = self.normalize(y)
        return y


class QubOBAF(nn.Module):
    """
    Legacy experimental OB activation module with cubic polynomial transformation.
    """

    def __init__(self, inp_dim, out_dim, **kwargs):
        super().__init__()
        self.size = kwargs.get("size", 9)
        self.stride = kwargs.get("stride", self.size // 3)
        self.dropout_rate = kwargs.get("dropout", 0.02)
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Create weight using proper MLX parameter initialization
        self.ob_weight = mx.random.normal((self.out_dim, 1, self.size)) * 0.1
        self.qpoly = QubicPolyLayer(self.out_dim, win=self.size)
        self.ob_transform = OBTransform(self.size, self.stride)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x):
        ob = self.ob_transform(x)
        ob_amp = mx.einsum("...ius,ous->...ious", ob, self.ob_weight)
        ob_qpoly = self.qpoly(ob_amp)
        out = self.ob_transform.inverse(ob_qpoly)
        ob_amp = self.dropout(ob_amp)
        out = out.sum(axis=-3)
        return out
