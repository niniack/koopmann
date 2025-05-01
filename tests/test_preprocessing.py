import pytest
import torch

from scripts.train_ae.shape_metrics import Processor


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.5, 0.8, 1.0])
def test_whitening(alpha):
    x = torch.randn(100, 500)

    x_whitened, e, v = Processor._whiten(x, alpha=alpha)
    x_unwhitened = Processor._unwhiten(x_whitened, e, v, alpha=alpha)

    torch.testing.assert_close(x, x_unwhitened, rtol=100, atol=1e-2)


def test_dim_red():
    x = torch.zeros(1000, 5000)
    # Only fill the first 100 columns with random data
    x[:, :100] = torch.randn(1000, 100)

    x_red, RSV = Processor._dim_reduce_svd(x, dim=100)
    x_recon = Processor._dim_restore_svd(x_red, RSV)

    # The reconstruction should be very close to the original
    torch.testing.assert_close(x, x_recon, rtol=1e-5, atol=1e-5)

    assert x_red.shape == (1000, 100)
    assert RSV.shape == (100, 5000)
