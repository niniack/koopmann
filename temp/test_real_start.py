"""Checks the `real_start` option and the overlap loss on random data, on the CPU."""
import sys

import torch

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train  # noqa: E402

torch.manual_seed(0)
kwargs = dict(in_features=64, latent_features=33, hidden_config=[128], bias=True, batchnorm=False, nonlinearity="leaky_relu")
states = torch.randn(32, 2, 64)

# Old behavior is untouched when the option is off
torch.manual_seed(1)
old = train.JordanKoopmanAutoencoder(8, shared_modes=True, **kwargs)
torch.manual_seed(1)
new = train.JordanKoopmanAutoencoder(8, shared_modes=True, real_start=True, **kwargs)
phi_old, phi_new = train.encode_pairs(old, states), train.encode_pairs(new, states)
raw = train.Autoencoder.encode(old, states.flatten(0, 1))
blocks = raw[:, 1:].unflatten(-1, (-1, 2))
magnitudes = torch.cat([raw[:, :1].abs(), blocks.norm(dim=-1)], dim=-1)[::2].repeat_interleave(2, dim=0)
keep = magnitudes >= magnitudes.topk(4, dim=-1).values[:, -1:]
expected = torch.cat([raw[:, :1] * keep[:, :1], (blocks * keep[:, 1:, None]).flatten(1)], dim=-1).unflatten(0, (32, 2))
assert torch.equal(phi_old, expected), "option off changed the encoding"

# Option on: start states have no second number, end states do, and both ends keep the same modes
pairs = phi_new[..., 1:].unflatten(-1, (-1, 2))
assert (pairs[:, 0, :, 1] == 0).all(), "start has a second number"
assert (pairs[:, 1, :, 1] != 0).any(), "end lost its second number"
kept = torch.cat([phi_new[..., :1] != 0, pairs.norm(dim=-1) != 0], dim=-1)
assert torch.equal(kept[:, 0], kept[:, 1]) and (kept[:, 0].sum(-1) == 4).all(), "modes differ between ends"
assert (new.encode(states[:, 0])[:, 1:].unflatten(-1, (-1, 2))[..., 1] == 0).all(), "start alone has a second number"
try:
    train.JordanKoopmanAutoencoder(8, real_start=True, **kwargs)
    raise AssertionError("no error without shared modes")
except ValueError:
    pass

# Overlap loss: zero when the latent is a rotated copy of the states, positive otherwise, and it reaches the eigenvalues
rotation = torch.linalg.qr(torch.randn(64, 64)).Q
assert train.compute_overlap_loss(states, states @ rotation) < 1e-8
assert train.compute_overlap_loss(states, phi_new) > 0.1


class Head:
    classes = None

    def embed(self, x):
        return torch.nn.functional.normalize(x[..., :16], dim=-1)


losses = train.compute_losses(new, states, 0.5, Head())
assert len(losses) == 9
sum(losses).backward()
assert new.block_imag.grad.abs().sum() > 0, "no gradient to the turn"
print("all checks passed;", "losses:", [round(v.item(), 4) for v in losses])
