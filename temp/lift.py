"""
TEMP: a KAE whose encoder reads the patch tokens together with the CLS token (arms `attend` and `mixer` of cls_train.py).

    z6 = encoder(CLS token, patch tokens after block 6)        z12 = λ·z6        CLS token after block 12 = decoder(z12)

There is no control: nothing is added after the eigenvalue step. The same encoder is used after block 12 (for the
latent loss), and the decoder gives the CLS token only, so encoder and decoder are not mirror images.

The CLS token is in charge of the reading. A linear layer writes it into 16 slots, and each slot then adds what it
reads from the 49 patch tokens. The encoder's first layer is widened to take the CLS token and the 16 slots. Two readers:
    attend   one cross-attention layer: each slot is a query over the patch tokens
    mixer    the token-mixing MLP of one MLP-Mixer block (Tolstikhin et al., arXiv 2105.01601) over slots and patch tokens
In both, the encoder's own MLP follows directly, as the MLP of a transformer block or the channel-mixing MLP of a Mixer block does.

With `reader_layers` above 1 (arm `attend_deep`), the attention reader reads again: each further layer queries the patch
tokens with what the slots have read so far, then runs an MLP on the slots (a Perceiver block, Jaegle et al., arXiv 2103.03206).

With `linear_decoder` (arms `linear` and `linear_real`) the decoder is one matrix and a bias in place of its MLP. A mode then
writes the same two directions of the CLS token after both blocks, so the change from block 6 to block 12 is the decoder's
matrix times (λ - 1)·z6: the MLP decoder can no longer turn a rescaled z6 into another state.
"""

import torch
import torch.nn as nn

import control_train as base
from control_train import train
from koopmann.models.layers import LinearLayer


class Reader(nn.Module):
    """What the readers share: the slots the CLS token writes, and the patch tokens embedded with their positions."""

    def __init__(self, width, num_patches, dim=512, slots=16):
        super().__init__()
        self.dim, self.slots, self.out_features = dim, slots, slots * dim
        self.write = nn.Linear(width, slots * dim)
        self.norm = nn.LayerNorm(width)
        self.embed = nn.Linear(width, dim)
        # NOTE: 3D, so train.get_optimizers gives these to AdamW (Muon is for weight matrices)
        self.positions = nn.Parameter(0.02 * torch.randn(1, num_patches, dim))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, cls, patches):
        # shape: [images, slots, dim] and [images, patches, dim]
        slots = self.write(cls).unflatten(-1, (self.slots, self.dim))
        tokens = self.embed(self.norm(patches)) + self.positions

        return self.out_norm(slots + self.read(slots, tokens)).flatten(1)


class AttendReader(Reader):
    """
    Each slot is a query of one cross-attention layer over the patch tokens (control_train's attention).
    NOTE: With `layers` above 1, the further layers are in `more`, so a one-layer reader keeps its weight names.
    """

    def __init__(self, width, num_patches, heads=8, layers=1):
        super().__init__(width, num_patches)
        self.heads = heads
        self.to_query, self.to_key, self.to_value = nn.Linear(self.dim, self.dim), nn.Linear(self.dim, self.dim), nn.Linear(self.dim, self.dim)
        self.more = nn.ModuleList([ReadAgain(self.dim, heads) for _ in range(layers - 1)])

    def read(self, slots, tokens):
        read = base.attend(self.to_query(slots), self.to_key(tokens), self.to_value(tokens), self.heads)
        for layer in self.more:
            read = read + layer(slots + read, tokens)

        return read


class ReadAgain(nn.Module):
    """One further layer of the attention reader: the slots query the patch tokens again, then an MLP runs on each slot."""

    def __init__(self, dim, heads, hidden=1_024):
        super().__init__()
        self.heads = heads
        self.norm, self.mlp_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_query, self.to_key, self.to_value = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, slots, tokens):
        query = self.to_query(self.norm(slots))
        read = base.attend(query, self.to_key(tokens), self.to_value(tokens), self.heads)

        return read + self.mlp(self.mlp_norm(slots + read))


class MixerReader(Reader):
    """
    The token-mixing MLP of an MLP-Mixer block over the slots and the patch tokens: one MLP across the tokens, the same
    for every channel. Its hidden size is the paper's for 49 patches (Mixer-B/32).
    NOTE: A Mixer block also updates the patch tokens. Nothing reads them after this one block, so only the slots are computed.
    """

    def __init__(self, width, num_patches, hidden=384, layers=1):
        super().__init__(width, num_patches)
        assert layers == 1, "the mixer reader has one layer"
        self.mix_norm = nn.LayerNorm(self.dim)
        self.mix = nn.Sequential(nn.Linear(self.slots + num_patches, hidden), nn.GELU(), nn.Linear(hidden, self.slots))

    def read(self, slots, tokens):
        # shape: [images, dim, slots + patches]
        both = self.mix_norm(torch.cat([slots, tokens], dim=1)).transpose(1, 2)

        return self.mix(both).transpose(1, 2)


READERS = {"attend": AttendReader, "mixer": MixerReader}


class LiftKae(train.JordanKoopmanAutoencoder):
    """
    train.py's Jordan KAE with a reader in front of its encoder (see the top of this file).
    NOTE: `set_control` has to be called with the patch tokens of the states that `encode` gets next. The name is
    ControlKae's, so the training and test code is shared, but these patch tokens are part of the state, not a control.
    """

    signal = None  # No control. angle_tests.ablate reads this.

    def __init__(self, reader, width, num_patches, linear_decoder=False, reader_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.reader = READERS[reader](width, num_patches, layers=reader_layers)
        # The encoder's first layer, widened to take the CLS token and the slots
        first = LinearLayer(width + self.reader.out_features, self.hidden_config[0], self.bias, self.batchnorm, self.nonlinearity)
        first.apply(LinearLayer.init_weights)
        self.components.encoder.encoder_0 = first
        if linear_decoder:
            self.components.decoder = self._build_decoder([(self.in_features, self.latent_features)])

    def set_control(self, patches):
        """Patch tokens [images, patches, width], or [images, 2, patches, width] for start and end states encoded as pairs."""

        self.patches = patches.flatten(0, -3)

    def encode(self, x, pairs=False, chunk=4_096):
        assert len(x) == len(self.patches), "set_control has to be given the patch tokens of these states"
        slots = torch.cat([self.reader(cls, tokens.float()) for cls, tokens in zip(x.split(chunk), self.patches.split(chunk))])

        return super().encode(torch.cat([x, slots], dim=-1), pairs)
