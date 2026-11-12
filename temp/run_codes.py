"""
TEMP: what the mode checks written for train.py's runs (name_modes.py, angle_vs_class.py, ms_score.py) need from a
model of cls_train.py, on ImageNet validation. Those scripts call this when the run's stats name an arm.
"""

import torch

import angle_tests as tests
import baseline_eval as fixed
from angle_tests import base, device, train


def get_codes(name):
    """
    Returns the model (on `device`) and, on the CPU: the CLS token after blocks 6 and 12 as CLIP gives it [images, 2, width],
    the labels, every mode's value in the encoder's code of each of the two [images, 2, 1025] (complex), and λ [1025].
    NOTE: A reader gets each block's own patch tokens, so the block-12 code is its code of the real block-12 state.
    """

    model, stats = tests.load_kae(name.partition("_step")[0], name)
    raw, patches, labels = base.get_val(None, fixed.ENCODER, device, ends=True)
    raw = raw[..., : len(stats["mean"])]
    model.set_control(patches)
    codes = tests.to_complex(train.encode_pairs(model, (raw - stats["mean"].to(device)) / stats["std"].to(device)))
    eigenvalues = torch.cat([model.real_eigenvalue.to(torch.cfloat), torch.complex(model.block_real, model.block_imag)])

    return model, raw.cpu(), labels.cpu(), codes.cpu(), eigenvalues.cpu()
