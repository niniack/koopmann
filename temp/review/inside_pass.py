"""
One pass over the 50,000 validation images per model. It saves what every `inside_*` figure needs:
  data/inside_<name>.pt           accuracy of each read-out and ablation, FVUs, eigenvalues, mode usage, shares of z12
  data/inside_latents_<name>.pt   each image's latent at block 6, the predicted latent and the encoded block 12

The read-outs (question a): the head reads the real block-12 state, the block-12 state encoded and decoded,
the block-6 state encoded and decoded, and the model's prediction.
The ablations (question d) change one part of the step z12 = λ·z6 + control and read the answer again.

Usage: inside_pass.py [<run name> ...] [--limit=<images>]     (no name = every model of the review)
"""

import fire
import torch

from inside_load import DATA, NAMES, get_eigenvalues, get_encoder, is_kae, load_head, load_model, load_val, train  # isort: skip  (sets the path)

import angle_tests as tests
from where_control import first, keep, second

torch.set_num_threads(16)
chunks = lambda function, values: torch.cat([function(part) for part in values.split(5_000)])


def turn(z6, eigenvalues):
    """K's step alone: every mode times its eigenvalue."""

    return tests.to_latent(eigenvalues * tests.to_complex(z6))


def get_kae_rows(kae, z6, z12):
    """Block-12 latents with one part of the step changed, and the control's part of z12.

    NOTE: Mode 0, the single real mode, keeps its eigenvalue in every row, as in angle_tests.ablate.
    """

    eigenvalues = get_eigenvalues(kae)
    own = turn(z6, eigenvalues)
    control = z12 - own
    size = eigenvalues.abs().to(torch.cfloat)
    changed = {"angle dropped": size, "size dropped": eigenvalues / size, "λ = 1": torch.ones_like(size), "λ = 0": torch.zeros_like(size)}
    rows = {"as trained": z12}
    for row, values in changed.items():
        values[0] = eigenvalues[0]
        rows[row] = turn(z6, values) + control
    if kae.control == "none":
        del rows["λ = 0"]
        return rows, own, control

    rows["control dropped"] = own
    # The same pieces by coordinate, as in where_control.py
    rows["second coordinates of λ·z6 removed"] = keep(own, 1, 0) + control
    rows["second coordinates of control removed"] = own + keep(control, 1, 0)
    rows["first coordinates of control removed"] = own + keep(control, 0, 1)
    rows["all second coordinates removed"] = keep(z12, 1, 0)
    return rows, own, control


def get_mlp_rows(model, z6, z12):
    """The same for the MLP baselines, where the step is z12 = z6 + MLP(z6) (+ control)."""

    own = train.MlpKoopmanAutoencoder.koopman_forward(model, z6, 1)
    rows = {"as trained": z12, "MLP dropped": z6 + (z12 - own)}
    if hasattr(model, "signal"):
        rows["control dropped"] = own
    return rows, own, z12 - own


def get_shares(z12, own, control):
    """Shares of z12's squared size. NOTE: λ·z6 and the control overlap, so the shares need not add up to 1."""

    total = z12.pow(2).sum()
    shares = {"λ·z6": own.pow(2).sum() / total, "control": control.pow(2).sum() / total}
    for label, part in {"λ·z6": own, "control": control}.items():
        shares[f"{label}, first coordinates"] = first(part).pow(2).sum() / total
        shares[f"{label}, second coordinates"] = second(part).pow(2).sum() / total
    return {key: value.item() for key, value in shares.items()}


def run(name, states, patches, labels):
    model, stats = load_model(name)
    head = load_head(get_encoder(name), stats)
    x = (states[..., : len(stats["mean"])] - stats["mean"]) / stats["std"]
    if hasattr(model, "set_control"):
        model.set_control(patches)

    # shape: [images, 2 (block 6, block 12), latent]
    phi = chunks(lambda part: train.encode_pairs(model, part), x)
    z6, z12 = phi[:, 0], model.koopman_forward(phi[:, 0], 1)
    decode = lambda latent: chunks(model.decode, latent)
    read_outs = {"real block 12": x[:, 1], "block 12, encoded and decoded": decode(phi[:, 1]), "block 6, encoded and decoded": decode(z6), "prediction": decode(z12)}
    answers = {key: chunks(head.classify, state) for key, state in read_outs.items()}
    clip, own_answers = answers["real block 12"], answers["prediction"]
    score = lambda found: {"accuracy": (found == labels).float().mean().item(), "agreement": (found == clip).float().mean().item(), "same as trained": (found == own_answers).float().mean().item()}

    results = {"name": name, "label": NAMES[name], "images": len(x), "read_outs": {key: score(found) for key, found in answers.items()}, "answers": {key: found.short() for key, found in answers.items()}}
    # State FVU against the real block-12 state: over the whole state, and over the CLS token alone
    results["fvu"] = {key: train.compute_fvu(state[:, None], x[:, 1:]).item() for key, state in read_outs.items()}
    results["fvu_cls"] = {key: train.compute_fvu(state[:, None, :768], x[:, 1:, :768]).item() for key, state in read_outs.items()}
    results["fvu_block6"] = train.compute_fvu(read_outs["block 6, encoded and decoded"][:, None], x[:, :1]).item()
    results["latent_fvu"] = train.compute_fvu(z12[:, None], phi[:, 1:]).item()

    rows, own, control = get_kae_rows(model, z6, z12) if is_kae(model) else get_mlp_rows(model, z6, z12)
    results["rows"] = {row: score(chunks(head.classify, decode(latent))) for row, latent in rows.items()}
    results["shares"] = get_shares(z12, own, control)
    if is_kae(model):
        results["eigenvalues"] = get_eigenvalues(model)
        results["usage"] = (tests.to_complex(z6) != 0).sum(dim=0)
        results["real_start"] = bool(stats.get("real_start", False))

    torch.save(results, DATA / f"inside_{name}.pt")
    torch.save({"z6": z6, "z12": z12, "end": phi[:, 1]}, DATA / f"inside_latents_{name}.pt")
    print(f"{name}: " + " | ".join(f"{key} {100 * value['accuracy']:.2f}" for key, value in results["read_outs"].items()), flush=True)
    print("  " + " | ".join(f"{row} {100 * value['accuracy']:.2f}" for row, value in results["rows"].items()), flush=True)
    print("  " + " | ".join(f"{key} {100 * value:.2f}%" for key, value in results["shares"].items()), flush=True)


def main(*names, limit=None):
    names = names or list(NAMES)
    for encoder in sorted({get_encoder(name) for name in names}):
        states, patches, labels = load_val(encoder, limit)
        for name in [name for name in names if get_encoder(name) == encoder]:
            run(name, states, patches, labels)


if __name__ == "__main__":
    fire.Fire(main)
