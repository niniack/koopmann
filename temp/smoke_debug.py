"""Runs train.main on Imagenette on the CPU and prints each epoch's logs, to find why the latent logs fail."""
import sys

import torch

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train  # noqa: E402

real_start, overlap = sys.argv[1] == "1", float(sys.argv[2])
original = train.get_latent_logs


def get_latent_logs(kae, states):
    phi = train.encode_pairs(kae, states)
    cov = train.get_covariances(phi)
    print("   phi finite", bool(phi.isfinite().all()), "| cov finite", bool(cov.isfinite().all()), "| phi abs max", phi.abs().max().item(), "| layer scale", phi.pow(2).mean(dim=[0, 2]).sqrt().tolist(), "| zero-variance dims", (cov.diagonal(dim1=1, dim2=2) == 0).sum(-1).tolist(), flush=True)
    try:
        return original(kae, states)
    except Exception as error:
        print("   latent logs FAILED:", str(error)[:150], flush=True)
        for layer in range(2):
            for dtype in [torch.float, torch.double]:
                try:
                    values = torch.linalg.eigvalsh(cov[layer].to(dtype))
                    print(f"      layer {layer} {dtype}: ok, min {values.min().item():.3g}, max {values.max().item():.3g}", flush=True)
                except Exception as inner:
                    print(f"      layer {layer} {dtype}: FAILED", flush=True)
        torch.save(cov, "/home/nsa325/work/koopmann/temp/failing_cov.pt")
        return {}


train.get_latent_logs = get_latent_logs
evaluate = train.evaluate


def show(kae, states, labels, classifier, clip_head, split):
    logs = evaluate(kae, states, labels, classifier, clip_head, split)
    if split == "val":
        angles = torch.rad2deg(torch.complex(kae.block_real, kae.block_imag).angle().abs())
        print({k.split("/")[1]: round(v, 4) for k, v in logs.items()}, "| angle median", round(angles.median().item(), 1), "max", round(angles.max().item(), 1), flush=True)
    return logs


train.evaluate = show
train.main(dataset="imagenette", tokens="cls", views=1, num_epochs=int(sys.argv[3]), lr=0.0075, adam_lr=0.00075, lambda_variance=0, lambda_covariance=0, lambda_isometric=0.1, ae_dim=2049, ae_hidden=1024, ae_layers=2, lambda_embedding=1, lambda_kl=0.03, jordan=True, top_k=256, shared_modes=True, real_start=real_start, lambda_overlap=overlap, weight_decay=0.1, name="smoke_real_start_tmp")
