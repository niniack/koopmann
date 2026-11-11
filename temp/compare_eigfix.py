"""Final-epoch metrics of the eigenvalue-fix runs against the 8-epoch reference nek1pwi8."""
import sys, wandb, pandas as pd
api = wandb.Api()
keys = {"val/zero_shot_accuracy": "zero-shot", "val/clip_agreement": "agree", "train/zero_shot_accuracy": "train zs",
        "val/one_step_fvu": "state", "val/embedding_loss": "embed", "val/kl_loss": "KL",
        "val/reconstruction_fvu": "recon*", "val/latent_fvu": "latent*"}
rows = {}
for name in ["im_std_kl0.03", *sys.argv[1:]]:
    runs = [r for r in api.runs("nishantaswani/tokae", filters={"display_name": name})]
    r = runs[-1] if name != "im_std_kl0.03" else api.run("nishantaswani/tokae/nek1pwi8")
    h = r.history(keys=list(keys))
    if len(h):
        rows[f"{name} ({r.id}, ep {len(h)})"] = h.iloc[-1][list(keys)].rename(keys)
print(pd.DataFrame(rows).T.round(4).to_string())
