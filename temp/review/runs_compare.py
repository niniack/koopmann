"""
The reference points that are not runs, and every pair of runs where one setting changed.

Writes:
  data/references.csv   each reference accuracy with where it was measured and its source
  data/comparisons.csv  per pair: the accuracy of each run at the point where both are compared, the difference in points,
                        the steps each had, and `settings_that_differ` (every setting column of runs.csv that is not equal)

Accuracy is `val/zero_shot_accuracy`. The pairs of 7 Nov only have the fitted classifier; `metric` says which one a row uses.

Usage: runs_compare.py
"""

import csv

from build_runs import write
from common import DATA

SETTINGS = ["encoder", "dataset", "crops", "train_fraction", "start_block", "state", "latent_size", "hidden_width", "hidden_layers", "operator", "top_k", "batch_top_k",
            "shared_modes", "real_start", "control", "control_penalty", "weight_decay", "kl_weight", "embedding_weight", "isometric_weight", "overlap_weight",
            "standardization", "lr", "seed", "planned"]  # fmt: skip

# (key, what it is, accuracy, measured on, source)
REFERENCES = [
    ("real_clip", "Real CLIP ViT-B/32, all 12 blocks", 0.63336, "50,000 validation images", "wandb e2gcy7w9 val/real_model_accuracy; night/probe_clip_b6.log line 70; night/im_iso_2x4k.log line 58"),
    ("real_clip_train", "Real CLIP on the stored training crops", 0.63498, "stored training crops", "night/probe_clip_b6.log line 62 (train/real_model_accuracy)"),
    ("head_block6", "CLIP's head read straight off block 6", 0.002, "validation images", "review/history/full_11-10.txt line 14161 (printed as 0.2%); night_results.md line 300"),
    ("frozen_patches", "Real blocks 7-12, patch tokens frozen at block 6", 0.0330, "5,000 validation images (real CLIP 0.6322 on them)", "review/history/full_11-10.txt lines 16041-16043 (ceiling_frozen.py)"),
    ("linear_cls", "Linear map from block 6 to block 12, CLS-only state", 0.07056, "50,000 validation images", "wandb o441g90b baseline/linear_zero_shot_accuracy"),
    ("linear_map", "Linear map from block 6 to block 12, three-part state", 0.1842, "50,000 validation images", "night/im_iso_2x4k.log line 56; wandb ez6cx5a3"),
    ("mlp_probe", "Plain MLP trained like the KAE on the block-6 state (no KAE)", 0.5303, "50,000 validation images", "wandb e2gcy7w9; night/probe_clip_b6.log line 22 (epoch 8)"),
    ("label_block6", "Label probe on block 6 (trained with labels, own head)", 0.62304, "50,000 validation images", "night/probe_clip_b6.log line 68; wandb e2gcy7w9"),
    ("label_block12", "Label probe on block 12 (trained with labels, own head)", 0.78176, "50,000 validation images", "night/probe_clip_b6.log line 67; wandb e2gcy7w9"),
    ("real_dino", "Real DINOv3 ViT-B/16 with its ImageNet linear probe", 0.83936, "50,000 validation images", "night/im_dino_w21_wd_e30.log line 61; wandb ze4m5akv"),
    ("linear_dino", "DINOv3: linear map from block 6 to block 12", 0.27158, "50,000 validation images", "night/im_dino_w21_wd_e30.log line 59; wandb ze4m5akv"),
]

ONE, TWO, LOOSE = "one change", "two changes at once", "schedules or data order not matched"
# (group, what changed from A to B, run A, run B, epoch or step where both are compared (None = the end of each), kind)
PAIRS = [
    ("fresh crops", "Add the control (block-6 patch tokens), three-part state", "on_none", "on_add", 15_000, ONE),
    ("fresh crops", "Bilinear control in place of additive", "on_add", "on_bilinear", 15_000, ONE),
    ("fresh crops", "State is the CLS token alone, not three parts (with control)", "on_add", "cls_add", 15_000, ONE),
    ("fresh crops", "Real start (CLS-only KAE with control)", "cls_add", "cls_add_real", None, ONE),
    ("fresh crops", "MLP in place of eigenvalues, and no TopK (CLS-only, with control)", "cls_add", "cls_mlp", None, TWO),
    ("fresh crops", "Penalty 0.1 on the control", "cls_add_real", "share_real_w01", None, ONE),
    ("fresh crops", "Penalty 0.3 on the control", "cls_add_real", "share_real_w03", None, ONE),
    ("fresh crops", "Penalty 1.0 on the control", "cls_add_real", "share_real_w1", None, ONE),
    ("fresh crops", "Fresh crops in place of 3 stored crops (no control)", "im_iso_2x4k", "on_none", None, LOOSE),
    ("stored crops, hidden 2 x 4,096", "Another seed", "im_iso_2x4k", "im_iso_2x4k_s1", None, ONE),
    ("stored crops, hidden 2 x 4,096", "Weight decay 0.1", "im_iso_2x4k", "im_iso_2x4k_wd", None, ONE),
    ("stored crops, hidden 2 x 4,096", "Hidden width 8,192 in place of 4,096 (with weight decay)", "im_iso_2x4k_wd", "im_iso_2x8k_wd", None, ONE),
    ("stored crops, hidden 2 x 4,096", "MLP in place of eigenvalues, and no TopK (three-part state)", "im_iso_2x4k", "im_mlpk_2x4k", None, TWO),
    ("stored crops, hidden 2 x 4,096", "Real start plus overlap loss 0.1 (with weight decay)", "im_iso_2x4k_wd", "im_realstart_2x4k_wd", None, TWO),
    ("stored crops, hidden 2 x 4,096", "Another seed (real-start run)", "im_realstart_2x4k_wd", "im_realstart_2x4k_wd_s1", None, ONE),
    ("stored crops, hidden 2 x 4,096", "KL weight 0.1 in place of 0.03 (hidden 2 x 4,096)", "im_iso_2x4k", "im_iso_2x4k_kl0.1", None, ONE),
    ("stored crops, hidden 2 x 4,096", "Learning rate doubled", "im_iso_2x4k", "im_iso_2x4k_lr2", None, ONE),
    ("stored crops, hidden 2 x 4,096", "Four patch means (one per quarter of the image) in place of one", "im_iso_2x4k", "im_grid_2x4k_r2", None, ONE),
    ("stored crops, hidden 2 x 4,096", "Half of the images, same number of steps", "im_iso_2x4k", "im_iso_2x4k_f50", None, ONE),
    ("stored crops, hidden 2 x 4,096", "A quarter of the images, same number of steps", "im_iso_2x4k", "im_iso_2x4k_f25", None, ONE),
    ("stored crops, hidden 2 x 4,096", "Isometric weight 0.1 in place of 0", "im_shared_noiso_2x4k", "im_iso_2x4k", None, ONE),
    ("stored crops, hidden 2 x 4,096", "ImageNet-21k states added (with weight decay)", "im_iso_2x4k_wd", "im_w21_wd_2x4k", None, ONE),
    ("stored crops, hidden 2 x 4,096", "Plain MLP of the same width, no KAE (with weight decay)", "im_iso_2x4k_wd", "probe_clip_b6", None, TWO),
    ("stored crops, hidden 2,048", "Hidden layers 2 x 4,096 in place of 1 x 2,048 (wider and deeper)", "im_shared_noiso", "im_shared_noiso_2x4k", None, TWO),
    ("stored crops, hidden 2,048", "KL weight 0.01 in place of 0", "im_std_kl0", "im_std_kl0.01", None, ONE),
    ("stored crops, hidden 2,048", "KL weight 0.03 in place of 0.01", "im_std_kl0.01", "im_std_kl0.03", None, ONE),
    ("stored crops, hidden 2,048", "KL weight 0.1 in place of 0.03 (hidden 2,048)", "im_std_kl0.03", "im_std_kl0.1", None, ONE),
    ("stored crops, hidden 2,048", "Shared modes (block 12 keeps the modes TopK picked at block 6)", "im_std_kl0.03", "im_std_shared", None, ONE),
    ("stored crops, hidden 2,048", "TopK over the whole batch, not per image", "im_std_kl0.03", "im_std_kl0.03_btk", None, ONE),
    ("stored crops, hidden 2,048", "40 epochs in place of 8", "im_std_kl0.03", "im_std_kl0.03_e40", None, ONE),
    ("stored crops, hidden 2,048", "CC3M states added, same number of steps", "im_std_kl0.03_e40", "im_cc3m_std_kl0.03", None, ONE),
    ("stored crops, hidden 2,048", "Each block standardized with its own mean and std", "im_std_kl0.03", "im_std_block", None, ONE),
    ("stored crops, hidden 2,048", "Each block centered on its own mean", "im_std_kl0.03", "im_std_center", None, ONE),
    ("stored crops, hidden 2,048", "Same settings run again (after a change in how states are stored)", "im_std_kl0.03", "im_std_ctrl", None, ONE),
    ("stored crops, hidden 2,048", "Isometric weight 0 in place of 1 (shared modes)", "im_std_shared", "im_shared_noiso", None, ONE),
    ("stored crops, hidden 2,048", "Own mean and std per block, on top of shared modes", "im_std_shared", "im_std_shared_block", None, ONE),
    ("stored crops, hidden 2,048", "Patch std added to the state (CLS + patch mean before)", "im_jordan256_2k_patch", "im_jordan256_2k_std", None, ONE),
    ("stored crops, hidden 2,048", "3 crops x 40 epochs in place of 1 crop x 120 epochs", "im_jordan256_2k_patch_v1", "im_jordan256_2k_patch", None, ONE),
    ("stored crops, hidden 2,048", "Learning rate 0.02 in place of 0.0075", "im_jordan256_2k_patch", "im_jordan256_2k_patch_lr", None, ONE),
    ("stored crops, hidden 2,048", "Weight decay 0.1 (hidden 2,048, at epoch 18)", "im_jordan256_2k_std", "im_jordan256_2k_std_wd", 18, ONE),
    ("stored crops, CLS state", "Embedding and KL losses added (CLS state)", "im_jordan_2k_v3", "im_jordan_2k_v3_both", 43, TWO),
    ("stored crops, CLS state", "MLP in place of eigenvalues, nothing else (CLS state)", "im_jordan_2k_v3_both", "im_mlp_2k_both", 13, ONE),
    ("stored crops, CLS state", "TopK 256 in place of no TopK (CLS state)", "im_jordan_2k_v3_both", "im_jordan256_2k_both", 12, ONE),
    ("stored crops, CLS state", "Latent 4,097 with TopK 512, not 2,049 with TopK 256", "im_jordan256_2k_both", "im_jordan512_4k_both", 6, TWO),
    ("stored crops, CLS state", "KL loss taken out, embedding loss kept (CLS state)", "im_jordan512_4k_both", "im_jordan512_4k_emb", 6, ONE),
    ("stored crops, CLS state", "Patch mean added to the state (CLS alone before)", "im_jordan256_2k_both", "im_jordan256_2k_patch", 12, LOOSE),
    ("DINOv3", "DINOv3: 30 epochs in place of 12", "im_dino_w21_wd", "im_dino_w21_wd_e30", None, ONE),
    ("7 Nov, fitted classifier", "Eigenvalues in place of a dense matrix (ImageNet, 20 epochs)", "im_dense", "im_jordan", None, ONE),
    ("7 Nov, fitted classifier", "Eigenvalues in place of a dense matrix (ImageNet, 100 epochs)", "im_dense_e100", "im_jordan_e100", None, ONE),
    ("7 Nov, fitted classifier", "TopK 128 in place of no TopK (ImageNet, 20 epochs)", "im_jordan", "im_jordan128", None, ONE),
    ("7 Nov, fitted classifier", "100 epochs in place of 20 (dense matrix)", "im_dense", "im_dense_e100", None, ONE),
    ("7 Nov, fitted classifier", "Latent and hidden 2,048 in place of 1,024 (dense matrix)", "im_dense_e100", "im_dense_2k", None, TWO),
    ("7 Nov, fitted classifier", "TopK 256 in place of no TopK (latent 2,049, at epoch 61)", "im_jordan_2k", "im_jordan256_2k", 61, ONE),
    ("7 Nov, fitted classifier", "Eigenvalues in place of a dense matrix (Imagenette)", "b6_novc", "b6_jordan", None, ONE),
    ("7 Nov, fitted classifier", "Another seed (Imagenette, dense matrix)", "b6_novc", "b6_novc_s1", None, ONE),
    ("7 Nov, fitted classifier", "Variance and covariance losses taken out (Imagenette)", "b6_30v", "b6_novc", None, ONE),
]


def load(name):
    with open(DATA / name) as file:
        return list(csv.DictReader(file))


def get_point(curves, run, at):
    """The curve row of `run` at epoch or step `at`, or its last row."""
    rows = [row for row in curves if row["run"] == run]
    return next(row for row in rows if int(row["at"]) == at) if at else rows[-1]


def compare(pair, runs, curves):
    """One row of comparisons.csv."""
    group, change, a, b, at, kind = pair
    first, second = get_point(curves, a, at), get_point(curves, b, at)
    metric = "accuracy" if first["accuracy"] else "fitted_accuracy"
    differ = [f"{key}: {runs[a][key]} | {runs[b][key]}" for key in SETTINGS if runs[a][key] != runs[b][key]]
    return {
        "group": group, "change": change, "run_a": a, "run_b": b, "kind": kind,
        "metric": "val/zero_shot_accuracy" if metric == "accuracy" else "val/predicted_accuracy (fitted classifier)",
        "compared_at": f"{first['unit']} {first['at']} | {second['unit']} {second['at']}",
        "steps_a": first["steps"], "steps_b": second["steps"],
        "accuracy_a": round(100 * float(first[metric]), 2), "accuracy_b": round(100 * float(second[metric]), 2),
        "difference_points": round(100 * (float(second[metric]) - float(first[metric])), 2),
        "settings_that_differ": "; ".join(differ),
    }  # fmt: skip


if __name__ == "__main__":
    runs = {row["name"]: row for row in load("runs.csv")}
    curves = load("curves.csv")
    write("references.csv", [dict(zip(("key", "what", "accuracy", "measured_on", "source"), row)) for row in REFERENCES])
    rows = [compare(pair, runs, curves) for pair in PAIRS]
    write("comparisons.csv", rows)
    for row in rows:
        print(f"{row['difference_points']:+6.2f} | {row['accuracy_a']:5.2f} -> {row['accuracy_b']:5.2f} | {row['steps_a']:>6} / {row['steps_b']:>6} steps | {row['kind'][:3]} | {row['change']} [{row['run_a']} -> {row['run_b']}] :: {row['settings_that_differ']}")
