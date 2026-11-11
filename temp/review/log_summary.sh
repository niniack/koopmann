#!/bin/bash
# Prints, for each training log given, the wandb id and the final-epoch numbers of the wandb run summary.
for f in "$@"; do
  id=$(grep -o "runs/[a-z0-9]\{8\}" "$f" | tail -1 | cut -d/ -f2)
  get() { grep -E "wandb: +$1 " "$f" | tail -1 | awk '{print $NF}'; }
  printf "%-38s id=%s zs=%s agree=%s recon=%s latent=%s state=%s emb=%s kl=%s | train zs=%s agree=%s recon=%s | best_ep=%s lin_zs=%s true_zs=%s\n" \
    "$f" "$id" "$(get val/zero_shot_accuracy)" "$(get val/clip_agreement)" "$(get val/reconstruction_fvu)" "$(get val/latent_fvu)" "$(get val/one_step_fvu)" "$(get val/embedding_loss)" "$(get val/kl_loss)" \
    "$(get train/zero_shot_accuracy)" "$(get train/clip_agreement)" "$(get train/reconstruction_fvu)" "$(get best/epoch)" "$(get baseline/linear_zero_shot_accuracy)" "$(get baseline/true_zero_shot_accuracy)"
done
