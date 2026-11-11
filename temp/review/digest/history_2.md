# History, part 2: 8 and 9 November 2026 (EST)

What was tried and learned on the two middle days, in time order. Written from `review/history/talk_11-08.txt`,
`talk_11-09.txt`, `full_11-08.txt`, `full_11-09.txt`, the scratchpad logs and wandb (project `nishantaswani/tokae`,
read through `review/data/wandb_raw.json`).

**How to read the numbers**

- "zero-shot" = ImageNet validation accuracy when the KAE's decoded block-12 CLS goes through CLIP's own zero-shot head.
  "agreement" = share of validation images where the KAE gives the same answer as real CLIP. Real CLIP ViT-B/32 scores
  63.3% (wandb `baseline/true_zero_shot_accuracy` 0.63336).
- "FVU" = fraction of variance unexplained (0 is perfect, 1 is no better than predicting the average).
  "recon" = reconstruction FVU, "latent" = latent prediction FVU, "state" = FVU of the predicted block-12 state.
- A "mode" is one eigenvalue of K with its latent coordinates: 1 real mode plus 1,024 two-coordinate modes
  (2,049 coordinates, 1,025 modes). TopK keeps 128 modes per image.
- wandb numbers are the last logged epoch unless "best" is written. Run times are wandb creation times moved to EST.
- Sources are written as a wandb run id, a log path, or `full_MM-DD.txt line N`.
- Scratchpad = `/home/nsa325/work/koopmann/temp`.
  Every run named below has a checkpoint at `/home/nsa325/work/koopmann_data/weights_cache/tokae/<run name>.safetensors`
  (checked with ls), except the ones marked "no checkpoint".
- Text only: several modes in this period fire on insects (for example a "butterflies" mode). They must never be pictured.

---

# 8 November

## 11-08 00:12 EST Starting point: the CLS-only KAE on ImageNet
- Question: where does the ImageNet-scale KAE stand before anything is changed?
- Setup: CLIP ViT-B/32. The KAE replaces blocks 7 to 12. Input and target are the CLS token only (768 numbers) at block 6
  and block 12. K in real Jordan form, 2,049 latent coordinates, TopK keeps 128 of 1,025 modes. Losses: reconstruction,
  latent prediction, state prediction, embedding (weight 1), KL to CLIP's answer (weight 1). ImageNet train (1.28M images,
  3 crops each), batch 1,024.
- Result (runs made the evening before, all stopped by hand, so wandb shows "crashed"):
  - `x3vr9o3p` Jordan + TopK, KL 1 + embedding 1, 12 epochs: best 38.98% (epoch 10); last 38.82% zero-shot, 44.33% agreement, recon .3256, state .6163; train 47.56%.
  - `7t4irsvy` same but 4,097 coordinates and TopK 512, 6 epochs: best 39.19%; agreement 44.49%.
  - `uxsr2s59` Jordan, no TopK, KL 1 + embedding 1, 43 epochs: best 41.07%; last 40.92% / 46.37%; train 50.60%.
  - `p2jnnyzz` an MLP in place of K, no TopK, 13 epochs: best 40.26%; last 40.02% / 45.34%.
  - `o441g90b` Jordan, no TopK, no CLIP losses, 140 epochs: best 34.22%; last 34.11% / 40.58%; recon .0065.
  - `va7o4iml` 4,097 wide, TopK 512, embedding loss only: about 30% at this point (talk, `full_11-08.txt line 6685`); stopped at 02:23 with 32.43% / 38.87% after 97 epochs (wandb; the talk says "epoch 96 at 32.4%", `full_11-08.txt line 6926`).
- Conclusion at the time: best Jordan + TopK model 39% against CLIP's 63%. KL is the only thing that had helped a lot (about 34% to 41%, `full_11-08.txt line 6804`).
- Status now: superseded. Every later model in this period beats it by 8 to 15 points once the input holds more than the CLS token.
- Goal: groundwork (speed).
- User said: nothing at this point.
- Files worth reusing: checkpoints `im_jordan256_2k_both`, `im_jordan_2k_v3`, `im_jordan_2k_v3_both`, `im_mlp_2k_both`, `im_jordan512_4k_both`, `im_jordan512_4k_emb` in the weights folder.

## 11-08 00:12 EST First mode notebook: are the modes real?
- Question: do single modes of the CLS-only KAE stand for something a person can name?
- Setup: model `x3vr9o3p`; for each mode, the validation images where it is largest; a "coherence" score (are a mode's top images more alike than random images, in CLIP's own embedding).
- Result (`full_11-08.txt line 6685`): 85% of modes have top images more alike than random. Named modes: snow, motion blur, sepia, old black-and-white photos, stripes, butterflies. Turning each mode off one at a time: none improves overall accuracy.
- Conclusion at the time: the modes are real features, but removing any one of them is not a free accuracy gain.
- Status now: stands. The same score was 89% on Run C (`full_11-08.txt line 7560`), 81% on `im_cc3m_2x4k_e20` (`line 9646`), 75% on `im_cc3m_iso0.1_2x4k_e20` (`full_11-09.txt line 10414`).
- Goal: meaning.
- User said: nothing tied to it.
- Files worth reusing: `/home/nsa325/work/koopmann/second_analysis/archive/jordan_topk_modes.ipynb`; `/home/nsa325/work/koopmann/temp/mode_coherence.pkl`; `/home/nsa325/work/koopmann/temp/removal.pkl`.

## 11-08 00:12 EST First snow edit: turn the snow mode's eigenvalue down
- Question: can one eigenvalue fix a domain? Snowy photos of vehicles are often answered "snowplow" or "snowmobile".
- Setup: model `x3vr9o3p`; snow mode 903 (it separates snowy from normal photos with AUC .95, `full_11-08.txt line 7068`); a hand-made snow test of 479 validation photos, 129 of them snowy; the mode's eigenvalue scaled down.
- Result: snowy photos 17.8% to 23.3%; normal photos unchanged (`full_11-08.txt line 6685`, reproduced at `line 7068`). When removing the mode changes an answer, 76% of the time the new answer was the model's second choice. Real CLIP on the same test: snowy 53.5%, normal 64.3% (`line 6771`).
- Conclusion at the time: the edit works in the right direction but the base model is far below CLIP on these photos.
- Status now: superseded by stronger models (Run C: 30.2% to 51.2%; `im_cc3m_2x4k_e20`: 32.6% to 55.0%; see 23:12 below). The limits found later still hold: the gain is on a hand-picked set, and over all validation images the edit breaks as many answers as it fixes.
- Goal: adaptation.
- User said: nothing at this point.
- Files worth reusing: `/home/nsa325/work/koopmann/temp/check_snow_mode.py`, `.../scratchpad/snow_collateral.py`. The file `snow_test.pt` no longer exists.

## 11-08 00:12 EST Confounder study: do modes explain CLIP's known mix-ups?
- Question: when CLIP confuses two classes because of context (baseball / "baseball player", Pickelhaube / "military uniform"), is there a mode that carries the context, and does turning it off fix the answer?
- Setup: model `x3vr9o3p` on ImageNet train. For every (class, mode) pair, CLIP's accuracy on the class when the mode is on against when it is off. 937,591 pairs tested. CLIP zero-shot on train: 64.41%.
- Result (`/home/nsa325/work/koopmann/temp/confounders.log`), snow mode 903, classes where CLIP does worst when the mode is on (images with mode on / CLIP accuracy with mode on / with mode off): clothes iron 172 / .302 / .521; shovel 469 / .339 / .415; gar fish 189 / .434 / .619; bell tower 226 / .562 / .712; grey wolf 250 / .196 / .328; lens cap 103 / .398 / .690; crash helmet 207 / .565 / .704; Samoyed 278 / .662 / .761. For the two textbook confounders a matching mode exists (sports action, old photo) but turning it off does not fix the answers (`full_11-08.txt line 6685`).
- Conclusion at the time: modes can flag where CLIP is weaker, but removing a context mode does not repair a confusion between two labels.
- Status now: stands; never extended. The script crashed with an IndexError after printing the table, so the saved result files may be incomplete.
- Goal: adaptation, meaning.
- User said: nothing tied to it.
- Files worth reusing: `.../scratchpad/proto_confounders.py`, `.../scratchpad/confound_pairs.pkl`, `confound_top.pkl`, `confound_top_classes.pkl`, `confound_val.pkl`, `confound_val_classes.pkl` (all in the scratchpad root).

## 11-08 00:12 EST Synthetic shifts: grayscale, sepia, motion blur
- Question: if a photo is altered in a known way, does a matching mode switch on, and does turning it off undo the damage?
- Setup: 5,000 validation photos and their grayscale, sepia and motion-blur copies, through model `x3vr9o3p`.
- Result (`full_11-08.txt line 6685`): the matching modes fire almost perfectly (AUC .98 to 1.00). Turning them off gains about 1 point. The KAE loses much more than CLIP on altered photos. Repeated on Run C at 09:28: the KAE loses 1.5 to 2 times as much as CLIP (`line 7560`).
- Conclusion at the time: the modes detect the alteration well; removing them does not repair it.
- Status now: stands; not repeated on later models. Exact per-shift accuracies were not kept in a log: not measured beyond the numbers above.
- Goal: adaptation, meaning.
- User said (00:21, on training with altered copies): "it will never end".
- Files worth reusing: none found.

## 11-08 00:24 EST Damping sweep on the snow mode
- Question: is a gentle reduction of the snow eigenvalue safer than removing it?
- Setup: model `x3vr9o3p`, snow mode 903. The eigenvalue is multiplied by a scale. The edit only touches the 7,372 validation images where TopK keeps the snow mode.
- Result (`full_11-08.txt line 6733`), scale: snowy fixed / snow-test normal broken / rest of ImageNet fixed / rest of ImageNet broken:
  1.0: 0 / 0 / 0 / 0; 0.8: 4 / 0 / 12 / 13; 0.6: 6 / 0 / 26 / 24; 0.5: 7 / 0 / 27 / 29; 0: 7 / 0 / 52 / 80.
  7 fixes are out of 106 wrong snowy photos.
- Conclusion at the time: 0.5 gets all the fixes that removal gets and breaks fewer other images. But 0.5 was picked on the same images it is scored on.
- Status now: stands as a small effect on a weak model; the tuning-on-test caveat was never removed.
- Goal: adaptation.
- User said: asked for gentle damping in place of training on altered images.
- Files worth reusing: none (`damping_sweep.py` no longer exists).

## 11-08 00:35 EST Do nameable modes need TopK?
- Question: "does it only happen with top k btw?"
- Setup: three trained models with the same data; the coherence score and the count of modes that fire on one class only.
- Result (`full_11-08.txt line 6771`): with TopK: 85% of modes beat random, 61 single-class modes. Without TopK, same losses: 26% and 1. Without TopK and without CLIP losses: 41% and 0.
- Conclusion at the time: TopK is what makes the modes nameable.
- Status now: stands; not re-tested on the bigger models (the MLP-K run on 11-09 dropped TopK but its units were never scored).
- Goal: meaning.
- User said: "does it only happen with top k btw?" and "how do we use this framework?"
- Files worth reusing: none specific.

## 11-08 00:41 EST Stock-take: "do we have something real here?"
- Question: is there a result worth building on?
- Setup: no new run.
- Result (`full_11-08.txt line 6780`): solid: the modes are real. Weak: 39% against 63%; the KAE gives CLIP's answer only 44% of the time; the snow fix is 7 of 106 photos from one seed.
- Conclusion at the time: an "adapter" framing was offered (keep CLIP, use the KAE only to correct it).
- Status now: the adapter framing was rejected by the user at 01:12.
- Goal: groundwork.
- User said: "do we have something real here?"
- Files worth reusing: none.

## 11-08 01:02 EST Sizes of the named modes' eigenvalues
- Question: do the eigenvalue sizes of named modes say anything?
- Setup: model `x3vr9o3p`.
- Result (`full_11-08.txt line 6795`): old black-and-white photos |λ| 1.10; snow 0.85; a typical mode about 0.71. Latent prediction error about .04.
- Conclusion at the time: named "context" modes sit above the typical size.
- Status now: superseded. The 13:53 check showed that in these models λ did not match the measured growth of rare modes; that was fixed by shared modes at 16:59.
- Goal: meaning.
- User said: nothing tied to it.
- Files worth reusing: none.

## 11-08 01:12 EST Decision: improve the KAE itself; find the bottleneck
- Question: why is the KAE stuck near 39 to 41%?
- Setup: review of the evening's runs plus linear checks on CLIP features.
- Result (`full_11-08.txt line 6804`): 4,097 coordinates did not help (39.2% against 39.0%). An MLP in place of K was no better (40.3%). Validation stops improving at epoch 10 to 15. A linear classifier gets 59% from the block-12 CLS but only 24% from the block-6 CLS. A linear map from block-6 CLS to block-12 CLS scores 7% zero-shot.
- Conclusion at the time: the block-6 CLS token alone does not hold enough about the image. The fix to try is to give the KAE a summary of the block-6 patch tokens.
- Status now: stands. This diagnosis drove every gain of the next two days.
- Goal: speed.
- User said: "i'm not sure we are there yet nor do i follow. i also don't want to relegate my model just as an adapter. i really think we can improve the kae and make it more useful, we just need to look harder." And at 02:22: "you're welcome to stop the run and try out some pooled patches or other things. ultimately, do runs omething for long. also i fear our learning rate might be a bit too low. things seem to be always falling but very very slowly. i'm gonna sleep!"
- Files worth reusing: none.

## 11-08 03:25 EST Run A: add the average patch token to the input
- Question: does a patch summary lift the ceiling?
- Setup: `w5zkvf1x` (`im_jordan256_2k_patch`). Input and target at both blocks become CLS + mean of the patch tokens (1,536 numbers). Otherwise as `x3vr9o3p` (KL 1, embedding 1, TopK 128 modes), 40 epochs so the learning rate decays to zero, 3 crops. About 103 s per epoch (`full_11-08.txt line 7201`).
- Result: after the first epochs 47.6% / 54.8% against 37.2% / 42.6% without patches (`line 7164`). Best 49.55% (epoch 29); last 49.27% zero-shot, 57.34% agreement, recon .2642, latent .0372, state .4660; train 62.71% / 79.40%. Linear-map baseline on the same input 15.84% (wandb `w5zkvf1x`).
- Conclusion at the time: +10.5 points. The biggest single gain so far.
- Status now: stands as a finding. On 11-10 the user rejected pooled patches inside the state; the patch information now enters as a control.
- Goal: speed.
- User said: see 02:22 above.
- Files worth reusing: checkpoint `im_jordan256_2k_patch`.

## 11-08 03:55 EST Pooling check: which patch summary carries the most?
- Question: before training, which block-6 summary lets a plain linear map predict block 12 best?
- Setup: linear map from block 6 to block-12 CLS, fitted on 100k train images, scored on a 10k validation subset (real CLIP on this subset 63.14%).
- Result (`/home/nsa325/work/koopmann/temp/pool_check.log`), zero-shot / agreement: CLS 6.5% / 8.3%; CLS + mean 15.7% / 19.6%; CLS + mean + std 18.5% / 22.8%; CLS + 2×2 grid 18.1% / 22.3%; CLS + mean + 2×2 19.3% / 23.8%; CLS + 3×3 20.1% / 24.7%.
- Conclusion at the time: the mean gives most of the gain; the spread (std) adds about 3 points; grids add little more.
- Status now: stands; repeated with more variants at 11-09 01:39.
- Goal: speed.
- User said: nothing tied to it.
- Files worth reusing: `.../scratchpad/pool_check.py`, `.../scratchpad/pool_check.log`.

## 11-08 04:35 EST Run B: higher learning rate
- Question: the user feared the learning rate was too low. Does 2.7 times more help?
- Setup: `71cjcg4j` (`im_jordan256_2k_patch_lr`): Run A with Muon .02 and AdamW .002.
- Result: best 49.55% (epoch 32); last 48.96% / 56.77%; recon .3041; train 62.63%. Run A: best 49.55%.
- Conclusion at the time: no gain. Decaying the rate to zero was worth about +0.2 (`full_11-08.txt line 7560`).
- Status now: stands. Confirmed again on 11-10 (`9ck676yj`, doubled rates on the 2×4096 model: 52.39% / 62.02% against 52.11% / 62.17%).
- Goal: speed.
- User said: "i fear our learning rate might be a bit too low."
- Files worth reusing: checkpoint `im_jordan256_2k_patch_lr`.

## 11-08 06:25 EST Run C: add the spread of the patch tokens
- Question: does the per-dimension standard deviation of the patch tokens add to the mean?
- Setup: `mnbci7o1` (`im_jordan256_2k_std`): input CLS + patch mean + patch std (2,304 numbers per block), else as Run A.
- Result: best 51.17% (epoch 19); last 50.43% / 58.91%; recon .2898; state .4773; train 63.25% / 81.59%. Linear baseline 18.42%. Mode checks on this model (`full_11-08.txt line 7560`): 89% of modes pass coherence; damping the snow mode lifts snowy photos 30% to 51% (30.2% to 51.2%, 50 fixed / 116 broken over all validation, `line 9646`).
- Conclusion at the time: +1.7 over Run A. This 2,304-number state became the standard input for the rest of the period.
- Status now: stands as a measurement. It used KL 1, which the user then ruled out, so its 51.2% was not the accepted baseline (KL 0.03 gives 47.7% on the same input).
- Goal: speed, adaptation.
- User said: nothing at the time.
- Files worth reusing: checkpoint `im_jordan256_2k_std`.

## 11-08 07:48 EST Run D: one crop per image in place of three
- Question: do the 3 stored crops matter?
- Setup: `b6kv3mne` (`im_jordan256_2k_patch_v1`): Run A with 1 crop and 120 epochs (same number of steps).
- Result: best 48.18% (epoch 18); last 46.08% / 53.90%; train 64.45% / 88.47%. Run A: best 49.55%.
- Conclusion at the time: 3 crops are worth about +1.3; with one crop the model overfits faster (`full_11-08.txt line 7482`).
- Status now: stands. It was the first sign the model is limited by data.
- Goal: speed.
- User said (12:16): "I have a feeling we were data limited".
- Files worth reusing: checkpoint `im_jordan256_2k_patch_v1`.

## 11-08 09:00 EST Run E: weight decay on the small model
- Question: does weight decay reduce the overfitting?
- Setup: `lm09dz9a` (`im_jordan256_2k_std_wd`): Run C with Muon weight decay 0.1.
- Result: stopped at 18 of 40 epochs: 50.36% / 58.33%; recon .3257; latent .0986; train 54.98% / 63.92%. (The talk at 09:28 quoted "49.9% so far", `full_11-08.txt line 7560`.) Run C at its best: 51.17%.
- Conclusion at the time: train and validation move closer but validation is not better.
- Status now: stands; same picture at 2×4096 on 11-09, and on 11-10 weight decay was found to leave about half the modes unused and was dropped.
- Goal: speed.
- User said: nothing tied to it.
- Files worth reusing: checkpoint `im_jordan256_2k_std_wd`.

## 11-08 09:28 EST Morning summary: "still hitting a ceiling?"
- Question: what moved the ceiling overnight?
- Setup: Runs A to E.
- Result (`full_11-08.txt line 7560`), best zero-shot: CLS only 39.0%; + patch mean 49.5% (+10.5); + spread 51.2% (+1.7); learning rate ×2.7: 49.5% (0); 3 crops against 1: about +1.3; weight decay: no gain.
- Conclusion at the time: "the ceiling moved each time we gave the KAE more of block 6, and nothing else moved it."
- Status now: stands. Three choices were made overnight without asking the user: KL put back in at weight 1, 40 epochs in place of the 200 asked for, and Run E.
- Goal: speed.
- User said: "So it sounds like we are still hitting a ceiling?"
- Files worth reusing: none.

## 11-08 09:31 EST What KL at weight 1 costs the other losses
- Question: "But by bringing in kl loss are you sacrificing other losses? That's what's the issue with KL loss!"
- Setup: the 4,097-wide pair from the evening before at epoch 5: embedding loss only (`va7o4iml`) against embedding + KL (`7t4irsvy`).
- Result (`full_11-08.txt line 7582`), embedding only against + KL: recon 0.10 / 0.29; latent 0.067 / 0.155; block-12 state 0.569 / 0.629; embedding 0.145 / 0.163; zero-shot 26.2% / 39.2%.
- Conclusion at the time: KL at weight 1 buys 13 points of zero-shot and makes every other loss worse (reconstruction about 3 times worse). The assistant: "I should have checked this before putting KL back in."
- Status now: stands. Led to the KL sweep.
- Goal: speed, meaning (a KAE that reconstructs badly is a poor stand-in for the state).
- User said: the quote above, then at 09:38: "Never mind let's stick to current plan. Just stop using KL so aggressively. Please only include it at lower weighting, if it doesn't damage other losses! Another thing to try is see if dinov3 does any better? Maybe it's more amenable to this?"
- Files worth reusing: none.

## 11-08 09:34 EST Idea: give the KAE the history of the CLS token (dropped)
- Question: would the CLS token after blocks 1 to 6 (delay coordinates, as in Takens' theorem) help?
- Setup: discussion only. Because the encoder is shared, a model that reads blocks 1 to 6 would have to predict blocks 7 to 12 (`full_11-08.txt line 7597`, `line 7605`).
- Result: not run.
- Conclusion at the time: dropped by the user.
- Status now: never tried.
- Goal: speed.
- User said: "Can't we somehow give the history of the cls token? Maybe that helps us? It would also touch takens embedding theorem a bit?" then "Never mind let's stick to current plan."
- Files worth reusing: none.

## 11-08 09:39 EST KL sweep: 0, 0.01, 0.03, 0.1
- Question: how much KL can be used before the other losses suffer?
- Setup: Run C's input; embedding 1, isometric 1; 8 epochs each; KL weight 0 / 0.01 / 0.03 / 0.1 (`8dey3fxy`, `vonjrorw`, `nek1pwi8`, `ah7z87i0`).
- Result (wandb, last epoch):

  | KL weight | zero-shot | agreement | recon | latent | state | train zero-shot |
  |---|---|---|---|---|---|---|
  | 0 | 43.91% | 53.06% | .2015 | .0241 | .4340 | 46.29% |
  | 0.01 | 45.78% | 55.03% | .2028 | .0245 | .4341 | 48.81% |
  | 0.03 | 47.72% | 56.76% | .2050 | .0251 | .4346 | 51.02% |
  | 0.1 | 49.67% | 58.66% | .2134 | .0272 | .4396 | 54.63% |

  For scale, KL 1 at epoch 3: recon .384, latent .076, state .529, zero-shot 49.8% (`full_11-08.txt line 7774`).
- Conclusion at the time: 0.03 is the largest weight that leaves the other losses almost untouched (recon +1.7%). 0.1 starts to cost (recon +6%). Chosen: 0.03.
- Status now: stands; 0.03 was used in every later run of the period. KL 0.1 was re-tested at 2×4096 on 11-09 (+0.5 points, recon 3% worse).
- Goal: speed.
- User said (10:34): "Okay I'm stuck, not sure how to improve."
- Files worth reusing: checkpoints `im_std_kl0`, `im_std_kl0.01`, `im_std_kl0.03`, `im_std_kl0.1`.

## 11-08 11:06 EST Tricks from the sparse-autoencoder literature (list only)
- Question: "Are there any tips/tricks from SAE literature or transcoder?"
- Setup: a list from memory, not checked against the papers (`full_11-08.txt line 7832`): end-to-end SAEs, KL fine-tuning, skip transcoders, BatchTopK / AuxK, Matryoshka, data scale, SAEBench. Also counted: 5 to 42 dead modes in the KL-sweep models.
- Result: only BatchTopK was tried (next but one entry).
- Conclusion at the time: data scale and BatchTopK were the two worth trying.
- Status now: the list was never verified; skip connections were ruled out by the user.
- Goal: groundwork.
- User said: "Maybe try batch topk? Don't think I want to deal with skip connections".
- Files worth reusing: none.

## 11-08 11:07 EST DINOv3, first check with linear maps
- Question: "see if dinov3 does any better? Maybe it's more amenable to this?"
- Setup: no KAE. For CLIP ViT-B/32 and DINOv3 ViT-B/16: a linear probe on block 6, and a linear map from block 6 to block-12 CLS judged by a probe on block 12.
- Result:
  - CLIP (`/home/nsa325/work/koopmann/temp/dino_check.log`): probe on true block-12 CLS .561. CLS: probe on block 6 .199, linear map FVU .773, accuracy .175. CLS + mean: .300, .688, .254. CLS + mean + std: .339, .676, .277.
  - DINOv3 (`.../scratchpad/dino_check2.log`): probe on true block-12 CLS .732; the published head .832. CLS: .209, .866, .202 (published head .132). CLS + mean: .313, .793, .316 (.305). CLS + mean + std: .364, .780, .345 (.348). CLS + registers + mean + std: .385, .772, .368 (.388).
  - Share of the ceiling reached by the linear map: 49% (CLIP) against 47% (DINOv3; 50% with registers) (`full_11-08.txt line 7848`).
- Conclusion at the time: DINOv3 is not more amenable; it has a higher ceiling, and the same share of it is reachable.
- Status now: stands. The trained DINOv3 KAE on 11-09 reached 61.8% of 83.9%. Registers were never added to the KAE's input.
- Goal: speed.
- User said: the 09:38 quote above.
- Files worth reusing: `.../scratchpad/dino_check.py`, `.../scratchpad/dino_check.log`, `.../scratchpad/dino_check2.log` (the script `dino_check2.py` no longer exists).

## 11-08 11:47 EST BatchTopK
- Question: does choosing the kept modes over the whole batch (not per image) help?
- Setup: `r0epg244` (`im_std_kl0.03_btk`): the KL 0.03 model with BatchTopK, 8 epochs.
- Result: 45.76% / 54.93%, state .4513, against 47.72% / 56.76%, state .4346 for per-image TopK (`nek1pwi8`). Per epoch (`full_11-08.txt line 8024`): BatchTopK at epochs 2 / 4 / 6 / 8: 41.87 / 44.01 / 44.86 / 45.76%.
- Conclusion at the time: 2.0 points worse. Dropped.
- Status now: stands, but **the result was never told to the user in the talk**; BatchTopK was dropped without a report.
- Goal: speed.
- User said: "Maybe try batch topk?"
- Files worth reusing: checkpoint `im_std_kl0.03_btk`.

## 11-08 12:06 EST Longer training: 40 epochs on ImageNet only
- Question: is the 8-epoch model just undertrained?
- Setup: `jm17ztqa` (`im_std_kl0.03_e40`): the KL 0.03 model for 40 epochs.
- Result: best 48.12% (epoch 39); last 48.04% / 57.18%; recon .1973; train 53.32% / 63.06%. The 8-epoch model: 47.72%.
- Conclusion at the time: 5 times the training buys 0.4 points. More passes over the same images do not help.
- Status now: stands; same picture on DINOv3 later (30 epochs against 12: +1.1 points).
- Goal: speed.
- User said: nothing tied to it.
- Files worth reusing: checkpoint `im_std_kl0.03_e40`.

## 11-08 12:32 EST CC3M prepared as extra data
- Question: can more images break the ceiling?
- Setup: Conceptual Captions 3M: 576 files, 2,905,954 images, 261 GB, one centre crop each. ImageNet gives 1.28M images (3.84M states with 3 crops) (`full_11-08.txt line 8147`). The model has about 18M parameters.
- Result: data ready; the run is the 13:58 entry.
- Conclusion at the time: worth testing at equal training steps.
- Status now: CC3M was used until the morning of 11-09, then set aside.
- Goal: speed.
- User said: "I have a feeling we were data limited" and "Yeah do cc3m".
- Files worth reusing: none in the scratchpad.

## 11-08 13:53 EST Do the eigenvalues say how strong a mode is?
- Question: "is there a reason to believe that the eigenvalues we learn indicate anything about how strong a feature is, at least relative to other features? Or is it just arbitrary?" Also: "Is the only thing setting us apart from sae the fact that we predict the final result? And maybe the Jordan block arch?"
- Setup: model `jm17ztqa`, 50,000 validation images. For each mode: how often TopK keeps it, its share of block-12 latent energy, and R² between "block-6 size times λ" and its true block-12 size.
- Result (`full_11-08.txt line 8187`):

  | Mode is kept on | modes | share of block-12 latent energy | R² |
  |---|---|---|---|
  | 50 to 100% of images | 84 | 87% | .985 |
  | 10 to 50% | 214 | 13% | .977 |
  | 1 to 10% | 277 | 0.3% | .29 |
  | under 1% | 449 | 0.1% | .05 |

  The four largest eigenvalues (|λ| about 8 to 10) carry 93% of latent energy. A rare mode that is on at block 6 is in block 12's top 128 only about 10% of the time.
- Conclusion at the time: λ is meaningful for the common modes and close to arbitrary for the rare ones, because TopK picks different modes at the two blocks. The four huge modes were called "bookkeeping" (wrong; corrected at 16:41).
- Status now: the problem was fixed by shared modes (16:59). The size of λ was later shown to agree across two seeds (rank correlation +.73, 11-10).
- Goal: meaning.
- User said: the two quotes above.
- Files worth reusing: `.../scratchpad/eigen_check.py`, `.../scratchpad/eigen_check2.py`, `.../scratchpad/eigen_report.py`.

## 11-08 13:58 EST ImageNet + CC3M at equal steps
- Question: does adding CC3M help when the number of training steps is held fixed?
- Setup: `kfu2tcdk` (`im_cc3m_std_kl0.03`): ImageNet + CC3M, 23 epochs, the same number of steps as the 40-epoch ImageNet run `jm17ztqa`.
- Result (`full_11-08.txt line 8303`; wandb): ImageNet only against + CC3M: zero-shot 48.0% / 46.3% (48.04 / 46.25); agreement 57.2% / 55.7%; train zero-shot 53.3% / 49.5%; train-validation gap 5.3 / 3.3 points.
- Conclusion at the time: less overfitting, 1.7 points lower on ImageNet. First reported as "CC3M didn't help".
- Status now: the claim was softened at 18:32 after the user objected (each ImageNet image was seen fewer times). Later, at 2×4096 with matched settings, ImageNet + CC3M for 20 epochs gave 51.57% best against 52.11% for ImageNet only in 8 epochs. CC3M also triggered latent inflation (22:04).
- Goal: speed.
- User said (18:32): "why aren't we running imagenet + cc3m. Surely more data will help."
- Files worth reusing: checkpoint `im_cc3m_std_kl0.03`.

## 11-08 16:41 EST Correction: the huge modes are not bookkeeping
- Question: what do the four modes with |λ| 8 to 10 do?
- Setup: model `jm17ztqa`; per-image sizes of the four modes at both blocks.
- Result (`full_11-08.txt line 8488`): their sizes vary from image to image, and the block-6 and block-12 sizes correlate .98. In standardised units block 12 has 4.6 times block 6's spread per dimension.
- Conclusion at the time: these modes carry real per-image information and the growth in spread from block 6 to block 12. The "bookkeeping" label was withdrawn.
- Status now: stands; refined on 11-09 12:19 (three growth modes with their own meaning) and on 11-10 for DINOv3.
- Goal: meaning.
- User said (16:57): "To avoid giant eigenvalues can't we just shape the spectrum?" (Answer given: no, the growth has to go somewhere. Never tried.) At 16:33: "Ignore smi issues. Try whatever makes sense".
- Files worth reusing: none specific.

## 11-08 16:39 EST Shared modes: the eigenvalue fix
- Question: if TopK is forced to keep the same modes at block 6 and block 12, do the eigenvalues become honest for rare modes?
- Setup: `cv6l70im` (`im_std_shared`): the KL 0.03 model with one TopK choice shared by both blocks, 8 epochs. Compared with `nek1pwi8` (8 epochs) and `jm17ztqa` (40 epochs).
- Result (`full_11-08.txt line 8551`; wandb):
  - Accuracy: 47.72% to 48.59% zero-shot; 56.76% to 57.67% agreement. A rerun of the reference on the new code (`banl0nu8`) gave 47.62% / 56.69%, so the gain is real (the talk says "+0.8").
  - Cost: reconstruction block 6 .170 to .184, block 12 .240 to .269.
  - Eigenvalues, reference (40 epochs) against shared: modes used on at least 1% of images 575 / 1,005; nearly dead 348 / 15; pooled R² for modes on 1 to 10% of images .29 / .57; median R² −6.9 / .40; median of measured growth divided by λ .05 / .74; rank correlation .92 / .95; largest |λ| 8 to 10 / 4 to 7. Six always-on modes carry 90% of block-12 latent energy.
- Conclusion at the time: shared modes make λ match the measured growth of a mode, nearly remove dead modes, and gain almost a point.
- Status now: stands; kept in every later run including the current plan.
- Goal: meaning.
- User said (18:32): "I think A is definitely important, otherwise the story breaks." (A = shared modes.)
- Files worth reusing: checkpoint `im_std_shared`; `.../scratchpad/compare_eigfix.py`.

## 11-08 16:57 EST Standardisation variants: centring, per-block, shared + per-block
- Question: can a different standardisation shrink the giant eigenvalues without hurting?
- Setup: 8 epochs each, KL 0.03: `mwfzxnwj` (subtract each block's mean), `mfq4foq4` (standardise each block on its own), `i0v80lf4` (shared modes + per-block), `banl0nu8` (reference rerun).
- Result (`full_11-08.txt line 8675`; wandb): reference 47.72% (rerun 47.62%); centring 48.46% / 57.73%, reconstruction up to 34% worse; per-block 48.51% / 57.71%, largest |λ| about 1.5, reconstruction up to 51% worse; shared + per-block 45.24% / 54.12%: collapse, about 180 always-on modes with λ near 1 and about 800 nearly dead. (wandb recon for these runs, .3366 / .3668 / .4171, is in each run's own units; the "34%" and "51%" are from the fair comparison in `fair_fvu.py`.)
- Conclusion at the time: per-block standardisation removes the giant eigenvalues but hurts reconstruction, and it breaks when combined with shared modes.
- Status now: rejected by the user; one global standardisation for both blocks has been kept since.
- Goal: meaning.
- User said (17:10): "Explain to me like I'm stupid." (18:13): "okay explain the options to me once again like i'm stupid, but don't lose meaningful vocabulary". (18:32): "i don't think we should allow per block standardization. we can shift the scale and mean together to make the numbers more manageable but they should be relatively the same to each other. i think its okay to have some modes handle the norm growth. not every mode has to be interpretable. carrying norm growth is in itself an interpretability thing, no?"
- Files worth reusing: checkpoints `im_std_center`, `im_std_block`, `im_std_shared_block`, `im_std_ctrl`; `.../scratchpad/fair_fvu.py`.

## 11-08 18:34 EST Dead modes before and after shared modes
- Question: how many of the 1,025 modes are never used?
- Setup: count over validation of modes never kept / kept on under 1% of images.
- Result (`full_11-08.txt line 8749`): old 8 epochs 129 / 382; old 40 epochs 183 / 450; shared 8 epochs 6 / 19; shared 40-epoch run at epoch 9: 13 / 25. (`active_latents` reads 255.94, not 256, because the single real mode is kept for 5.7% of images.)
- Conclusion at the time: longer training without shared modes kills more modes; shared modes keep nearly all alive.
- Status now: stands. On 11-10 weight decay was found to do the opposite (about half the modes dead).
- Goal: meaning.
- User said: nothing tied to it.
- Files worth reusing: none specific.

## 11-08 18:12 EST Three runs started and stopped within an hour
- Question: shared modes for 40 epochs; then with CC3M; then without the isometric loss.
- Setup and result:
  - `8q13zz66` (`im_shared_e40`): stopped. The talk says "~10 epochs, 47.3%" (`full_11-08.txt line 8781`); wandb has 14 epochs, last 47.58% / 56.49%. Both are given because they disagree.
  - `v8efepb4` (`im_cc3m_shared_e40`): stopped after 2 epochs, 42.73% / 51.45% (after one pass 40.5% / 48.8%, `line 8800`).
  - `axg8piz0` (`im_cc3m_shared_noiso_e40`, isometric weight 0): stopped at 6 of 40 epochs, 43.81% / 52.70%, latent .0079.
- Conclusion at the time: none; each was replaced by the next request.
- Status now: superseded by the 2×4096 runs.
- Goal: speed.
- User said (18:43): "end this and run the more data version". (18:54): "i'm not sure we should be doing isometric loss, can we stop this and run without it?"
- Files worth reusing: checkpoints `im_shared_e40`, `im_cc3m_shared_e40`, `im_cc3m_shared_noiso_e40` (partial training).

## 11-08 19:14 EST Why reconstruction stops improving early: the autoencoder is too small
- Question: "i feel like reconstruction plateaus far too early, what's our architecutre?"
- Setup: the model so far: one hidden layer of 2,048 in the encoder and the decoder, about 17.8M parameters. Compared with PCA on the same states.
- Result (`full_11-08.txt line 8870`): train and validation reconstruction are almost equal (.191 / .197), so the model is underfitting, not overfitting. PCA with 256 dimensions: .176 (block 6) / .290 (block 12). KAE with 256 active coordinates: .170 / .240. PCA with 512 dimensions: .103 / .169.
- Conclusion at the time: the KAE is barely better than PCA at the same number of active coordinates. A bigger encoder and decoder should help.
- Status now: stands; led directly to the 2×4096 model.
- Goal: speed.
- User said: the quote above, then "can you please try a bigger encoder and decoder. do two layers".
- Files worth reusing: none specific.

## 11-08 19:22 EST Bigger autoencoder: two hidden layers of 4,096
- Question: does a wider and deeper encoder and decoder lift accuracy?
- Setup: `y2vhharw` (`im_shared_noiso_2x4k`): 2 hidden layers of 4,096 on each side (about 69M parameters), shared modes, no isometric loss, KL 0.03, ImageNet, 8 epochs, 63 minutes. Matched small run without isometric loss: `kxzt19qo`. Small run with isometric loss: `cv6l70im`.
- Result (`full_11-08.txt line 9165`; wandb):

  | | 1×2048, isometric 1 | 1×2048, no isometric | 2×4096, no isometric |
  |---|---|---|---|
  | zero-shot | 48.59% | 46.50% | 51.45% |
  | agreement | 57.67% | 55.65% | 61.57% |
  | state FVU | .428 | .443 | .369 |
  | recon block 6 / block 12 | .184 / .269 | .198 / .285 | .185 / .229 |
  | train zero-shot | 52.41% | 50.58% | 61.25% |
  | modes on under 1% of images | 15 | 523 | 243 |

  After epoch 1: small 41.4% / 49.9%, big 47.8% / 56.8% (`line 8967`).
- Conclusion at the time: +5 points from size at matched settings. The train-validation gap grows from 4 to 10 points, but validation still rises.
- Status now: stands; 2×4096 became the standard size. A further doubling to 2×8192 gave +1.4 more (53.60%, `882lfgmo`, 11-10) at 4.5 times the training time.
- Goal: speed.
- User said: "can you please try a bigger encoder and decoder. do two layers".
- Files worth reusing: checkpoints `im_shared_noiso_2x4k`, `im_shared_noiso`.

## 11-08 21:45 EST What the isometric loss was doing
- Question: "so isometric was just fixing dead modes?"
- Setup: the three models of the previous entry.
- Result (`full_11-08.txt line 9170` to `9174`): with against without the isometric loss (1×2048): modes used on 1 to 10% of images 784 / 70; largest |λ| 4 to 7 / about 2.3; zero-shot 48.59% / 46.50%.
- Conclusion at the time: partly. It keeps rare modes alive and is worth 2 points on the small model.
- Status now: completed by the next entry: its main job is to stop latent inflation.
- Goal: meaning, speed.
- User said: the quote above.
- Files worth reusing: none specific.

## 11-08 22:04 EST Latent inflation: why the latent loss looked good and then climbed
- Question: "wtf why is latent fvu climbing so hard in this run?!" (run `augtmhvc`, ImageNet + CC3M, 2×4096, no isometric loss: validation latent FVU went .066 to .141 over 5 epochs.)
- Setup: per-image latent energy at block 12 in three models; ImageNet validation against CC3M images.
- Result (`full_11-08.txt line 9267`):

  | | 1×2048, isometric | 1×2048, no isometric | 2×4096, no isometric |
  |---|---|---|---|
  | largest ÷ median image latent energy | 2.4× | 20× | 14,000× |
  | share of energy in the top 1% of images | 2% | 7% | 57% |
  | pooled latent FVU (what the loss sees) | .025 | .0095 | .010 |
  | latent error of a typical image | .010 | .011 | .116 |

  In `augtmhvc`: mean block-12 latent energy per image 1,227 on ImageNet validation against 140,836 on CC3M (about 115 times); latent FVU .12 against .003.
- Conclusion at the time: without the isometric loss the model gives a few images (CC3M graphics on plain backgrounds) enormous latents. The pooled latent FVU is then dominated by those images and looks good while the typical image is predicted 10 times worse. Two earlier statements were withdrawn: "R² .99" and "half the energy in 12 modes" for the 2×4096 model.
- Status now: stands. Fixed by isometric weight 0.1. A per-image size anchor was proposed as an alternative and never built.
- Goal: meaning, speed.
- User said: the quote above, then (22:07) "Bring isometric back at a low weight".
- Files worth reusing: `.../scratchpad/latent_by_source.py`, `.../scratchpad/inflated_sample.py`.

## 11-08 22:13 EST Error by part of the input; matched CLS-only test
- Question: which part of the 2,304-number state is hardest, and how much do patches really add?
- Setup: the 2×4096 model `y2vhharw`; errors split by CLS / patch mean / patch std. Plus a matched 2-epoch pair: CLS only against CLS + patch mean.
- Result (`full_11-08.txt line 9360`):

  | Part | share of block-12 variance | recon block 6 / 12 | prediction FVU |
  |---|---|---|---|
  | CLS | 52% | .109 / .197 | .360 |
  | patch mean | 37% | .162 / .188 | .315 |
  | patch std | 11.5% | .426 / .505 | .580 |

  Matched 2-epoch test: CLS only 36.2% zero-shot / 41.8% agreement; CLS + patch mean 46.9% / 54.4%.
- Conclusion at the time: the spread is the worst-fitted part; the patch summary is worth about 10 points at matched settings.
- Status now: stands.
- Goal: speed.
- User said (22:15): "lets try cls only. i suspect that earlier tests with patch mean + patch std were also including kl loss, which made agreement good. am i wrong? my point is we haven't tried cls only with wide latent". The run was queued, then cancelled; it did not run in this period. (On 11-10 the plan moved to CLS-only state with patches as a control.)
- Files worth reusing: `.../scratchpad/fvu_by_part.py`.

## 11-08 22:09 EST Isometric 0.1 tested on the wrong data, then on the right data
- Question: does isometric weight 0.1 stop the inflation on the ImageNet + CC3M run?
- Setup: first `axlqnvkz` (`im_shared_iso0.1_2x4k`), launched on ImageNet only by mistake; stopped after 2 epochs (49.96% / 59.32%). Then `bzkc5flf` (`im_cc3m_iso0.1_2x4k_e20`): ImageNet + CC3M, 2×4096, isometric 0.1, 20 epochs, started 22:30, 268 minutes. Reference without isometric: `augtmhvc` (stopped after 5 of 20 epochs: 50.71% / 60.62%, latent .1409).
- Result (`full_11-08.txt line 9360` onward): isometric 0 against 0.1: latent error of a typical image .116 / .013; share of energy in the top 1% of images 57% / 2.5%; zero-shot at epoch 2: 49.7% / 50.0%.
- Conclusion at the time: 0.1 removes the inflation and costs nothing in accuracy.
- Status now: stands; isometric 0.1 is the setting of the control model and later runs. Final numbers of `bzkc5flf` are in the 11-09 03:08 entry.
- Goal: meaning, speed.
- User said: "is the current run including cc3m?" then "dude you fucking buffoon, the isometric loss was supposed to test if it would help the cc3m run".
- Files worth reusing: checkpoints `im_shared_iso0.1_2x4k` (2 epochs), `im_cc3m_2x4k_e20`, `im_cc3m_iso0.1_2x4k_e20`.

## 11-08 23:12 EST Mode notebook on the CC3M model: the snow edit at its strongest
- Question: "do we have saved weights for our best ccm version? can we run some notebooks on it?"
- Setup: `augtmhvc` (`im_cc3m_2x4k_e20`, 50.7% / 60.6%). Snow mode 471: AUC .96, on for 11% of validation, λ = 0.34 + 0.31i. Its eigenvalue is multiplied by α. Snow test: 129 snowy and 350 normal photos.
- Result (`full_11-08.txt line 9646`):

  | α | snowy | normal |
  |---|---|---|
  | 1 (as trained) | 32.6% | 56.3% |
  | 0.5 | 40.3% | 56.0% |
  | 0 (removed) | 55.0% | 56.0% |
  | real CLIP | 53.5% | 64.3% |

  Removal fixes 29 of the 129 snowy photos and breaks none of them. Over all 5,635 validation images where the mode is on: 64 fixed / 79 broken (α 0.5: 34 / 33). On Run C the same edit gave 30.2% to 51.2% with 50 fixed / 116 broken. Also: 81% of modes beat random coherence; 6 modes never used; the top 1% of images hold 64% of block-12 latent energy.
- Conclusion at the time: one eigenvalue lifts the snowy photos to CLIP's own level, but across all of validation the edit is a wash, and it repairs an error the KAE itself added (the KAE is 21 points below CLIP on snowy photos before the edit).
- Status now: stands with two cautions. (1) A different collateral count, "78 fixed against 156 broken", was quoted on 11-09 14:50 (`full_11-09.txt line 12139`) with no source; the two do not agree and the second was not traced. (2) Repeated later on `im_cc3m_iso0.1_2x4k_e20` (27.9% to 42.6%) and on `im_iso_2x4k_wd` (30.2% to 49.6% with one mode, 53.5% with two; 11-10).
- Goal: adaptation.
- User said: the quote above.
- Files worth reusing: `/home/nsa325/work/koopmann/second_analysis/archive/modes_im_cc3m_2x4k_e20.ipynb`; figures in `.../scratchpad/figs_augtmhvc`; `.../scratchpad/make_modes_notebook.py`.

## 11-08 23:36 EST The huge modes of the no-isometric CC3M model
- Question: what are the large, almost real eigenvalues in the eigenvalue plot? ("interestingly they're also mostly real")
- Setup: `augtmhvc`.
- Result (`full_11-08.txt line 9701`, `line 9782`): 55 modes with |λ| about 1.2 to 1.37, almost real. They fire on graphics on plain backgrounds (about 10% of CC3M). Their share of block-12 latent energy: 63% on ImageNet validation, 99.7% on CC3M. Capping them at |λ| = 1 or removing them leaves zero-shot at 50.7%. The single real mode has λ about 0.47 to 0.51; it is kept on 4.2% of images (huskies and malamutes); removing it fixes 15 and breaks 29.
- Conclusion at the time: these are the inflation modes. They hold nearly all the latent energy and do nothing for ImageNet accuracy.
- Status now: stands; they disappear with isometric 0.1.
- Goal: meaning.
- User said: asked what the eigenvalue plot shows.
- Files worth reusing: `/home/nsa325/work/koopmann/second_analysis/archive/big_modes_im_cc3m_2x4k_e20.ipynb`; figures in `.../scratchpad/figs_big`; `.../scratchpad/make_big_modes_notebook.py`.

## 11-08 23:43 EST Modes on the unit circle
- Question: "are there any modes that lie on the unit circle, what do they do work on?"
- Setup: `augtmhvc`; modes with |λ| between 0.9 and 1.1.
- Result (`full_11-08.txt line 9844`): 21 of 1,025 modes. 6 are kept on more than 0.5% of images: 436 (cut-out objects), 508, 535, 217 (stock-photo watermarks), 640 (mugs and jugs), 203 (paper and pencils). Most modes have |λ| 0.3 to 0.6 (median 0.47).
- Conclusion at the time: no special role for |λ| = 1 was found; a few such modes are data artefacts of CC3M.
- Status now: stands for that model. The isometric CC3M model has no mode near the unit circle (11-09 03:08).
- Goal: meaning.
- User said: the quote above. Then (23:59): "is the jordan block buying us anything useful or could it be used for something down the line? if i just made them all real eigenvalues, then i could effectively double the number of modes".
- Files worth reusing: `/home/nsa325/work/koopmann/second_analysis/archive/unit_modes_im_cc3m_2x4k_e20.ipynb`; figures in `.../scratchpad/figs_unit`; `.../scratchpad/make_unit_modes_notebook.py`.

---

# 9 November

## 11-09 00:05 EST Is the rotation used? Rotating against scaling the snow eigenvalue
- Question: (user, 11-08 23:59) does the Jordan form, with eigenvalues that rotate, buy anything over all-real eigenvalues?
- Setup: three trained models: the isometric CC3M run at epoch 5 (`bzkc5flf`), the CC3M run without isometric (`augtmhvc`), the small isometric model (`cv6l70im`). Then, on the snow mode of `augtmhvc` (λ = 0.34 + 0.31i), the eigenvalue is either scaled or turned, and the snow test is scored.
- Result (`full_11-09.txt line 9902`):

  | | isometric CC3M | CC3M, no isometric | small, isometric |
  |---|---|---|---|
  | latent energy in modes that rotate more than 15° | 87% | 85% | 87% |
  | typical rotation, weighted by energy | 51° | 52° | 39° |
  | modes that use only one direction of their plane | 1% | 7% | 6% |
  | zero-shot as trained | 51.1% | 50.7% | 48.6% |
  | zero-shot with rotations removed after training | 1.0% | 0.5% | 2.0% |

  Edit to the snow mode, snowy / normal accuracy: as trained 32.6 / 56.3; size ×2: 24.0 / 57.1; size ×0.5: 40.3 / 56.0; size ×0: 55.0 / 56.0; turn +45°: 32.6 / 56.6; +90°: 36.4 / 56.3; −90°: 57.4 / 56.0; 180°: 55.8 / 54.6 (the CLS shrinks 12%). Over 40 random modes a rotation changes 1 to 4% of answers.
- Conclusion at the time: the trained models lean on rotation, but removing it after training only shows damage to a model that expected it; it does not show an all-real model would be worse. As an edit, a turn either does nothing (+45°) or acts like removal (−90°, 180°). No use was found that scaling cannot do.
- Status now: stands. On 11-10 the angle of λ was found not to reproduce across two seeds (rank correlation +.21 to +.26, direction of turn at chance, 49%), while the size does (+.73).
- Goal: meaning.
- User said (11-08 23:59): "is the jordan block buying us anything useful or could it be used for something down the line? if i just made them all real eigenvalues, then i could effectively double the number of modes".
- Files worth reusing: `.../scratchpad/rotation_test.py`, `.../scratchpad/rotate_vs_scale.py`.

## 11-09 00:08 EST Justifying rotation against scaling: an ablation that was proposed, not run
- Question: is there a test that shows the Jordan form matters?
- Setup: none run. An older matched test was recalled: Jordan K 25.7% against a dense K 26.8% (`full_11-09.txt line 9910`). A three-model ablation was proposed (Jordan / real diagonal with the same 2,049 coordinates / dense).
- Result: not measured.
- Conclusion at the time: the Jordan form costs about a point against a dense K and gains nothing measurable yet.
- Status now: never run. The user declined the real-eigenvalue model at 21:51. The MLP-K run (19:25) bounds the cost of the linear K plus TopK at 1.7 points.
- Goal: meaning.
- User said: "we need to really justify having the rotation vs scaling, otherwise it just looks like dressing".
- Files worth reusing: none.

## 11-09 00:50 EST Use cases: fallback by confidence, auditing, steering an image generator
- Question: what is the KAE for? Can it answer most images and hand the hard ones back to CLIP?
- Setup: `bzkc5flf` at epoch 5 (51.1%). The KAE answers when its own top probability is high; otherwise CLIP's real blocks 7 to 12 run. Compute is estimated from layer sizes, **not timed**.
- Result (`full_11-09.txt line 9971`): the KAE's confidence separates "agrees with CLIP" from "disagrees" with AUC .835 (reconstruction error: .48, useless).

  | KAE answers the most confident | zero-shot | agreement on those | compute against CLIP |
  |---|---|---|---|
  | 100% | 51.1% | 61% | 52% |
  | 70% | 58.5% | 83% | 67% |
  | 50% | 61.2% | 93% | 77% |
  | 30% | 62.7% | 98% | 87% |
  | 0% (CLIP alone) | 63.3% | | 100% |

  The saving is capped at about 48% because blocks 1 to 6 always run. Auditing with the KAE would be only about 2 times faster. An image generator built on DINOv3 (RAE) works on patch tokens, so a CLS-only KAE cannot steer it. The idea "a hue shift could be a rotation" was first floated here.
- Conclusion at the time: fallback is the most honest speed story: 2 points of accuracy for about a quarter of the compute.
- Status now: never timed and never repeated on a later model. The speed claim rests on an estimate.
- Goal: speed.
- User said (00:14): "let's step back a bit. so i've been wanting to use this as a tuning/correction model + faster computation. for that, i'm not seeing any reason to include other intermediate steps. i'm also thinking maybe it could help with retrieval tasks (cls is used for that often). However, do you see any other uses cases? do you see ways to make the rotation vs scaling matter a lot more, perhaps via additional reg/losses?" (00:47): "Steering image generators would be crazy. I know that dinov3 has like raev2, which I could maybe steer to generate! fucking wild." and "I don't want to do intermediate blocks, I don't see a reason why including block 9, for example, would at all help us?"
- Files worth reusing: `.../scratchpad/fallback.py`.

## 11-09 01:13 EST Memorising: train against validation
- Question: "the latest run is definitely memorizing but i guess that's still not a terrible sign, at least its learnable. maybe we are overparameterized or maybe we don't have enough data?"
- Setup: `bzkc5flf` (ImageNet + CC3M, 2×4096, isometric 0.1).
- Result (`full_11-09.txt line 10006`), zero-shot train / validation: epoch 1: 50.2 / 47.3; epoch 5: 56.8 / 51.1; epoch 12: 59.3 / 51.3.
- Conclusion at the time: validation is flat after epoch 5 while train keeps rising. More data, not more passes, is what is missing.
- Status now: stands; confirmed by the data-fraction runs (07:05, 09:22).
- Goal: speed.
- User said: the quote above, then (01:16) "i'm going to sleep. can you take over and research ways to improve validation. i'll be up at 9AM est."
- Files worth reusing: none.

## 11-09 01:42 EST Where the accuracy goes, step by step
- Question: of the 12 points between the KAE and CLIP, how many are lost by the autoencoder and how many by the prediction?
- Setup: `bzkc5flf` at epoch 5. Three readings through CLIP's head: real CLIP; block 12 encoded and decoded by the KAE (no prediction); block 12 predicted from block 6.
- Result (`full_11-09.txt line 10190`, `line 10246`), validation / train crops: real CLIP 63.3% / 63.6%; autoencoding block 12 only 60.4% / 60.7%; predicting from block 6: 51.1% / 57.4%. A dense linear map between the two codes, fitted afterwards, gives 49.4% (below the Jordan K). CLIP's answer is in the KAE's top 5 on 85% of images. CLIP's own top probability is 0.43 where the two disagree and 0.80 where they agree.
- Conclusion at the time: about 3 points are lost by squeezing block 12 through 128 modes (same on train and validation, so it is a limit of the autoencoder). About 9 are lost in the prediction, and 6 of those are a train-validation gap. The form of K is not the limit. With this autoencoder the KAE cannot exceed about 60.4%.
- Status now: stands. On 11-10 a plain MLP on the same block-6 summary with the same losses reached 53.0% / 63.6% against the KAE's 52.2% / 62.5%: the KAE's structure costs under 1 point; the rest is what the block-6 summary does not contain.
- Goal: speed.
- User said (02:26): "Interesting that KAE and CLIP disagree where clip is uncertain. maybe we can slightly do kl training to close that up? Just slightly as it has previously harmed us? Tell me more about the 2x2 grid?"
- Files worth reusing: `.../scratchpad/errors.py`.

## 11-09 01:39 EST Probe table: which extra block-6 summaries add information
- Question: would a richer summary of the patch tokens (max, grids, attention pooling) give the KAE more to work with?
- Setup: stored features of the 50,000 validation images, split at random 70% fit / 10% select / 20% test. A ridge map and a small MLP from the block-6 summary to the block-12 CLS. No KAE.
- Result (`/home/nsa325/work/koopmann/temp/probe_fit_mlp.log`), zero-shot / agreement / CLS FVU:

  | Block-6 summary | ridge | MLP |
  |---|---|---|
  | CLS | 6.99% / 8.61% / .778 | |
  | CLS + mean | 15.03% / 18.74% / .699 | |
  | CLS + mean + std (the current state) | 17.99% / 22.27% / .700 | 24.81% / 30.35% / .667 |
  | current + max | 18.21% / 22.43% | |
  | current + 2×2 grid | 19.55% / 23.90% / .746 | |
  | current + 3×3 grid | 19.86% / 24.52% / .749 | |
  | current + attention-pooled | 19.04% / 23.43% / .692 | |
  | current + 2×2 grid + attention-pooled | 20.77% / 25.12% / .750 | 25.74% / 30.91% / .678 |
  | CLS + 2×2 grid + std (mean dropped) | 18.88% / 23.16% / .732 | 24.62% / 29.66% / .692 |

- Conclusion at the time: every richer summary adds 1 to 3 points to a linear map and about 1 point to an MLP. Going from a linear map to an MLP adds 7. The 2×2 grid was chosen for a real run.
- Status now: stands; the grid run (06:48) gained 0.5 points. On 11-10 pooled summaries were replaced by block-6 patch tokens as a control.
- Goal: speed.
- User said: the 02:26 quote above.
- Files worth reusing: `.../scratchpad/probe_extract.py`, `.../scratchpad/probe_fit.py`, `.../scratchpad/probe_fit_mlp.log`, `.../scratchpad/probe_extract.log` (`probe_fit_mlp.py` no longer exists).

## 11-09 03:08 EST The isometric CC3M run finished: numbers and modes
- Question: with isometric 0.1, what does the 20-epoch ImageNet + CC3M model look like inside?
- Setup: `bzkc5flf` (`im_cc3m_iso0.1_2x4k_e20`), 2×4096, KL 0.03, isometric 0.1, shared modes, 20 epochs, 268 minutes.
- Result (wandb; `full_11-09.txt line 10414`): best 51.57% (epoch 11) with 61.6% agreement; last 51.20% / 61.66%; recon .2047; latent .0097; state .3627; train 61.42% / 77.22%.
  - Inflation gone: the top 1% of images hold 1.4% (block 6) / 2.8% (block 12) of latent energy; latent error of a typical image .007.
  - Two modes (436 and 772): almost real, |λ| about 8.2, on about 95% of images, 99% of block-12 latent energy, R² .99. Capping them at |λ| = 1 drops zero-shot 51.6% to 47.4%.
  - Median |λ| 0.28; no mode near the unit circle; all 1,025 modes used; 75% beat random coherence.
  - Snow mode 413: on for 4.9% of validation, AUC .99. Removing it: snowy 27.9% to 42.6%, 19 fixed, none broken, normal flat at 56.9%.
- Conclusion at the time: a healthy model. The 55 inflated modes are replaced by two growth modes that the model needs. The snow edit still works but reaches 42.6%, below CLIP's 53.5%.
- Status now: stands. CC3M was not used again after this run.
- Goal: meaning, adaptation.
- User said: nothing (asleep).
- Files worth reusing: `/home/nsa325/work/koopmann/second_analysis/archive/modes_im_cc3m_iso0.1_2x4k_e20.ipynb`, `/home/nsa325/work/koopmann/second_analysis/archive/big_modes_im_cc3m_iso0.1_2x4k_e20.ipynb`, `/home/nsa325/work/koopmann/second_analysis/archive/unit_modes_im_cc3m_iso0.1_2x4k_e20.ipynb`; figures in `.../scratchpad/figs_final` (16 files).

## 11-09 04:08 EST The control model, and the overnight queue
- Question: what is the one-hour reference that every overnight test is compared with?
- Setup: `ez6cx5a3` (`im_iso_2x4k`): CLIP ViT-B/32, state CLS + patch mean + patch std, 2×4096, 2,049 coordinates, TopK 128 modes shared by both blocks, KL 0.03, embedding 1, isometric 0.1, ImageNet only, 3 crops, 8 epochs, seed 0, 63 minutes. Queue behind it: KL 0.1, 2×2 grid, 25% and 50% of the images, a second seed, weight decay, an MLP in place of K, 2×8192.
- Result (wandb): 52.11% zero-shot, 62.17% agreement; recon .2061; latent .0115; state .3647; KL .8072; train 61.61% / 77.34%. Against the small model: 1×2048 to 2×4096 widened the train-validation gap from 4 to 10 points but lifted validation by 5 (`full_11-09.txt line 10631`).
- Conclusion at the time: this is "the control" for the rest of the period.
- Status now: stands; it is still one of the main checkpoints. A softened KL (temperature 2 to 4) was proposed here and never queued. Side effect noted: a `uv run` re-synced the project's `.venv` (15 packages removed, 38 installed).
- Goal: speed.
- User said: nothing (asleep).
- Files worth reusing: checkpoint `im_iso_2x4k`; `.../scratchpad/night/im_iso_2x4k.log`; `.../scratchpad/night/queue.txt`, `.../scratchpad/night/runner.sh`, `.../scratchpad/night/runner.log`, `.../scratchpad/night/table.py`, `.../scratchpad/night/summary.py`, `.../scratchpad/night/report.md` (the overnight draft report).

## 11-09 05:08 EST KL 0.1 at 2×4096, and what kind of mistakes the KAE makes
- Question: does a little more KL close the gap where CLIP is unsure? What are the KAE's extra mistakes?
- Setup: `wkjot063` (`im_iso_2x4k_kl0.1`): the control with KL 0.1. Error study on the control: validation images where CLIP is right and the KAE is wrong.
- Result (wandb; `full_11-09.txt line 10729`):
  - KL 0.1 against the control: zero-shot 52.63% / 52.11%; agreement 62.35% / 62.17%; recon .2126 / .2061 (3% worse); state .3694 / .3647; train agreement 82.02% / 77.34%. Agreement moves by less than 0.5 point in every bin of CLIP's confidence.
  - CLIP right and KAE wrong on 8,241 validation images; the reverse on 2,629.
  - The lost accuracy is mostly where CLIP is fairly sure: 6 of the 11 points come from images with CLIP top probability .5 to .9.
  - The KAE's wrong answer is a close relative of the right one: median text similarity .76, against .79 for CLIP's own mistakes and .65 for random pairs. 93% of the lost dog images go to another dog breed.
- Conclusion at the time: KL 0.1 gives +0.5 points, mostly by fitting train harder, and does not target the uncertain images. The KAE's mistakes are fine-grained confusions, not nonsense.
- Status now: stands. KL stayed at 0.03.
- Goal: speed.
- User said: the 02:26 quote (asked for slightly more KL).
- Files worth reusing: checkpoint `im_iso_2x4k_kl0.1`; `.../scratchpad/night/kl_uncertain.py`, `.../scratchpad/night/error_kind.py`.

## 11-09 06:00 EST The first grid run crashed (memory), and the runner hid it
- Question: none; a failure.
- Setup: `l8ochbwg` (`im_grid_2x4k`, no checkpoint): state = CLS + four quarter averages + spread (4,608 numbers), 71 GB cache.
- Result (`full_11-09.txt line 10838`): died after extraction. The standardisation subtracted a float32 mean from a half-precision tensor in place; a 4.1 GB tensor peaked at 20.5 GB on a 251 GB machine. The queue runner reported exit code 0 for every job because it read the exit code after running `date`.
- Conclusion at the time: fixed in `scripts/tokae/train.py` by standardising in chunks; the run was restarted as `im_grid_2x4k_r2`.
- Status now: fixed.
- Goal: groundwork.
- User said: nothing.
- Files worth reusing: `.../scratchpad/night/im_grid_2x4k.log`, `.../scratchpad/night/grid_mem.txt`.

## 11-09 07:05 EST A quarter of the images
- Question: how much does accuracy depend on the number of distinct images, at the same number of training steps?
- Setup: `4j3cjq9z` (`im_iso_2x4k_f25`): the control on 25% of ImageNet for 32 epochs.
- Result (wandb; `full_11-09.txt line 10911`): best 45.54% at epoch 6 of 32; then it falls to 44.10% / 52.87% while train agreement climbs to 88.52%. Control 52.11%.
- Conclusion at the time: 6.6 points lost; clear overfitting.
- Status now: stands; part of the data curve (09:22).
- Goal: speed.
- User said: nothing.
- Files worth reusing: checkpoint `im_iso_2x4k_f25`; `.../scratchpad/night/im_iso_2x4k_f25.log`.

## 11-09 08:22 EST The 2×2 grid state
- Question: does a coarse spatial layout of the patches help?
- Setup: `qhodopmm` (`im_grid_2x4k_r2`): state = CLS + the average of each image quarter + spread (4,608 numbers per block); otherwise the control. 89 minutes against 63; 71 GB cache.
- Result (wandb; `full_11-09.txt line 11013`): 52.62% / 62.88% against 52.11% / 62.17%; train 62.11% / 78.37%; linear-map baseline 20.43% against 18.4%. After epoch 1: 48.5% against 47.6%.
- Conclusion at the time: +0.5 zero-shot and +0.7 agreement, for 1.4 times the training time.
- Status now: real (above seed noise) but small; not adopted. The grid caches were deleted at 14:35. Superseded on 11-10 by patch tokens as a control.
- Goal: speed.
- User said (02:26): "Tell me more about the 2x2 grid?"
- Files worth reusing: checkpoint `im_grid_2x4k_r2`; `.../scratchpad/night/im_grid_2x4k_r2.log`.

## 11-09 09:22 EST The data curve: about 3 points per doubling
- Question: how fast does accuracy grow with the number of distinct images?
- Setup: `3dhdyep7` (`im_iso_2x4k_f50`, 50% of ImageNet, 16 epochs) joins the 25% run and the control; all at the same number of steps.
- Result (wandb; `full_11-09.txt line 11066`): 25%: best 45.54% (epoch 6 of 32); 50%: best 48.98% (epoch 10 of 16; last 48.48% / 57.75%, train agreement 83.11%); 100%: 52.11%. Steps of +3.4 and +3.1 points per doubling. For comparison, 1 crop to 3 crops gave +1.3 (about +0.8 per doubling of crops).
- Conclusion at the time: the model is limited by distinct images. If the trend held, twice ImageNet would give about 55%; the autoencoder's own limit is about 60.4%.
- Status now: stands as a measurement. The trend was not confirmed by adding other images (CC3M and ImageNet-21k gave no gain at equal steps), so "3 points per doubling" holds for ImageNet images only.
- Goal: speed.
- User said (10:45): "So how do we double images? Imagenet 21k is a lot more classes no?"
- Files worth reusing: checkpoint `im_iso_2x4k_f50`; `.../scratchpad/night/im_iso_2x4k_f50.log`.

## 11-09 10:26 EST A second seed: how big is the noise?
- Question: which of the night's small differences are real?
- Setup: `32mst724` (`im_iso_2x4k_s1`): the control with seed 1.
- Result (wandb; `full_11-09.txt line 11115`): 52.11% / 62.16% against 52.11% / 62.17%. Per epoch the two stay within 0.3 points; recon and state FVU differ by about .001.
- Conclusion at the time: differences of 0.5 points at the last epoch are probably real: KL 0.1 (+0.5), the grid (+0.5 / +0.7), isometric 0.1 against 0 (52.1 against 51.5).
- Status now: stands. The pair of seeds was reused on 11-10 for the test of whether size and angle of λ reproduce.
- Goal: groundwork.
- User said: nothing.
- Files worth reusing: checkpoint `im_iso_2x4k_s1`; `.../scratchpad/night/im_iso_2x4k_s1.log`.

## 11-09 11:01 EST More images: ImageNet-21k (called "w21")
- Question: can images from other ImageNet-21k classes stand in for "twice ImageNet"?
- Setup: `timm/imagenet-w21-wds` (13.15M images in 2,048 files). The first 200 files were downloaded (the talk says 114 GB at `full_11-09.txt line 11209` and 107 GB later; both are given). The ImageNet-1k classes were dropped (about 10% of each file), leaving 1,158,002 images. Near-copies of validation images were dropped by CLIP similarity: 321 above 0.99; 2,336 above 0.95, leaving 1,155,666; about 75,000 that are merely similar (above 0.9) were kept (`line 11468`). One centre crop per new image. New loader `ShardImages`, option `--extra_dataset=w21`. A test proposed first (train on 500 classes, score on the other 500) was declined.
- Result: data ready at 11:38.
- Conclusion at the time: about +90% distinct images, no validation leak above 0.95 similarity.
- Status now: the set is still on disk and was used for the DINOv3 runs.
- Goal: speed.
- User said: "So how do we double images? Imagenet 21k is a lot more classes no?"; "Don't do the test, let's just try some more images. How to get access?"; "Done"; "200 ifiles".
- Files worth reusing: `.../scratchpad/night/prep_w21.py`, `.../scratchpad/night/download_w21.py`, `.../scratchpad/night/w21_dedupe.pt`, `.../scratchpad/night/w21_dedupe_dinov3-vitb16.pt`.

## 11-09 11:19 EST Weight decay at 2×4096
- Question: "weight decay is really helping reconstruction and kl divergence". Is it?
- Setup: `4ujgz6hv` (`im_iso_2x4k_wd`): the control with weight decay 0.1 on the encoder and decoder matrices only. The eigenvalues are not decayed (`scripts/tokae/train.py`, `get_optimizers`; the user asked at 11:04).
- Result (wandb; `full_11-09.txt line 11395`, `line 11468`), weight decay against the control: zero-shot 52.17% / 52.11%; agreement 62.47% / 62.17%; recon .1904 / .2061 (8% better); KL .7458 / .8072 (7% better); latent .0196 / .0115 (worse); train agreement 70.15% / 77.34% (gap to validation 7.7 against 15 points). At epoch 7: latent error of a typical image .0165 against .0088.
- Conclusion at the time: a net positive: same accuracy, better reconstruction, half the overfitting. Adopted for the following runs.
- Status now: **later reversed.** At 12:19 the weight-decay model was seen to use only 479 of 1,025 modes. On 11-10 this was confirmed across models (dead modes of 1,024: 545 here, 445 at 2×8192, 567 for DINOv3, 735 with the real-start rule; 0 without weight decay). The user: "Dead modes are no good", "yeah we should drop weight decay then". Runs after 11-10 use no weight decay.
- Goal: speed, meaning.
- User said (11:04): "also are we doing weight decay on the jordan blocks? i think that makes no sense". (11:18): "weight decay is really helping reconstruction and kl divergence". (11:20): "yeah test it with wd on the new job".
- Files worth reusing: checkpoint `im_iso_2x4k_wd`; `.../scratchpad/night/im_iso_2x4k_wd.log`.

## 11-09 12:19 EST Modes that most images share
- Question: "do we ever end up with modes that most images share?"
- Setup: the control `im_iso_2x4k`; modes kept on more than 95% of validation images.
- Result (`full_11-09.txt line 11571`): 21 such modes. They hold about 94% of block-6 and 99% of block-12 latent energy. Each image adds about 108 modes of its own.
  - Three growth modes (321, 466, 26): nearly real (about 4°), 99.9% of their energy along one line, |λ| 5.5 to 6.7, 95% of block-12 latent energy. Each has two opposite ends: 321: electronics on white against small animals in grass; 466: wide outdoor scenes against close-ups of yellow and orange food; 26: blue sky or sea with one small object against close-ups of dogs and people. Fixing them at their average drops zero-shot 52.1% to 38.8%.
  - Eighteen small always-on modes carry block 6's offset: decoded alone they give block 6's average state (cosine .97); |λ| about 0.45, rotating 20 to 45°; their size tracks how uniform the patches are (correlation −.63). Fixing them at their average costs 6 points.
  - In the weight-decay model only 479 of 1,025 modes are ever used.
- Conclusion at the time: yes. A few always-on modes carry the growth from block 6 to block 12 and a coarse "what kind of picture is this" map; the rest are per-image.
- Status now: stands. On 11-10 the DINOv3 model showed the same split (10 offset modes plus 2 growth modes).
- Goal: meaning.
- User said: the quote above. Earlier (11-08 18:32): "i think its okay to have some modes handle the norm growth. not every mode has to be interpretable."
- Files worth reusing: `.../scratchpad/figs_shared/axis_im_iso_2x4k_321.jpg`, `.../scratchpad/figs_shared/axis_im_iso_2x4k_466.jpg`, `.../scratchpad/figs_shared/axis_im_iso_2x4k_26.jpg` (photo sheets: check each for hidden images before reuse); `.../scratchpad/night/shared_modes.py`, `.../scratchpad/night/shared_modes2.py`, `.../scratchpad/night/shared_what.py`, `.../scratchpad/night/shared_pca.py`, `.../scratchpad/night/shared_wheel.py`, `.../scratchpad/night/shared_im_iso_2x4k.pt`.

## 11-09 12:41 EST ImageNet + 21k images at equal steps, and the corrected reading
- Question: do the extra 1.16M images help?
- Setup: `zvtaer5s` (`im_w21_wd_2x4k`): ImageNet (3 crops) + w21 (1 crop), weight decay 0.1, 6 epochs (the same number of steps as 8 epochs of ImageNet). Reference: `im_iso_2x4k_wd`.
- Result (wandb; `full_11-09.txt line 11786`, `line 11802`): weight decay alone against + 21k: zero-shot 52.17% / 51.76%; agreement 62.47% / 62.24%; KL .7458 / .7479; recon .1904 / .1942; train agreement 70.15% / 67.43%. At the same number of passes over ImageNet (6): 51.1% against 51.8%, in favour of the 21k run. Gain in the last epoch +0.5 against +1.1; drop in KL in the last epoch −.026 against −.044; train-validation agreement gap 7.7 against 5.2.
- Conclusion at the time: first written as "didn't help". Corrected after the user objected: at equal passes over ImageNet the 21k run is ahead, it overfits less and was still improving faster when it stopped.
- Status now: open. The pair of longer runs that would settle it (16 against 12 epochs) was proposed and never run on CLIP.
- Goal: speed.
- User said: "you are dumb, the extra images certainly helped, we're just undertriaing."
- Files worth reusing: checkpoint `im_w21_wd_2x4k`; `.../scratchpad/night/im_w21_wd_2x4k.log`.

## 11-09 12:32 EST Switch to DINOv3: setup
- Question: "i really want to do this on dinov3's cls token. can we please swap to that?"
- Setup (`full_11-09.txt line 11755`): DINOv3 ViT-B/16, blocks 7 to 12 replaced, same state (CLS + patch mean + patch std) and same KAE as the control. The four register tokens are left out of the patch statistics (register norm about 3,880 against about 40 for patches). The judge is a published linear probe on the final CLS, since DINOv3 has no text head; the wandb names `zero_shot_accuracy` and `clip_agreement` are kept. `transformers` was installed in a side folder because the project `.venv` cannot import it (4.57.3 next to `huggingface-hub` 2.4.0). DINOv3 extraction runs at 420 to 470 images per second against 1,650 for CLIP.
- Result: first run `ulovynus` (`im_dino_2x4k`, no checkpoint) was ImageNet only without weight decay; stopped after extraction at the user's request.
- Conclusion at the time: rerun with weight decay and the 21k images.
- Status now: superseded by `im_dino_w21_wd`.
- Goal: speed.
- User said: the quote above; then (13:22) "is the dinov3 run doing imagenet-1k only with wd?" and "i want weight decay with the new images jfc."
- Files worth reusing: `.../scratchpad/night/im_dino_2x4k.log`, `.../scratchpad/night/dino_job.sh`, `.../scratchpad/night/dino_w21_job.sh`.

## 11-09 14:35 EST Disk clean-up
- Question: "If there is trash we can clean up to save memory, let's do so"
- Setup: caches no longer needed.
- Result (`full_11-09.txt line 12056`): 167 GB deleted: the old CLS-only, CLS + patch and grid caches (156 GB) and 11 GB of scratch tensors. Disk at 42% afterwards.
- Conclusion at the time: done.
- Status now: the CLS-only, CLS + patch-mean and grid models from this period cannot be re-scored without extracting their states again.
- Goal: groundwork.
- User said: the quote above.
- Files worth reusing: none.

## 11-09 14:46 EST What could rotation against scaling be for? Ideas and a first literature search
- Question: is there a use that gives the angle of λ a meaning of its own, and a pitch that sets the KAE apart from sparse autoencoders?
- Setup: discussion and a search of paper abstracts (not full reads). Nothing was trained.
- Result (`full_11-09.txt line 12122` to `line 12373`): "Right now only scale has a demonstrated use". Ideas put forward, none run:
  - separate axes for CLS content and patch content;
  - size = content, phase = viewpoint;
  - a budget on total rotation;
  - a cyclic change (hue) as rotation and blur as decay, on a second set of eigenvalues;
  - adapting to a new domain by tuning only angles against only sizes;
  - an "accurate mode": add the KAE's edit to the true block-12 state, so the stand-in's own error drops out.
  Papers named as related: angular steering (arXiv 2510.26243), Householder pseudo-rotation, crosscoder diffing, Redman et al., Rotating Features, cross-layer transcoders for ViTs (arXiv 2604.13304).
- Conclusion at the time: the angle has no natural meaning in the trained models; one would have to be imposed through a loss or a constraint.
- Status now: led to the real-start rule (22:22). None of the listed ideas was tested in this period.
- Goal: meaning.
- User said: "Okay while those run, I want to return to the question of how to make the jordan block useful. I would like have a meaningful use case for rotation vs scale in the eigenvalues. Additionally, some good pitches for why this is a meaningful structure and what sets it apart from sparse autoencoders". "Typically, its really hard to see things naturally emerge in the latent space. I think we probably want to imbue meaning, so we have to include some losses/reg and be creative on how to encourage them to do something meaningful." "None of these seem to have a meaningful use case for practitioners. We want to really exploit the jordan block structure to see if we can assign special meaning to rotation vs scale." "I'm not sure I fully understand your plan and why it would be useful, can you explain more simply without losing vocab?" "I would like a use that gives the depth eigenvalue's angle a meaning of its own. Maybe we can look online for some inspiration?" "Bsaed on your reserach do you see a good way for us to standout from previous work and deliver something meaningful? So far, I've been thinking about it as a way to speed up computation + adapt to new environments /fix some overzealous features" and "We do have to assume that we will train for longer with more data. All of our work so far has just been pilot runs of a few epochs."
- Files worth reusing: none.

## 11-09 15:49 EST Second literature search
- Question: "let's do more research and see how we can further stand out, please look for more work relevant that we could beat or use to inspire us"
- Setup: abstracts and partial reads; nothing was verified in full.
- Result (`full_11-09.txt line 12670`): Dhor & Chen 2026 (arXiv 2608.10172); Fernando & Guitchounts 2026 (arXiv 2605.14258; in language models about 98% of per-block eigenvalues come as complex pairs); Transformer² (tunes sizes only); OFT / BOFT (tune rotations only); ReplaceMe; a ViT pruning study; cross-layer transcoders (CLIP B/32 on ImageNet-100: 80.4% falls to 60 to 72%); SAE steering and RoboShot; BATCLIP / TENT; the distilled DINOv3 ViT-S scores 81.4%. Three next steps were proposed: an all-real-eigenvalue KAE, a pilot of tuning eigenvalues on a shifted domain, and the KAE's accuracy on ImageNet-100 for comparison with the transcoder paper.
- Conclusion at the time: the opening is "one object whose eigenvalues can be tuned by size or by angle"; no published work does both.
- Status now: the three steps were given unclear names and rejected or left (21:51). The small distilled model (81.4%) is a speed baseline the KAE has not been compared with.
- Goal: meaning, adaptation.
- User said: the quote above.
- Files worth reusing: none.

## 11-09 18:12 EST DINOv3 result
- Question: how well does the same KAE stand in for DINOv3's blocks 7 to 12?
- Setup: `urzqs9i6` (`im_dino_w21_wd`): DINOv3 ViT-B/16, ImageNet (3 crops) + w21 (1,158,002 images extracted in 48 minutes; 363 near-copies of validation dropped, leaving 1,157,639; `full_11-09.txt line 12709`), weight decay 0.1, 12 epochs, otherwise the control's settings.
- Result (wandb; `full_11-09.txt line 12397`, `line 12739`): 61.78% against 83.94% for the real model and 27.16% for a linear map; agreement 66.49%; train 69.25% / 73.09%; recon .1444; latent .0227; state .4387 (against .73 for the linear map); embedding .3005. Validation rose every epoch, 52.2% to 61.8%, with +0.6 to +0.7 in each of the last three.
- Conclusion at the time: 22 points below the real model, a bigger gap than CLIP's 11, but still rising when it stopped.
- Status now: extended on 11-10 to 30 epochs (`ze4m5akv`): 62.90% / 67.91%, train 72.30%. So 2.5 times the epochs bought about 1 point. The linear-map baseline here (27.2%) is lower than the 35% of the quick check on 11-08; the difference was never traced.
- Goal: speed.
- User said (23:22): "dinov3 run for 11 epochs is our best performing one yet right? can we explore it to see how its spectrum looks, how snow works, any interesting features/modes?"
- Files worth reusing: checkpoints `im_dino_w21_wd`, `im_dino_w21_wd_e30`; `.../scratchpad/night/im_dino_w21_wd.log`; `/home/nsa325/work/koopmann/second_analysis/archive/dino_modes_im_dino_w21_wd.ipynb` (built from 23:22 on; its findings belong to 11-10).

## 11-09 19:25 EST An MLP in place of K (no TopK)
- Question: how much accuracy does the linear, sparse K cost?
- Setup: `51bwecvl` (`im_mlpk_2x4k`): the control's encoder and decoder, with a residual MLP between the two codes and no TopK. No weight decay.
- Result (wandb; `full_11-09.txt line 12780`): 53.79% / 64.21% against 52.11% / 62.17%; state .3487 against .3647; recon .0633 against .2061; train 62.27% / 77.34%.
- Conclusion at the time: +1.7 points. The run changes two things at once (the operator and the sparsity), so 1.7 is an upper bound on what the linear K costs.
- Status now: stands; agrees with the 11-10 probe (the KAE's structure costs under 1 point).
- Goal: speed, meaning (the price of having eigenvalues at all).
- User said: nothing tied to it.
- Files worth reusing: checkpoint `im_mlpk_2x4k`; `.../scratchpad/night/im_mlpk_2x4k.log`.

## 11-09 21:51 EST Evening summary, and feedback on unclear labels
- Question: "So give me a summary"
- Setup: no new run. In passing: the snow mode of the weight-decay control has size .46 and angle 42°.
- Result (`full_11-09.txt line 12793` to `line 12811`): the summary used three labels the user did not accept.
- Conclusion at the time: the assistant: "Those were my own labels, not standard terms".
- Status now: the rule "define any label at first use, no made-up terms" comes from here.
- Goal: groundwork.
- User said: "1. I'm not sure what you mean by "angle adapts without forgetting while size removes or boosts, and angles add when you stack two adaptations." This is very vague. 2. I don't want to run a real-eigenvalue-only KAE. I'm sure it will work. It's a waste of time. The goal is to imbue the Jordan structure. 3. What the fuck is "eigenvalue-tuning pilot"? 4. What the fuck is "imagenet-100 number"?"
- Files worth reusing: none.

## 11-09 22:22 EST The real-start rule is proposed
- Question: how can the angle get a meaning for every mode, not just for a handful of hand-made alterations?
- Setup: a training constraint in two parts (`full_11-09.txt line 12830`, `line 13027`). (1) Each mode is a pair of numbers; at block 6 the second number is forced to zero, at block 12 both are allowed. (2) An extra loss term ties the geometry of the codes at the two blocks to the geometry of the states. Intended reading: size of λ = how much blocks 7 to 12 amplify the feature ("volume"); angle of λ = how much they change it, from 0° (passed through, "kept") to 90° (replaced, "built").
- Result: not yet run in this period.
- Conclusion at the time: the user agreed to try it.
- Status now: tested on 11-10 (`butbn6t1`: 50.90% / 61.07%; second seed `wjx7afbv`: 51.07% / 61.03%; reference `im_iso_2x4k_wd` 52.17% / 62.47%). The rule costs about 1.3 points. The angle still does not reproduce across seeds and no property was found that tracks the angle once size is held fixed.
- Goal: meaning.
- User said (22:16): "my problem with your approach is that we can only create a handful of those "alterations" (blurred, grayscale, etc). We need something scalable and something that is meaningful across all/most of the features". (22:43): "going back to angles: i'm interested in the idea that block start is only strength, no angle, while later block works through angle. but i'm not sure what you mean by "fully built" do you expect that the feature will evolve as the angle is changed? that could be interesting but wil lthat be useful? in terms of pitching, it hink we have to take a multi prong approach of delivering faster results of the clip/dino while providing some tuning knobs that are interpretable." (22:54): "what is the "rule"? is it the angle idea?" (23:07): "let's try it out".
- Files worth reusing: checkpoints `im_realstart_2x4k_wd`, `im_realstart_2x4k_wd_s1`; `.../scratchpad/test_real_start.py`; `.../scratchpad/night/im_realstart_2x4k_wd.log`.

## 11-09 22:29 EST The 2×8192 run without weight decay is killed
- Question: does doubling the width again help?
- Setup: `l4sdh1k1` (`im_iso_2x8k`): the control at 8,192 wide. It had been queued at 04:08, before weight decay was adopted, and was never updated.
- Result (wandb; `full_11-09.txt line 12972`): killed at 22:29 after 5 of 8 epochs (177 minutes): 52.43% / 62.69%; at the same epoch the control had 51.7% / 61.4%.
- Conclusion at the time: about +0.7 at epoch 5. Restarted with weight decay as `im_iso_2x8k_wd`.
- Status now: `882lfgmo` finished on 11-10: 53.60% / 64.48% (4 h 40 min against 63 min). Since weight decay was later dropped, the killed run is the only 2×8192 model without it, and it is unfinished.
- Goal: speed.
- User said: "man also, why is the current run training without weight decay, didn't we make it clear that weight decay is a net positive for us?" then "kill the current run jfc".
- Files worth reusing: checkpoints `im_iso_2x8k` (5 epochs), `im_iso_2x8k_wd`; `.../scratchpad/night/im_iso_2x8k.log`, `.../scratchpad/night/im_iso_2x8k_wd.log`.

## 11-09 23:21 EST A measurement before the rule: block 6 and block 12 are almost perpendicular
- Question: what should the angle come out as, on average, if the rule works?
- Setup: 4,096 validation images in the standardised space the KAE sees (`full_11-09.txt line 13210`).
- Result: block-12 states are spread about 17 times more than block-6 states (mean squared distance from the block's average 2,482 against 143). An image's block-6 and block-12 states, each measured from its block's average, are almost perpendicular (cosine 0.07). So the average angle should come out near 86°. The distance from a block-6 state to a block-12 state is the same for the same image as for different images (50.1 against 51.0), so matching distances cannot work; the extra term matches inner products instead, weight 0.1.
- Conclusion at the time: the risk is that every mode's angle piles up near 90°. The run was queued anyway: `im_realstart_2x4k_wd` (ImageNet, 3 crops, 8 epochs, 4,096 wide, weight decay 0.1). Code added to `scripts/tokae/train.py`: `--real_start`, `--lambda_overlap`.
- Status now: see the previous entry but one. The geometric fact (17 times the spread, cosine 0.07) stands and matters for any reading of the angle.
- Goal: meaning.
- User said: "let's try it out".
- Files worth reusing: `.../scratchpad/cross_block_geometry.py`.

## 11-09 23:39 EST DINOv3 mode notebook started
- Question: "can we explore it to see how its spectrum looks, how snow works, any interesting features/modes?"
- Setup: a notebook on `im_dino_w21_wd`, executing from 23:39 to 23:58.
- Result: not reported before midnight; the findings belong to 11-10 (see the next part of the history).
- Conclusion at the time: none yet.
- Status now: reported on 11-10: the always-on modes are 10 offset modes plus 2 growth modes; DINOv3's ImageNet mistakes can be flagged, not fixed.
- Goal: meaning, adaptation.
- User said: the quote above.
- Files worth reusing: `/home/nsa325/work/koopmann/second_analysis/archive/dino_modes_im_dino_w21_wd.ipynb`; `.../scratchpad/make_dino_modes_notebook.py`.

---

# Closing lists

## 1. The findings of this period that matter most, ranked

1. **What the KAE is given at block 6 matters most.** CLS only 39.0%; + patch mean 49.5%; + patch spread 51.2% (all at KL 1). Matched 2-epoch test at 2×4096: 36.2% against 46.9%. A linear map goes 7% / 16% / 18% on the same inputs.
2. **A bigger encoder and decoder is worth 5 points.** 1×2048 to 2×4096: 46.5% to 51.5% at matched settings; the control is 52.1% / 62.2%. Doubling again: 53.6% at 4.5 times the training time.
3. **The model is short of distinct images, not of training time.** 25% / 50% / 100% of ImageNet: 45.5% / 49.0% / 52.1%, about +3 points per doubling. Five times more passes: +0.4 (47.7% to 48.1%). Two seeds differ by 0.01 point at the end.
4. **Where the 11 points go.** Passing the true block 12 through the autoencoder alone gives 60.4% (3 points lost, same on train). Prediction from block 6 loses the rest. The mistakes are near misses: CLIP's answer is in the KAE's top 5 on 85% of images; 93% of lost dog images go to another breed.
5. **The linear K with TopK is cheap.** An MLP without TopK in its place: 53.8% against 52.1% (an upper bound of 1.7 points). A dense map between the codes: 49.4% against 51.1%.
6. **Isometric loss at 0.1 stops latent inflation at no cost.** Without it, the top 1% of images hold 57% of latent energy and the typical image's latent error is .116; with it 2.5% and .013. The pooled latent FVU hid the problem (.010 in both).
7. **Shared modes make the eigenvalues honest.** Median R² between λ and the measured growth of a mode: −6.9 to .40; nearly dead modes 348 to 15; +0.9 points against the 8-epoch reference (47.72% to 48.59%), +1.0 against its rerun (47.62%); the talk said +0.8.
8. **KL is a trade.** Weight 1 buys 13 points and triples the reconstruction error. Weight 0.03 buys 3.8 points over none for +1.7% reconstruction error.
9. **One eigenvalue fixes snowy photos on a hand-picked set, and only there.** 32.6% to 55.0% on 129 snowy photos (CLIP 53.5%), none broken in the test. Over all validation images with the mode on: 64 fixed, 79 broken. The edit repairs an error the KAE itself added.
10. **A few always-on, nearly real modes carry the growth.** Three modes hold 95% of block-12 latent energy; fixing them at their average costs 13 points (52.1% to 38.8%). They sort pictures by type.
11. **Rotation is used but has no demonstrated use.** 85 to 87% of latent energy is in modes that turn more than 15°, yet turning the snow eigenvalue either does nothing or acts like removal.
12. **DINOv3: 61.8% of 83.9% after 12 epochs,** still rising by 0.6 to 0.7 per epoch.
13. **Fallback by confidence** gives 61.2% at an estimated 77% of CLIP's compute. The compute was never timed.

## 2. Things tried that did not help (each with its number)

- Higher learning rate (×2.7): 49.55% against 49.55% (`71cjcg4j` / `w5zkvf1x`).
- One crop for more epochs: 48.2% against 49.5% (`b6kv3mne`).
- 4,097 latent coordinates on the CLS-only model: 39.2% against 39.0% (`7t4irsvy`).
- MLP in place of K on the CLS-only model: 40.3% (`p2jnnyzz`).
- 40 epochs in place of 8: 48.1% against 47.7% (`jm17ztqa`).
- BatchTopK: 45.8% against 47.7% (`r0epg244`).
- Adding CC3M at equal steps: 46.3% against 48.0% (`kfu2tcdk`); at 2×4096, 20 epochs: 51.6% best against 52.1% (`bzkc5flf` / `ez6cx5a3`).
- Adding 1.16M ImageNet-21k images at equal steps: 51.8% against 52.2% (`zvtaer5s`); disputed by the user, not settled.
- Centring and per-block standardisation: 48.5% each against 47.7%, but reconstruction up to 34% and 51% worse (`mwfzxnwj`, `mfq4foq4`). Shared modes + per-block: collapse, 45.2% (`i0v80lf4`).
- Removing the isometric loss: 46.5% against 48.6% on the small model (`kxzt19qo`); latent inflation on the big one.
- Weight decay: small model 50.4% at epoch 18 against 51.2% best (`lm09dz9a`); 2×4096: 52.2% against 52.1% (`4ujgz6hv`), later found to leave about half the modes unused.
- KL 0.1 in place of 0.03 at 2×4096: +0.5 point, reconstruction 3% worse, train agreement +4.7 (`wkjot063`).
- 2×2 grid state: +0.5 point for 1.4 times the training time and a 71 GB cache (`qhodopmm`).
- Capping or removing the 55 inflated modes: 50.7% unchanged. Capping the two growth modes: 51.6% to 47.4%.
- Turning off a context mode to fix a known confusion (baseball, Pickelhaube): no fix. Turning off any single mode: never improves overall accuracy.
- Turning off the grayscale / sepia / blur modes on altered photos: about +1 point.
- Rotating the snow eigenvalue: +45° no change (32.6%); other turns act like removal.
- A dense linear map between the codes fitted after training: 49.4% against 51.1%.

## 3. Mistakes made and how they were corrected

- KL was put back at weight 1 overnight without checking its cost on the other losses. Corrected at 09:31 (reconstruction 0.10 against 0.29) and by the KL sweep.
- Overnight choices were made without asking: KL back in, 40 epochs in place of the 200 asked for, an unrequested weight-decay run. Reported the next morning.
- The four huge eigenvalues were called "bookkeeping". Withdrawn at 16:41: their sizes vary per image and correlate .98 across blocks.
- "R² .99" and "half the energy in 12 modes" for the first 2×4096 model were artefacts of latent inflation. Withdrawn at 22:04.
- "CC3M didn't help" was too strong (fewer passes over each ImageNet image). Softened at 18:32.
- The isometric 0.1 test was launched on ImageNet only when the question was about the CC3M run. Stopped after 2 epochs and rerun on the right data.
- "The ImageNet-21k images didn't help" was too strong. Corrected at 13:07 after the user objected.
- The 2×8192 and MLP-K runs were queued before weight decay was adopted and never updated; the 2×8192 run was killed after 5 epochs.
- Weight decay itself was adopted on 11-09 on the strength of reconstruction and KL, without counting used modes. Reversed on 11-10.
- The first grid run crashed from a memory spike in the standardisation; fixed by working in chunks. The queue runner reported exit code 0 for every job.
- A `uv run` re-synced the project `.venv` (15 packages removed, 38 installed).
- The BatchTopK result (2 points worse) was never reported to the user.
- Undefined labels in a summary ("eigenvalue-tuning pilot", "ImageNet-100 number", "the rule"). Called out by the user at 21:51 and 22:54.
- The snow damping scale 0.5 was picked on the same images it was scored on. Never corrected.
- The confounder script crashed (IndexError) after printing its table.
- Two snow collateral counts are in the record and disagree: 64 fixed / 79 broken (`full_11-08.txt line 9646`) and 78 fixed / 156 broken (`full_11-09.txt line 12139`). Not reconciled.

## 4. Questions raised and never answered

- Would an all-real-eigenvalue KAE with twice the modes do as well? (Proposed three times; the user declined: "I'm sure it will work.")
- Does tuning only angles adapt differently from tuning only sizes on a shifted domain?
- CLS-only state with the wide autoencoder (asked 11-08 22:15, queued, cancelled; answered only in a different form on 11-10).
- Do the extra ImageNet-21k images help with enough training? (The 16-against-12-epoch pair was never run on CLIP.)
- Does "3 points per doubling" continue past ImageNet's own images?
- How fast is the KAE in wall-clock time, alone and with fallback? (Only estimated from layer sizes.)
- Can the spectrum be shaped to avoid giant eigenvalues? (Answered "no" by argument, never tried.) A per-image size anchor in place of the isometric loss was never built.
- A softened KL (temperature 2 to 4) aimed at images where CLIP is unsure.
- The CLS history (blocks 1 to 6) as input.
- DINOv3 with its register tokens in the state (the quick check suggested +2 to +4 points for a linear map).
- Why the DINOv3 linear-map baseline is 27.2% in the run and about 35% in the quick check.
- More than 3 stored crops per image.
- Is the snow fix better than a simple baseline, such as removing a "snow" direction from CLIP's final embedding?
- The ideas for giving the angle a meaning that were listed and not run: hue as rotation and blur as decay; CLS against patch axes; a rotation budget; the "accurate mode".
- How the KAE compares with a small distilled model (DINOv3 ViT-S, 81.4%) as a speed baseline.
- Do the units of the MLP-K model, which has no TopK, still look like nameable modes?
