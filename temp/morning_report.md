# Overnight report, 2026-11-10

Terms used below. A **mode** is one 2×2 block of K: per image it holds a complex number z (a size and a **phase**), and K multiplies it by the eigenvalue λ (a **size** and an **angle**). **KAE accuracy** is ImageNet val top-1 when the real model's head reads the KAE's predicted block-12 state; **agreement** is how often that answer equals the real model's answer.

## The three goals, in one line each

1. **Inversion: done, with a mixed result.** The KAE's prediction paints back the right scene. Removing one mode does not visibly remove its quality, except for the fire mode. The gray and sepia knobs work on some photos.
2. **Angles and scaling: two of the three numbers have a meaning; the angle only gets one under the real-start rule.** Size of λ and phase of z mean something and reproduce across seeds. The angle of λ does not, unless the rule you approved pins it; then it says how much of a mode blocks 7–12 pass along versus rewrite.
3. **Performance: longer training buys about one point.** The 30-epoch DINOv3 run finished at 62.9 / 67.9, against 61.8 / 66.5 for the 12-epoch run. More epochs on the same data is not the lever.

## 1. Inversion

What was done, and how it differs from Splice: DINOv3 ViT-B/16 in place of DINO ViT-B/8; the target is the whole block-12 state (CLS, patch mean, patch std), not the CLS alone; about 3,000 steps on the CPU in place of 20,000; edited pictures continue from the unedited picture so the layout stays put. 106 pictures from 9 validation photos, on `im_dino_w21_wd`.

- **Scenes come back.** Painting the KAE's prediction gives winter woods, snowy woods, a citrus pile, open water, striped socks and a volcano. Object detail is weaker than when painting DINOv3's own state (state error left .160 against .078).
- **One-mode edits mostly do not show.** They arrive in DINOv3's reading of the painted picture (74–88% of the asked change), but a control mode does almost as well (73%). CLIP, as an outside judge, sees the mode's word move the expected way in 6 of 9 photos when removed and 5 of 9 when doubled, which is chance.
- **The exception is mode 211 (fire or glow).** Removing it dims the fire in both of its photos; doubling adds glow.
- **Snow stays when the snow mode is removed.** This fits the knob result below: the mode is the classifier's snow cue, not the snow in the scene.
- **Colour knobs** (every mode multiplied by its gray or sepia factor): the fruit photo loses colour (colourfulness .53 to .38), water and sky lose it with the sepia knob, socks and the volcano do not.
- **A limit of the painting itself:** DINOv3's own state of a gray copy still paints in colour, and oranges come out lemon yellow. Part of the weak effect is the painter, not the KAE.

Sheets (two cells are blank because they read as a class on your no-show list):
- `/home/nsa325/work/koopmann/temp/inv/sheet_modes_1.png`
- `/home/nsa325/work/koopmann/temp/inv/sheet_modes_2.png`
- `/home/nsa325/work/koopmann/temp/inv/sheet_colour.png`

## 2. Angles and scaling

| Number | What it means | Across modes | Across two seeds |
|---|---|---|---|
| Size of λ | Small: a "look" mode (texture, lighting). Large: an "object" mode tied to classes. Rank correlation with class tie .76 CLIP, .69 DINOv3 | all modes | reproduces in rank (+.73); the absolute level depends on the recipe |
| Phase of z | Which variant of the mode's concept the image shows. Images with the same phase look more alike than images with opposite phase in 99–100% of CLIP modes, 87% of DINOv3 modes | nearly all modes | reproduces up to a fixed offset or a mirror flip (strength .72, .87 for the most alike pairs) |
| Angle of λ, no rule | Nothing stable. Three outside checks failed or did not replicate | none | weakly (+.21); its sign is at chance (49%) |
| Angle of λ, real-start rule | How much of the mode is passed along (small angle) versus rewritten (large angle) by blocks 7–12 | all 289 live modes | not tested yet; second seed is queued |

**Why the angle is free without a rule (my explanation).** Nothing in the loss fixes how the block-12 coordinates of a mode's plane are rotated against its block-6 coordinates, so each run picks its own. This is the same freedom the DMD and latent-dynamics literature describes for mode phases.

**What the real-start rule did** (`im_realstart_2x4k_wd`; at block 6 a mode has only its first coordinate, so the second can only come from the eigenvalue's turn):
- K now works through the turn: with only the turned-in part the KAE keeps 21.0% accuracy, with only the kept part 2.2%. In the control it is the reverse (0.6% and 25.8%).
- The angle tracks how little of a mode's block-6 write survives to block 12: rank correlation −.61 inside the KAE, and **−.47 in the real model's own states**, with no K and no decoder involved. Three runs without the rule give +.07, +.01 and −.07.
- The angle knob acts: setting the snow mode's angle to 0 fixes the snow failure as well as removing the mode (48.1% against 48.8%). In the control, setting the angle to 0 does nothing. Caveat: that mode's angle is 80°, so setting it to 0 is close to removing it.

**What the rule costs:** 1.3 points (50.9 against 52.2); 289 live modes against 479; modes are less single-meaning (median score 0.6% against 3.0%); at block 6 a mode keeps only a sign, so the phase code book is gone there. It did not make the angle track when a concept becomes readable (−.12).

**The knob demo holds in three runs.** On 129 snowy vehicle photos the KAE fails where CLIP does not. Removing one mode fixes most of it, and normal photos are unchanged:

| Run | KAE on snowy photos | One mode removed | Real CLIP |
|---|---|---|---|
| `im_iso_2x4k_wd` | 30.2% | 49.6% (53.5% with two) | 53.5% |
| `im_realstart_2x4k_wd` | 27.9% | 48.8% | 53.5% |
| `im_iso_2x4k` (no weight decay) | 27.1% | 46.5% | 53.5% |

**Steering.** Pushed to 4 times its usual large value, 67% of modes shift CLIP's class probabilities by more than the steerability threshold of Joseph et al., and different directions of one mode steer to about 4 different classes. That supports the "phase picks the variant" reading. The protocol differs from theirs (modes at block 6 through K and the decoder, class prompts as the vocabulary), so the 67% is not comparable with their 11%.

## 3. Performance

| Run | Epochs | KAE accuracy / agreement | Real model | Linear map |
|---|---|---|---|---|
| `im_dino_w21_wd` | 12 | 61.8 / 66.5 | 83.9 | 27.2 |
| `im_dino_w21_wd_e30` | 30 | 62.9 / 67.9 | 83.9 | 27.2 |

- Chart of both runs by epoch: `/home/nsa325/work/koopmann/temp/chart_long_run.png`
- The run finished at 09:15 EST (5 hours). 2.5 times the epochs bought 1.1 points of accuracy and 1.4 of agreement.
- Train accuracy is 72.3% against 62.9% on val, a gap of 9.4 points; the 12-epoch run ended with 7.5. The extra epochs went mostly into fitting the training images, so the KAE is not short of training time.
- Earlier CLIP pilots point the same way: an MLP in place of K gained about 1.6 points and doubling the width 1.4. No single change so far moves the gap to the real model (about 21 points for DINOv3, about 10 for CLIP).
- My best guess for the ceiling is the input: the KAE sees only a summary of block 6 (CLS, patch mean, patch std). Published probes read middle layers much better with attentive pooling than from the CLS or the patch mean. This is untested here; see decision 1.

## Something you should know about weight decay

Every run with weight decay 0.1 ends with 43–72% of its 1,024 modes dead (firing on fewer than 200 of the 50,000 validation images). No run without weight decay loses any.

| Run | Dead modes | Accuracy | Nameable modes |
|---|---|---|---|
| `im_iso_2x4k` (no weight decay) | 0 | 52.1 | 61% of 1,022, about 620 |
| `im_iso_2x4k_wd` | 545 | 52.2 | 69% of 479, about 330 |
| `im_iso_2x8k_wd` | 445 | 53.6 | not measured |
| `im_dino_w21_wd` | 567 | 61.8 | not measured |

Weight decay does not make the surviving modes cleaner in total; it removes modes. Accuracy is equal at width 4096 (it gave +1.2 at width 8192). I found no paper that ties weight decay to dead modes; this is our own observation.

## Decisions for you

1. **Measure the input ceiling (my first recommendation; queued on 2026-11-10 at 09:40 EST after you said to do what I think is best).** Plain MLPs from the block-6 summary with no KAE: one to the block-12 state with the KAE's own prediction losses, one to the class. If they also stop near the KAE's numbers, the summary is the bottleneck and a richer block-6 summary is the fix. Script `/home/nsa325/work/koopmann/temp/ceiling_probe.py`; wandb runs `probe_clip_b6` then `probe_dino_b6`.
2. **Dead modes.** Keep weight decay as is, or try it on the hidden layers only, or add the AuxK revival loss from Gao et al. One 63-minute CLIP run each.
3. **The real-start rule.** Keep it (the angle gets a meaning, at 1.3 points), or try the overlap loss alone without real-start, which should pin the angle while keeping the phase code book. One 63-minute run.
4. **Inversion.** Either accept it as a scene-level picture, or try removing a group of modes at once and the full 20,000 steps on the GPU.
5. **Two queued repeats you did not ask for.** `im_realstart_2x4k_wd_s1` and `im_iso_2x4k_wd_s1` (seed 1 of two finished runs, 63 minutes each). The first started at 09:15 EST when the long run ended; the second runs after the two probes. They test whether the rule makes the angle reproducible. To cancel, delete their lines in `/home/nsa325/work/koopmann/temp/night/queue.txt`.

Not started, lower priority: BatchTopK, a block-sparse dictionary with a linear decoder, per-image eigenvalues, a view-consistency loss, spurious-correlation benchmarks (Waterbirds, Spurious ImageNet).

## Honest negatives

- The angle of λ is not reproducible across seeds without a rule.
- A tie between the angle and "how early the concept is readable" showed in two CLIP runs with weight decay and did not replicate in the other five runs. I had written it up as a lead before checking; it is retracted.
- One-mode edits are not visible in the inversion, apart from the fire mode.
- The real-start result on the real model's states is one seed, and the survival it measures is small in absolute terms (median .06).
- The 2× speed figure for the CLIP image tower is my estimate from operation counts, not a measurement.

## One question

The 129 snowy photos look like the "vehicles on snow" setting of Santurkar et al. Is that the benchmark you want named in the pitch?

## Files

- Full ledger of every number: `/home/nsa325/work/koopmann/temp/night_results.md`
- 103 research leads with links: `/home/nsa325/work/koopmann/temp/research_notes_night.md`
- Notebook of DINOv3 modes: `/home/nsa325/work/koopmann/second_analysis/dino_modes_im_dino_w21_wd.ipynb`

## Sources

Leads that shaped the reading above:
- Seeds share about 30% of SAE features, close to our 31% of mode pairs: [Paulo & Belrose](https://arxiv.org/abs/2501.16615v2)
- A small weight penalty raises the cross-seed share (tested by the queued pair): [arXiv 2603.04198](https://arxiv.org/abs/2603.04198)
- Dead features and the AuxK revival loss: [Gao et al.](https://arxiv.org/pdf/2406.04093), [outliers and feature death](https://arxiv.org/pdf/2605.31518)
- BatchTopK: [Bussmann et al.](https://arxiv.org/html/2412.06410v1)
- The phase of a DMD or Koopman mode is a free convention: [arXiv 1312.0041](https://arxiv.org/pdf/1312.0041), [arXiv 2001.05043](https://arxiv.org/pdf/2001.05043)
- Koopman eigenfunctions are closed under products and powers: [arXiv 1206.3164](https://arxiv.org/pdf/1206.3164), [arXiv 1810.08733](https://arxiv.org/pdf/1810.08733)
- Rotation as the leftover freedom of latent dynamics models: [arXiv 2603.22886](https://arxiv.org/pdf/2603.22886), [arXiv 2209.00905](https://arxiv.org/pdf/2209.00905)
- Feature persistence across layers, the outside meaning for the real-start angle: [crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html), [Feature Flow](https://arxiv.org/abs/2502.03032)
- Kernel-alignment losses, relatives of the overlap loss: [arXiv 1807.07868](https://arxiv.org/pdf/1807.07868)
- Steering CLIP with SAE features: [Joseph et al.](https://arxiv.org/html/2504.08729v1)
- Single-meaning score for vision features: [Pach et al.](https://arxiv.org/abs/2504.02821)
- Block-sparse featurizers, the closest relative on the SAE side: [arXiv 2606.25234](https://arxiv.org/abs/2606.25234)
- Middle layers read far better with attentive pooling: [arXiv 2506.10178](https://arxiv.org/pdf/2506.10178)
- "Your Transformer is Secretly Linear", a contrast with our linear-map baseline: [arXiv 2405.12250](https://arxiv.org/abs/2405.12250v1)
- No published surrogate for the last blocks of a pretrained ViT; nearest are token-pruning methods: [arXiv 2409.14607](https://arxiv.org/abs/2409.14607v2)
