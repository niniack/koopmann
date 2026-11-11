# Overnight, 2026-11-09: what limits validation

## Headline
- **Val is limited by data.**
  - At the same number of steps, 25% / 50% / 100% of the images give a best val zero-shot of 45.5 / 49.0 / 52.1. That is about +3 points per doubling of distinct images.
  - With fewer images, val peaks early and then falls while train keeps rising.
  - A larger model still helps (1×2048 → 2×4096: +5 points), so the KAE is not "too big" in a way that hurts.
  - CC3M, mixed in the way we tried it, did not help. That run is confounded, though (see below), so other data is not ruled out.
- **KL 0.1 and the 2×2 grid each add about half a point, and it is probably real.**
  - Both reach 52.6 against 52.1.
  - A seed-1 copy of the control finished at 52.11 / 62.16, against 52.11 / 62.17 for seed 0. Per epoch the two seeds stay within 0.3 points, and within 0.2 over the last four.
  - So +0.5 is several times the seed difference. That is only one pair of seeds, though.
  - By the same measure, isometric 0.1 vs 0 (52.1 vs 51.5) is also likely real.
- **KL 0.1 fails your rule.** It worsens reconstruction by 3% and state by 1%. Recommendation: stay at 0.03.
- **KL 0.1 did not close the disagreements.** Agreement was flat in every bin of CLIP confidence.
  - Where CLIP is unsure, its top classes are near ties, and small errors flip the answer at little cost.
  - The accuracy actually lost is mostly on images where CLIP is fairly sure (top probability .5-.9). There the KAE picks a close neighbour of the right class: another dog breed, often CLIP's own runner-up.
- **The next step is more ImageNet-like images.** The target is CLIP's own computation, so labels are not needed.
  - New images are worth far more than extra crops of the same ones: about +3 against +0.8 per doubling (the crop figure is from an older setup, 1×2048 with CLS + patch mean).
  - So ImageNet-21k (or other photos that look like ImageNet) beats storing more crops, which would give roughly +1.

## Where the val gap lives
On 50k ImageNet val, judged with CLIP's zero-shot head (bzkc5flf, epoch 5; seen train crops for comparison):

| | val | seen train crops |
|---|---|---|
| CLIP itself | 63.3 | 63.6 |
| KAE reconstructs block 12 | 60.4 | 60.7 |
| KAE predicts block 12 from block 6 | 51.1 | 57.4 |
| ridge map between latents (no K training) | 49.4 | 55.1 |

- Reconstruction generalizes. The 6-point memorization is all in the prediction step (block-6 encoding + K + decoding).
- Disagreements with CLIP are near misses: CLIP's answer is in the KAE's top 5 for 85% of them. They also sit where CLIP is unsure: its top probability is .43 on disagreements, .80 on agreements.

## Runs before tonight (ImageNet val; KL 0.03, embedding 1, 3 views, Jordan TopK 256, shared modes, CLS + patch mean + std)

| run | encoder/decoder | isometric | extra data | val zero-shot | val agreement | train zero-shot | val KL / train KL |
|---|---|---|---|---|---|---|---|
| kxzt19qo | 1×2048 | 0 | none | 46.5 | 55.7 | 50.6 | .98 / .78 |
| y2vhharw | 2×4096 | 0 | none | 51.5 | 61.6 | 61.3 | .83 / .31 |
| ez6cx5a3 (tonight's control) | 2×4096 | 0.1 | none | 52.1 | 62.2 | 61.6 | .81 / .30 |
| bzkc5flf (best epoch 11 of 20) | 2×4096 | 0.1 | CC3M | 51.6 | 61.6 | 59.4 | .81 / .41 |

Older, same steps, different setup (CLS + patch mean, 1×2048, KL 1): 3 views gave a best val of 49.5, 1 view 48.2, and the 1-view run kept falling after epoch 18 (46.1 at the end).

## Tonight's runs
All on ImageNet, 2×4096, isometric 0.1, KL 0.03 unless named, 3 views, Jordan TopK 256, shared modes; same number of steps; final epoch unless noted.

| run | change | best val zero-shot (epoch) | val zero-shot | val agreement | train zero-shot | train agreement | val KL | train KL |
|---|---|---|---|---|---|---|---|---|
| ez6cx5a3 | control | 52.1 (8) | 52.1 | 62.2 | 61.6 | 77.3 | .807 | .304 |
| wkjot063 | KL 0.1 | 52.6 (7) | 52.6 | 62.4 | 63.3 | 82.0 | .784 | .219 |
| qhodopmm | 2×2 grid input | 52.6 (8) | 52.6 | 62.9 | 62.1 | 78.4 | .770 | .286 |
| 4j3cjq9z | 25% of images, 32 epochs | 45.5 (6) | 44.1 | 52.9 | 63.5 | 88.5 | 1.18 | .099 |
| 3dhdyep7 | 50% of images, 16 epochs | 49.0 (10) | 48.5 | 57.8 | 62.8 | 83.1 | .966 | .184 |
| 32mst724 | control again, seed 1 (to measure seed noise) | 52.1 (6) | 52.1 | 62.2 | 61.5 | 76.9 | .803 | .304 |
| 4ujgz6hv | weight decay 0.1 (2D weights only) | 52.2 (8) | 52.2 | 62.5 | 57.9 | 70.2 | .746 | — |
| zvtaer5s | weight decay 0.1 + 1.16M ImageNet-21k images, 6 epochs | 51.8 (6) | 51.8 | 62.2 | 55.9 | 67.4 | .748 | .556 |
| | MLP in place of K (ceiling) | queued | | | | | | |
| | 2×8192 | queued | | | | | | |

Seed noise, from one pair (seed 0 vs 1): 0.0 on final val zero-shot and agreement, ≤0.2 per epoch over the last four, .001 on reconstruction and state FVU.

### 2×2 grid input (qhodopmm)
Input: CLS, the means of the four quarters of the 7×7 patch grid, and the patch std. That is 4,608 per block, against 2,304 for CLS + mean + std.
- +0.5 zero-shot, +0.7 agreement, val KL .770 vs .807. Train-val gap about the same (agreement 78.4 vs 62.9).
- It gets there sooner: 52.5 by epoch 5. Both have plateaued: over the last three epochs the grid gained +0.1 and the control +0.4.
- Cost: 1h30 vs 1h03 per run, and a 71 GB cache. Its reconstruction and state FVU are not comparable with the control's, since the input differs.
- The ridge probe predicted a little more (+1.6 on a weak baseline). The small gain fits the data limit: more input detail is also more to memorize.

### KL 0.1 (wkjot063) vs control 0.03 (ez6cx5a3), 8 epochs each, final epoch
| | control | KL 0.1 |
|---|---|---|
| val zero-shot | 52.1 | 52.6 |
| val agreement | 62.2 | 62.4 |
| val KL | .807 | .784 |
| val state FVU | .365 | .369 |
| val reconstruction FVU | .206 | .213 |
| val latent error per image | .0083 | .0080 |
| train agreement | 77.3 | 82.0 |

Saved checkpoints (control epoch 8, KL 0.1 epoch 7: both 62.2 agreement), binned by CLIP's top probability on each val image:

| CLIP top probability | ≤.3 | .3-.5 | .5-.7 | .7-.9 | >.9 |
|---|---|---|---|---|---|
| images | 7,307 | 10,638 | 9,919 | 10,435 | 11,701 |
| CLIP right | 24.3 | 42.6 | 61.4 | 78.7 | 94.5 |
| KAE right, control | 16.8 | 32.0 | 46.2 | 64.2 | 86.7 |
| KAE right, KL 0.1 | 17.8 | 32.7 | 46.4 | 64.8 | 86.9 |
| agreement, control | 22.7 | 43.1 | 60.8 | 77.8 | 91.4 |
| agreement, KL 0.1 | 23.1 | 42.7 | 60.3 | 78.2 | 91.6 |

- KL 0.1 buys +0.5 zero-shot (the seed pair differs by 0.0) and nothing on agreement, in any bin. It costs 3% reconstruction and 1% state FVU, and mostly raises memorization (train agreement +4.7).
- Agreement in each bin is about CLIP's own accuracy in that bin. Where CLIP is unsure, its top classes are near ties, and any small error in the prediction flips the argmax. Those flips cost little accuracy, since CLIP is mostly wrong there too.
- What the lost images look like (control): CLIP right and KAE wrong on 8,241 val images, the reverse on 2,629. The KAE's wrong answer is a near neighbour of the right class:
  - median text cosine to the true class is .76, against .79 for CLIP's own mistakes and .65 for a random class;
  - 93% of lost dog images go to another dog breed (CLIP's own dog mistakes: 96%);
  - 32% of the time it is CLIP's runner-up.

  So the KAE makes CLIP-like, fine-grained mistakes. Its prediction is in the right place but blurred, which points at detail the input or the map loses.
- The lost accuracy (63.3 → 52.1) sits mostly where CLIP is fairly sure: 1.1 points come from ≤.3, 2.3 from .3-.5, 3.0 from .5-.7, 3.0 from .7-.9, 1.8 from >.9. Those are the images to target, not the uncertain ones.

### Data size: 25% of the images (4j3cjq9z), same number of steps as the control (32 epochs)
| epoch (control-equivalent) | 4 (1) | 6 (1.5) | 8 (2) | 16 (4) | 32 (8) |
|---|---|---|---|---|---|
| val zero-shot | 45.3 | **45.5** | 45.5 | 44.9 | 44.1 |
| train zero-shot | 58.1 | 60.0 | 61.2 | 62.1 | 63.5 |
| train agreement | 72.1 | 75.9 | 78.0 | 82.7 | 88.5 |
| val KL | 1.08 | 1.08 | 1.08 | 1.12 | 1.18 |

The control at the same step counts: 47.6 (1), 50.0 (2), 51.5 (4), 52.1 (8).

- A quarter of the images costs 6.6 points at the best checkpoint (45.5 vs 52.1). Val peaks after each crop has been seen about 6 times, then falls while train keeps climbing.
- Even at the step count of one control epoch, it is 2.3 points behind. Repeats start hurting almost at once.
- Reconstruction also starts to overfit here (val .224, train .195, against .206 / .196 for the control).
- So the KAE is data-limited on ImageNet-like images.
- CC3M (bzkc5flf) is the one test of other data, and it is confounded.
  - It trained 20 epochs, with the LR decaying over 20, so each ImageNet crop was seen about 20 times instead of 8.
  - It did cut memorization (train-val gap 7.8 vs 9.5) while val stayed flat (51.6).
  - That fits "web images transfer poorly", but it also fits "the extra passes over ImageNet ate the gain".

### Data curve (best val zero-shot, same number of steps)
| share of the 1.28M images | 25% | 50% | 100% |
|---|---|---|---|
| best val zero-shot (epoch) | 45.5 (6 of 32) | 49.0 (10 of 16) | 52.1 (8 of 8) |
| gain from doubling | | +3.4 | +3.1 |

- About +3 points per doubling of distinct images, nearly constant over this range.
- Do not extrapolate far. The KAE's own ceiling is about 60.4, which is what it gets by reconstructing block 12 alone, and gains will bend before that.
- One more doubling (~2.5M ImageNet-like images) would plausibly give about +3. ImageNet-21k (about 3 doublings) might give +5 to +8, capped by the reconstruction ceiling.
- Extra crops are worth much less than extra images. In the older setup, going from 1 to 3 crops (1.6 doublings) gave +1.3, about +0.8 per doubling against +3 for new images. So going from 3 to 8 crops would give roughly +1.

### Weight decay 0.1 (4ujgz6hv)
Only the 2D weights decay (encoder and decoder, under Muon); the eigenvalues and biases do not.

| final epoch | control (2 seeds) | weight decay 0.1 |
|---|---|---|
| val zero-shot | 52.1 / 52.1 | 52.2 |
| val agreement | 62.2 / 62.2 | 62.5 |
| val KL | .807 / .803 | .746 |
| val reconstruction FVU | .206 / .207 | .190 |
| val state FVU | .365 / .364 | .366 |
| val latent error per image | .0083 / .0079 | .0153 |
| train agreement | 77.3 / 76.9 | 70.2 |

- Reconstruction −8% and val KL −7%, far beyond the seed spread (.001 / .004). It halves memorization: the train-val agreement gap is 7.7 points against 15.
- It is slower early (51.7 at epoch 7, against 52.1) and still improving at the last epoch, while the control had flattened. More epochs would likely help it.
- Cost: latent prediction error per image is about 2× the control's, though still small. State FVU ends level.

### ImageNet-21k images (the user's call: added with weight decay)
- Downloaded the first 200 of 2,048 pre-shuffled shards of `timm/imagenet-w21-wds` (114 GB).
- Dropped ImageNet-1k synsets (10%), as those are 1k's own photos, leaving 1,158,002 images. One center crop each, 11.4 min of CLIP.
- Val check, by CLIP-embedding cosine to the nearest of the 50k val images: 321 are above .99 (copies of val images), 336 above .98, 2,336 above .95 (all dropped), and 75,246 above .9 (kept: similar, not copies).
- Run zvtaer5s: control + weight decay 0.1 + 1,155,666 extra images, 6 epochs (about the control's steps). Compare it with weight decay alone.

| final epoch, same steps | weight decay alone | + 21k images |
|---|---|---|
| val zero-shot | 52.2 | 51.8 |
| val agreement | 62.5 | 62.2 |
| val KL | .746 | .748 |
| val reconstruction FVU | .190 | .194 |
| val state FVU | .366 | .368 |
| train agreement (ImageNet crops) | 70.2 | 67.4 |

- **At equal steps it is level, which says little, because both runs are undertrained.**
  - With weight decay, val was still rising steeply at the last epoch, so the binding limit is steps, not data. Extra images just spread the same steps over more data.
  - The control without weight decay had flattened by epoch 6, so it was the data-limited case.
- **Signs that the images help:**
  - At an equal number of ImageNet passes (6), the 21k run is ahead: 51.8 vs 51.1.
  - It gains twice as fast in its last epoch (+1.1 vs +0.5 zero-shot; KL −.044 vs −.026).
  - Its train-val agreement gap is smaller (5.2 vs 7.7).
- **The clean test:** train both to convergence at the same steps, e.g. weight decay alone for 16 epochs against weight decay + 21k for 12 epochs.
- Caveat: the new images had 1 crop each against 3 for ImageNet.

## Recommendations
(weight decay, MLP-K and 2×8192 still to come)

1. More training pairs that look like ImageNet. This is the biggest lever measured: about +3 points per doubling of distinct images (25 → 50 → 100%: 45.5 → 49.0 → 52.1).
   - The target is CLIP's own computation, so no labels are needed. Any image, augmented however, gives a correct (block 6, block 12) pair.
   - Options, cheapest first:
     - **More stored crops per image** (e.g. 8 instead of 3), kept in half precision like the grid cache. Cheapest, but likely only about +1 (see the data curve). Extraction runs at about 1,650 crops a second with 16 workers on a 64-core machine, so about 1h45 for 8 views (faster with more workers).
       - Expect less than the 6.6 points: an extra crop of an image already seen adds less than a new image. In the older setup, 3 crops beat 1 by 1.3 points at the same steps.
     - **Heavier augmentation at extraction** (colour jitter, RandAugment, CutMix of two images): more variety, still ImageNet-like.
     - **Fresh crops every epoch**, with CLIP run online: about 13 minutes of CLIP per pass over 1.28M images, at today's loader speed.
     - **ImageNet-21k**: about 10× the images, and close to the val distribution, but a large download. Before trusting any val gain, check it for overlap with the 1k val images.
   - CC3M, mixed in as we did, did not help. That one run is confounded, so other datasets are not ruled out, but they are not the first thing to try.
2. Stop at the best val epoch whenever data is the limit. With repeats, val turns down early (epoch 6 of 32 at 25%).
3. Keep KL at 0.03. At 0.1 it gains +0.5 zero-shot but worsens reconstruction (3%) and state (1%), so it fails your rule. The seed pair moved reconstruction by only .001, so the 3% is real damage.
4. The 2×2 grid is a small, likely real gain (+0.5 zero-shot, +0.7 agreement, val KL −5%). It costs 1.4× the time and a 71 GB cache. Worth adopting if the cost is fine, but it is worth far less than doubling the images.

## Incidents
- The first grid run (l8ochbwg, left on wandb, empty) extracted its 71 GB cache, then died at 05:45.
  - Cause: `train_states.sub_(mean)` on a half tensor with float statistics makes a float copy of the whole tensor first (measured: 4.1 GB tensor, 20.5 GB peak).
  - Fix in `train.py`: the half branch now standardizes in chunks. The float path is unchanged.
  - An end-to-end smoke test on a synthetic half cache passed. The rerun is `im_grid_2x4k_r2`, after f25.
- The queue runner's "exited with code" lines always say 0: `$?` is read after `$(date)`. Ignore them; wandb is the record.

## Follow-ups not run
- Temperature-softened KL: the KL is at CLIP's own temperature (logit scale 100), so its gradient goes almost only to the top class. At temperature 2-4, it would also teach the order of near-miss classes, which is where disagreements sit.
- Fresh crops every epoch instead of 3 fixed views: CLIP blocks 1-12 run online, roughly 20 minutes an epoch.
