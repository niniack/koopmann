# History digest 3: 10 and 11 November 2026

What was tried and learned from 10 Nov 00:00 EST to 11 Nov 10:39 EST (the moment the review was asked for), in time order.

## How to read this file

- `SP` stands for the scratchpad: `@SP@`. Sources are written `SP/<path>`.
- `L<n>` means `full_11-10.txt line n` (the label in the entry header of the history file, not the file's own line count). `11-11 L<n>` means `full_11-11.txt line n`.
- `NR` = `SP/night_results.md`, `MR` = `SP/morning_report.md`, `DF` = `SP/dino_findings.md`.
- All times are EST. Scratchpad file times are EST + 9 hours.
- KAE = the Koopman autoencoder that stands in for blocks 7 to 12 of the image model: an encoder turns the block-6 state into modes, each mode is multiplied by its eigenvalue λ, a decoder turns the result into a block-12 state.
- Mode = one pair of numbers (a complex number z). λ has a size and an angle. z has a size and a phase.
- "Accuracy / agreement" for CLIP runs = ImageNet validation accuracy of CLIP's zero-shot head on the KAE's decoded block-12 state (`val/zero_shot_accuracy`) / share of images where that answer equals real CLIP's answer (`val/clip_agreement`). Real CLIP ViT-B/32 scores 63.3 (63.34 on 50,000 images, 62.99 on 10,000, 63.2 on 5,000).
- For DINOv3 ViT-B/16 the logs say "probe accuracy" (a linear ImageNet probe on the block-12 state; which probe was not re-checked for this digest). Real DINOv3 scores 83.94.
- "wd" = weight decay 0.1.
- "Kept part" of a mode = Re(λ)·z₆ (what stays in the block-6 direction). "Built part" = Im(λ)·z₆ (what the angle of λ adds in the other direction).
- "Real-start rule" = at block 6 the second number of every mode is forced to zero, so a mode starts as one signed strength; only block 12 uses both numbers. Flags `--real_start=True --lambda_overlap=0.1`.
- "Older state" = 2,304 numbers per image (CLS token + mean of patch tokens + standard deviation of patch tokens). Runs named `im_*` use it with stored crops; 8 epochs = 30,000 steps.
- Common settings unless an entry says otherwise: encoder and decoder of 2 layers × 4,096 units, 1 real mode + 1,024 two-number modes (latent 2,049), TopK keeps 256 coordinates (128 modes) per image, modes shared between block 6 and block 12.

---

# 10 November

## 11-10 00:00 EST DINOv3 KAE with extra images (`im_dino_w21_wd`) and where its gap opens
- Question: does the recipe work on DINOv3 ViT-B/16, and at which step are the points lost?
- Setup: run `im_dino_w21_wd` (wandb urzqs9i6), older state from DINOv3 blocks 6 and 12, wd 0.1, 12 epochs on ImageNet (3 stored views) plus ImageNet-21k images ("w21"). Trained 9 Nov 15:22 to 18:10 (`SP/night/runner.log`); studied from 10 Nov 00:00 (L13729). The gap was split by feeding true block-12 values into parts of the KAE.
- Result: accuracy 61.78, agreement 66.49 (`SP/night/im_dino_w21_wd.log`: .61778 / .66488; `SP/gate_built.log`: KAE .6178, DINOv3 .8394, same answer .6649). Ladder (DF): true block 12 through encoder and decoder 81.0; restricted to the modes picked at block 6 79.5; λ's size with the true angle 74.2; λ's angle with the true size 73.1; λ as trained 61.8. 457 of 1,025 modes are ever kept (DF). A plain linear map from block 6 to block 12 scores 27.16 (NR line 180; MR says 27.2).
- Conclusion at the time: almost all of the 22-point gap opens at the eigenvalue step, not in the encoder or decoder. Size and angle each account for about half of it.
- Status now: stands. For accuracy it is superseded by the 30-epoch run (62.90, see 09:15). DINOv3 was not trained again after this day.
- Goal: speed, groundwork
- User said: nothing tied to this run.
- Files worth reusing: @SP@/dino_findings.md, @SP@/gate_built.log, @SP@/night/im_dino_w21_wd.log

## 11-10 00:48 EST Rule: no pictures of bugs or arachnids, and the filter built for it
- Question: none; a standing rule from the user.
- Setup: a 50,000-long mask over ImageNet validation (`hidden_mask.pt`) was built at 02:26 (L15378). Every photo sheet made afterwards drops the masked images. Phase-wheel sheets with snakes, lizards or salamanders were also held back (modes 750, 810, 569, 247, 644, 37, 535; `SP/phase_wheel.log`, `SP/phase_wheel_chosen.log`).
- Result: two cells of the inversion sheets are blank because of the mask (MR). The constant-mode gallery in the old report "does include one newt and one green snake" (L23093).
- Conclusion at the time: filter by class and by top-5 answers before showing any photo.
- Status now: in force. The newt and snake in `constant.png` show the mask does not cover reptiles and amphibians; a reviewer should look at every photo.
- Goal: groundwork
- User said: "btw, never ever show me pictures of bugs or arachnids. ever." (L13743)
- Files worth reusing: @SP@/hidden_mask.pt

## 11-10 01:00 EST What the always-on modes of the DINOv3 KAE do
- Question (user, L13743): "what are the modes that every image has doing? what is the common thing being captured across all modes. and, relative to other modes, are they growing fast? are they rotated a lot?"
- Setup: `im_dino_w21_wd`, 50,000 validation images. Modes that are kept on every image were tested three ways: give the mode the same value for all images, set its angle to 0 at the same size, remove it. Source `SP/dino_always_on.log` (first appears L13896) and DF.
- Result:
  - 2 growth modes: mode 331 (size of λ 18.94, angle 10.5°) and mode 77 (14.43, 8.1°). They hold 78% of the latent energy at block 12. Removing either costs 7 to 10 points. For mode 331, using the average size costs 1.3 points; using the average direction costs 8.8.
  - 10 offset modes (861, 338, 144, 560, 117, 988, 871, 746, 652, 533): nearly the same value for every image. Angles 14° to 174°, median 82°. They hold 63% of the latent energy at block 6. Size of λ 0.6 to 3.5, against a median of 7.6 for feature modes. Mode 861 (λ 1.18 at −173.5°): same value for all images costs 0.5 points; angle set to 0 costs 9.7; removed costs 5.1. Mode 338: angle set to 0 costs 7.6.
  - 4 inert modes (68, 653, 877, 465): λ about 0.96, angle 0° to 4°, push size 0.000.
  - 441 feature modes (on for some images only): 305 have size of λ between 5 and 10 and angle 5° to 45°; 136 have size under 5.
  - The average block-6 state and the average block-12 state point in opposite directions (cosine −1.00). Block-12 states are 10.7 times wider.
  - The true gain of a mode tracks the size of λ (.97). λ·z₆ explains 58% of z₁₂ for the median feature mode.
- Conclusion at the time: the always-on modes carry the average state and its sign flip from block 6 to block 12 (the offset modes, with large angles that are needed), plus a strong overall growth (the two growth modes). The large angles belong to these bookkeeping modes, not to features.
- Status now: stands for this one DINOv3 KAE. Not repeated on the 30-epoch run or on any CLIP KAE in the same form.
- Goal: meaning
- User said: see the question above.
- Files worth reusing: @SP@/dino_always_on.log, @SP@/dino_findings.md, @SP@/dino_mode_table.csv

## 11-10 01:10 EST Snow on DINOv3: the KAE fails where DINOv3 does not
- Question: does the snowy-vehicle test (our internal test for fixing errors by editing one mode) carry over to DINOv3?
- Setup: `im_dino_w21_wd`; snowy and normal validation photos of vehicle classes (the CLIP version of the test uses 129 snowy and 350 normal photos of 7 classes; the DINOv3 count was not re-checked). Snow modes were found by how well a mode separates snowy from normal photos, then removed. Source DF; reported by 01:26 (L14248).
- Result: real DINOv3 86.0% on snowy, 82.9% on normal. KAE 34.1% on snowy, 67.7% on normal. Snow modes 667, 644, 37, 247. Removing the top 4: 57.4% on snowy (31 fixed, 1 broken). On held-out halves: 56% and 58%. Applied to 25,815 validation images: 312 fixed, 574 broken. Giving only the snow modes their true block-12 values: 38.8%. Giving every other mode its true value: 69.8%.
- Conclusion at the time: "A patch, not a repair." The snow error is made by the KAE, not by DINOv3. Removing the snow modes helps the snowy photos but hurts more photos elsewhere than it helps.
- Status now: stands. The same pattern (fix on the target set, net loss on all images) was found on CLIP at 03:15.
- Goal: adaptation
- User said (later, L19415): ""vehicles on snow" is just our internal test to see whether we can fix errors with this. It can't be the only example we show in the paper."
- Files worth reusing: @SP@/dino_findings.md

## 11-10 01:02 EST DINOv3's own mistakes: can be flagged, not fixed
- Question (user, L13743): "what are dino's failures, if not snow? can we fix them?"
- Setup: `im_dino_w21_wd`, 50,000 validation images, split into images real DINOv3 gets wrong and right. Modes that are more active on the wrong images were looked for; each was removed to see how many answers flip. Source `SP/dino_failures.log` (L13968), DF.
- Result: DINOv3 is wrong on 8,032 validation images; the label is in its top 5 for 80% of them. The KAE is right on 12.9% of those images, against 71.1% on the rest. 28 modes flag the mistakes (split-half agreement 0.67). Editing does not fix them: mode 347 removed gives 282 fixed, 426 broken; on held-out images accuracy goes from 79.40 to 79.33. Summed over modes, removal fixes 51,909 answers and breaks 97,332.
- Conclusion at the time: the KAE inherits DINOv3's mistakes and adds its own. Some modes tell you a mistake is likely. No single-mode edit repairs them.
- Status now: stands. Never tried with tuned eigenvalues on ImageNet itself.
- Goal: adaptation
- User said: see the question above.
- Files worth reusing: @SP@/dino_failures.log, @SP@/dino_findings.md

## 11-10 01:31 EST Decision: the night's three jobs
- Question: none; the user set the work for the night.
- Setup: the user proposed inverting the KAE's output to an image in the way of Splice (arXiv 2201.00424) at 00:48, then left at 01:31. The assistant worked alone until about 05:05 and again from 08:52.
- Result: three jobs were taken on: (1) the inversion, (2) tests of what the angle and the size of λ mean, (3) better accuracy through the queued runs. Results are the entries from 01:48 to 09:15 below.
- Conclusion at the time: not applicable.
- Status now: done; see below.
- Goal: meaning, speed
- User said: "okay crazy paper: https://arxiv.org/pdf/2201.00424. Here's the vision. [...] I want to do this with the output of our KAE. However, i want to see if we can make image more black/white, more sepia, or what if we remove a feature and see if it erases that quality. does that make sense?" (L13743). "i'm going to sleep, so your goal is to do the cls inversion + see if we can make the angles/scaling meaningful + definitely improve the kae's performance. we still only have short pilots. no long runs yet, so we don't know how far it goes" (L14311).
- Files worth reusing: @SP@/night_results.md (the running log of the night), @SP@/morning_report.md

## 11-10 01:48 EST Inversion pilot: which part of the state to paint from
- Question: can a small network, trained for one image, paint a picture whose DINOv3 block-12 state matches a target state? Should the target be the CLS token only or the whole state?
- Setup: `inversion.py` (L14248), Splice's skip network, Adam 0.01, 4,000 steps on CPU, targets = real block-12 states of two validation photos (15059 a pile of oranges, 39437 a snowmobile). Two settings: `all` (CLS + patch mean + patch std) and `cls`. Source `SP/inv/pilot_15059_all.log`, `SP/inv/pilot_15059_cls.log`, `SP/inv/pilot_39437_all.log`, `SP/inv/pilot_39437_cls.log` (L14658) and NR.
- Result: share of the state error left: photo 15059 `all` .063, `cls` .160; photo 39437 `all` .108, `cls` .176. `all` keeps colour and scene but loses the snowmobile. `cls` keeps the object but the colour drifts.
- Conclusion at the time: use `all`, with about 3,000 steps on CPU instead of Splice's 20,000.
- Status now: stands; used for the batch at 08:58.
- Goal: meaning
- User said: see 01:31.
- Files worth reusing: @SP@/inv/pilot_15059_all_4000.png, @SP@/inv/pilot_15059_cls_4000.png, @SP@/inv/pilot_39437_all_4000.png, @SP@/inv/pilot_39437_cls_4000.png, @SP@/inversion.py

## 11-10 01:50 EST Phase against size at block 6
- Question: does the information of a mode at block 6 sit in its size or in its phase? If the phase carries content, the angle of λ has no fixed starting point.
- Setup: `im_dino_w21_wd`, 439 feature modes (of 455), 1,500 images per mode. For each mode: set its phase to the average, set its size to the average, or remove it, then read the answer. Source `SP/phase_size_all.log`, `SP/phase_size_all.csv` (L14692).
- Result: mean accuracy drop in points: phase to average 0.151, size to average 0.167, removed 0.471. Answers changed: 1.9% / 1.7% / 4.0%. Phase matters more than size in 240 of 439 modes. The share carried by phase tracks how much the phase varies across images (.78). Growth mode 77: 17.53 / 3.53 / 8.80 points. Growth mode 331: 10.73 / 1.87 / 7.93.
- Conclusion at the time: at block 6 the phase carries as much as the size. So each image starts a mode at a different phase, and the angle of λ is a turn from a different starting point each time. This is what motivated the real-start rule.
- Status now: stands. It is the reason for the real-start rule (04:14).
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/phase_size_all.csv, @SP@/phase_size_all.log

## 11-10 01:58 EST Gray and sepia copies: do colour changes show up as a turn or as a rescaling?
- Question: when a photo is made gray or sepia, do the modes at block 6 change in size or in phase? Could one eigenvalue-like factor per mode stand for "make it gray"?
- Setup: `im_dino_w21_wd`, 2,000 validation images, each with a gray and a sepia copy. A single factor per mode was fitted to move the original's modes to the copy's modes: complex (size and turn), real (size only), turn only. Source `SP/alter_states.log`, `SP/alter_factors.pt` (L14675).
- Result: DINOv3 | KAE | agreement: original .820 | .596 | .657; gray .801 | .491 | .540; sepia .797 | .440 | .487. Gray: 80.2% of the latent change at block 6 is size, 19.8% is turn; fitted angle quartiles 0.6° / 1.4° / 2.7°; the complex factor covers .29 of the way, the real factor .274, turn only .009, ceiling .812. Sepia: 82.1% size, 17.9% turn; .324, .298, .031, ceiling .794. Mode 117 carries 19.1% of the sepia change (factor 0.60 at −15.1°).
- Conclusion at the time: gray and sepia act as a rescaling of modes, hardly as a turn. One factor per mode covers under a third of the change. The KAE is also much less robust to gray and sepia than DINOv3 is (minus 10 to 16 points against minus 2).
- Status now: stands. The fitted factors were used as the "colour knobs" in the inversion batch.
- Goal: meaning, adaptation
- User said: see 01:31 (the black/white and sepia idea).
- Files worth reusing: @SP@/alter_factors.pt, @SP@/alter_states.log

## 11-10 02:06 EST Cascade gate: send the unsure images to the real blocks
- Question: if the images the KAE is least sure about are passed to the real blocks 7 to 12, how much accuracy comes back for how much extra cost? (The assistant's own addition.)
- Setup: `gate.py`, `im_dino_w21_wd`, validation images ranked by the KAE's top probability. Source `SP/gate.pt`, NR.
- Result: share passed to the real blocks 0 / 10 / 20 / 30 / 40 / 50 / 60 / 70 / 80 / 100% gives accuracy 61.8 / 67.2 / 71.8 / 75.6 / 78.7 / 81.0 / 82.4 / 83.3 / 83.8 / 83.9. The top probability separates right from wrong answers with AUROC 0.840. Reaching 80% accuracy needs 45.5% of images passed on.
- Conclusion at the time: a cascade recovers accuracy smoothly, but it needs almost half of the images to get within 4 points of DINOv3.
- Status now: stands. Never repeated on CLIP or on the later KAEs. Not in the old report.
- Goal: speed
- User said: nothing tied to it.
- Files worth reusing: @SP@/gate.pt, @SP@/gate.py

## 11-10 02:28 EST Depth clock: do modes turn steadily through blocks 7 to 11?
- Question: if the KAE's encoder is applied to the states of blocks 7 to 11 (which it never saw), does a mode's phase move steadily from its block-6 value to its block-12 value, like a clock? (The assistant's own addition; it reads intermediate blocks although the user had said "I don't want to do intermediate blocks". It does not train on them.)
- Setup: `im_dino_w21_wd`, 2,000 validation images, states of blocks 6 to 12. Source `SP/depth_clock.log`, `SP/depth_clock.pt` (L15394, L15413).
- Result: reconstruction error by block 6 to 12: .038, .323, .945, 1.615, 2.263, 1.939, .243 (the encoder does not fit the middle blocks). Cosine of the average state with block 6's: 1.00, .99, .96, .87, .64, .07, −1.00. For 329 feature modes that turn more than 5°, median share of the final turn done by block: 0, .07, .31, .49, .66, .89, 1. For 394 modes, median share of the growth done: 0, −.06, −.07, .07, .19, .42, 1. Half of the turn is done by block 8.48 (median), half of the growth by block 11.12. Only 10% of modes never turn backwards by more than 2°. Offset mode 338 turns 0, −4.8, −16.8, −41.7, −67.5, −90.3, −93.9°. Offset modes 861 and 652 pass through near-zero size at block 10. Growth mode 331 grows 1, 1.32, 2.41, 3.70, 4.75, 7.44, 18.92.
- Conclusion at the time: the always-on modes behave like a clock (steady turn, growth late). For feature modes the reading is noisy because the encoder does not fit the middle blocks. Only suggestive.
- Status now: never checked again. Weak evidence because of the large reconstruction error in the middle blocks.
- Goal: meaning
- User said (earlier, still in force): "I don't want to do intermediate blocks"
- Files worth reusing: @SP@/depth_clock.pt, @SP@/depth_clock.log

## 11-10 02:33 EST Kept part and built part on the CLIP KAE with weight decay
- Question: how much of the KAE's answer comes from the part of each mode that stays in its block-6 direction (kept, Re λ) and how much from the part the angle adds (built, Im λ)? Do the eigenvalues match what the real states do?
- Setup: `kept_built.py` (L15188) on `im_iso_2x4k_wd` (wandb 4ujgz6hv; CLIP, older state, wd 0.1, 8 epochs), all 50,000 validation images, CLIP's zero-shot head. The first try ran 20 minutes with no output (it encoded 80,000 text prompts on CPU and the output was held in a pipe); fixed by caching the head in `zero_shot_head.pt`, checked at 62.99% on 10,000 real states. Source `SP/kept_built_im_iso_2x4k_wd.log`.
- Result (accuracy | agreement): real CLIP .6334. KAE as trained .5217 | .6247. Kept part only .2584 | .2883. Built part only .0062 | .0065. No turn, same size .1152 | .1184. Turn only, size 1 .0260 | .0282. No eigenvalue step .0020 | .0014. True block-12 values of the kept modes .6164 | .8138. True kept part with the KAE's built part .5829 | .7313. 479 modes in use, 19 always on. Size of λ at 10/25/50/75/90%: .83 / 1.94 / 3.18 / 3.69 / 4.13. Angle: 1.7 / 3.9 / 9.7 / 16.0 / 23.0°. Gap between the angle of λ and the true mean turn: 0.2 / 0.6 / 1.2°. Correlation of sin²(angle) with the true built share .631; of size of λ with true gain .988. Images sorted into fifths by built share (.037 to .094): KAE right .562, .558, .530, .498, .460; CLIP right .664 down to .596.
- Conclusion at the time: the eigenvalues are faithful summaries of the real average gain and turn of each mode. Most of the answer rides on the kept part, but both are needed (kept only 25.8, both 52.2). Images that need more built part are the ones the KAE gets wrong more often.
- Status now: stands for this run. With the real-start rule the split reverses (built only .2105, kept only .0215; see 04:14).
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/kept_built_im_iso_2x4k_wd.log, @SP@/kept_built.py, @SP@/zero_shot_head.pt

## 11-10 02:35 EST View consistency: is the phase a nuisance of the crop?
- Question: do two crops of the same image give the same modes, sizes and phases? If the phase only reflected the crop, it would differ between views. (The assistant's own addition.)
- Setup: `im_dino_w21_wd`, 6,000 training images × 3 stored views. Source `SP/view_consistency.log`, `SP/view_consistency.pt` (L15559).
- Result: overlap of kept modes (Jaccard) .719 for two views of one image, .270 for two different images. Across feature modes (25/50/75%): kept by both views .626 / .659 / .695; size at block 6 agrees .807 / .900 / .925; phase at block 6 .937 / .963 / .974; size at block 12 .803 / .935 / .951; phase at block 12 .844 / .927 / .946; turn .796 / .822 / .841. Size agrees more than phase in 0% of the 195 modes with a wide phase spread.
- Conclusion at the time: the phase is as stable across crops as the size, so it is content of the image, not noise from the crop.
- Status now: stands (one DINOv3 KAE).
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/view_consistency.pt, @SP@/view_consistency.log

## 11-10 02:41 EST Mode parts: where in the state does a mode's push land?
- Question: does a mode push the CLS token, the patch mean or the patch spread? Does the angle or the size of λ tell which?
- Setup: `im_dino_w21_wd`, 439 feature modes; each mode's push on the decoded state was split over the three parts of the older state. Source `SP/mode_parts.log`, `SP/mode_parts.csv` (L15711).
- Result: share of the variation across images, true block 12: CLS .487, patch mean .433, patch std .081. Block 6: .299 / .489 / .212. KAE's block 12: .427 / .515 / .057. Per mode, median share of the push: CLS .490, patch mean .448, patch std .044. Correlation of a mode's CLS share with its angle .032, with size of λ .083, with usage −.080. The built part has a higher CLS share than the kept part in 66% of modes (median difference .011).
- Conclusion at the time: the eigenvalue does not say where a push lands.
- Status now: stands; negative result. Not relevant to the CLS-only state used from 17:27 on.
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/mode_parts.csv

## 11-10 02:46 EST Built share and other eigenvalue signals as a gate for "the KAE is wrong"
- Question: can numbers read off the eigenvalues (how much built part an image uses, its mean angle, its mean size of λ) tell when the KAE's answer is wrong, better than the KAE's own confidence?
- Setup: `im_dino_w21_wd`, validation images; AUROC for telling right from wrong answers; cascade accuracy on a held-out half. Source `SP/gate_built.log` (L15799).
- Result: AUROC: top probability .840; minus entropy .830; minus built share over all modes .525; over feature modes .473; minus mean angle .484; mean log size of λ .574. Cascade with 10 to 60% of images passed to the real blocks: top probability .6698, .7155, .7547, .7872, .8096, .8244; eigenvalue signals only .6482 to .7733 (AUROC .602).
- Conclusion at the time: the eigenvalue signals are close to chance. The plain confidence is the better gate.
- Status now: stands; negative result.
- Goal: speed, meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/gate_built.log

## 11-10 02:47 EST Angle and size of λ against how much a mode follows the class
- Question: do modes with a larger eigenvalue, or a larger angle, follow the class label more? ("Class tie" = how much of a mode's strength across images is explained by the class.)
- Setup: three KAEs, all feature modes, 50,000 validation images. Sources `SP/angle_vs_class_dino.log`, `SP/angle_vs_class_im_iso_2x4k_wd.log`, `SP/angle_vs_class_im_realstart_2x4k_wd.log` (L15850).
- Result (rank correlations): DINOv3 (439 modes): angle with class tie at block 6 .354, at block 12 .376; size of λ .686 / .711; angle with size .539; class tie by fifths of size .077, .101, .397, .431, .421. CLIP wd (460 modes): angle .336 / .362; size .760 / .783; angle with size .425; fifths .078, .242, .356, .444, .505. CLIP real-start: angle .185, size .818. Later, on the CLIP run without weight decay (`im_iso_2x4k`), size against class tie is +.79 (demo notebook, L20080).
- Conclusion at the time: the size of λ says how much a mode follows the class. The angle's link to class is mostly explained by its link to size.
- Status now: stands. This is the firmest "meaning" result of the period: +.69 to +.82 in every KAE tested. It is the scatter in the old report (`nb/cell15_0.png`).
- Goal: meaning
- User said (11:36, L20085): "in the trend where the larger the scale, the more a mode follows, is that after we do our trick where we only allow block 12 to use the phase?" Answer given: it holds both without the rule (+.79) and with it (+.82) (L20098).
- Files worth reusing: @SP@/report/nb/cell15_0.png, @SP@/angle_vs_class_dino.log, @SP@/angle_vs_class_im_iso_2x4k_wd.log, @SP@/angle_vs_class_im_realstart_2x4k_wd.log

## 11-10 02:54 EST Angle scan: does any property of a mode track its angle?
- Question: among many measured properties of a mode, which follow the angle of λ once size and usage are held fixed?
- Setup: `im_dino_w21_wd`, 439 feature modes, 34 properties per mode (from the earlier tests). Rank correlation with the angle with size and usage held fixed, and with the size with angle and usage held fixed. Source `SP/angle_scan.log`, `SP/angle_scan.csv`, `SP/dino_mode_table.csv` (L15929).
- Result: with the angle, no property is above .106 (top three: true turn alike across images .106; push lands in patch std −.100; push size .091). With the size: up to .651 (two views agree on phase at block 12 .651; class tie gained from block 6 to 12 .642; .616; .534; true gain over size of λ −.587; −.327).
- Conclusion at the time: nothing tracks the angle beyond what the size already explains.
- Status now: stands for KAEs without the real-start rule. With the rule one property does track the angle (persistence, see 04:50).
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/angle_scan.csv, @SP@/dino_mode_table.csv

## 11-10 02:57 EST Mode names: do the kept part and the built part mean different things?
- Question: does a mode's push on the state match the photos it is strongest on? Does the built part (from the angle) say something different from the kept part, in CLIP's words?
- Setup: `im_iso_2x4k_wd` (CLIP), 479 modes. Each mode's push was compared with the average CLIP embedding of its top images, and with 19,404 words through CLIP's text encoder. Sources `SP/names_im_iso_2x4k_wd.log`, `SP/kept_built_words_im_iso_2x4k_wd.log`, `SP/clip_vocab.log` (L15508, L15778, L16334).
- Result: push matches the top images (25/50/75%): .183 / .365 / .463; kept part .159 / .333 / .447; built part −.010 / .097 / .234. The mode itself is the best match for its top images in 68.7% of modes (in the best 5: 79.1%); kept part 63.3%; built part 21.9%. Class focus of the words: mode .068, kept .067, built .066; built is higher in 50% of the 319 modes that turn more than 5°. By angle: 0 to 5° (140 modes) kept matches .265, built .022; 20 to 45° (60 modes) .346 / .304. Angle against "built matches the top images" .496 (.395 with size fixed; mostly mechanical, since a larger angle makes a larger built part). Examples: mode 521 kept "spaghetti mop noodles carrot clifford", built "coral fungal fungi mushroom mushrooms"; mode 312 kept "pottery mugs mug cup vases", built "coffee tea brew mug caffeine". Real-start run: for modes at 45 to 180°, words match kept .095, built .316; best-match rates 69.9 / 44.3 / 61.9 (mode / kept / built).
- Conclusion at the time: modes are nameable from their push. The built part is a real, different direction only for modes with a large angle; a few examples read as "texture kept, object built", but this was not scored.
- Status now: the examples were never scored against a baseline. The user liked the reading as a pitch (see quote) and it is still open.
- Goal: meaning
- User said (10:10, L19415): "So, the "real-start" rule forces block 6 reps to learn a single scaling while the block 12 reps get to rescale it + change the phase, right? And, you are saying that the first scaling shows texture, while the later scaling + rotation shows different objects using that texture? That's a decent interpretability pitch."
- Files worth reusing: @SP@/names_im_iso_2x4k_wd.log, @SP@/kept_built_words_im_iso_2x4k_wd.log

## 11-10 03:10 EST Width 8,192 with weight decay (`im_iso_2x8k_wd`)
- Question: does a wider encoder and decoder close the gap?
- Setup: CLIP, older state, 2 layers × 8,192 units, wd 0.1, 8 epochs, wandb 882lfgmo. Ran 9 Nov 22:29 to 10 Nov 03:10 (`SP/night/runner.log`). The run without weight decay at the same width (`im_iso_2x8k`, wandb l4sdh1k1) was killed after epoch 5 on 9 Nov; the queue runner still reported exit code 0.
- Result: accuracy .53604, agreement .64476; on training crops .60914 / .75396 (`SP/night/im_iso_2x8k_wd.log`). 579 modes in use (NR line 211). Width 4,096 with weight decay: .52166 / .62466 (`SP/night/im_iso_2x4k_wd.log`).
- Conclusion at the time: doubling the width buys 1.4 points at twice the cost.
- Status now: stands. It was the best fully trained KAE on stored crops, with `im_mlpk_2x4k` (53.8 / 64.2). Both were passed by the online-crop runs with a control (58 to 59).
- Goal: speed
- User said: nothing tied to it.
- Files worth reusing: @SP@/night/im_iso_2x8k_wd.log

## 11-10 03:12 EST Push and turn: does a larger angle mean the mode's pattern changes more between block 6 and block 12?
- Question: is the angle of λ a measure of how different a mode's block-12 pattern is from its block-6 pattern?
- Setup: three KAEs; for each mode the cosine between its state pattern at block 6 and at block 12, through the eigenvalue step and in the true states. Sources `SP/push_turn_dino.log` (439 modes), `SP/push_turn_im_iso_2x4k_wd.log` (460 modes), `SP/push_turn_im_realstart_2x4k_wd.log` (L16071).
- Result: DINOv3: cosine at 10/25/50/75/90% .054 / .066 / .086 / .310 / .386; rank correlation with the angle −.436, with size fixed −.128; with the size (angle fixed) −.615; true states median .069; the KAE's block-12 pattern against the true one, median .666. CLIP wd: median .091; −.418, with size fixed −.124; size −.748; true median .069; KAE against true, median .639. Correlation of cos(angle) with the state cosine .330 (DINOv3), .244 (CLIP). Real-start: with size fixed −.607.
- Conclusion at the time: block-6 and block-12 patterns of a mode are nearly unrelated for most modes (cosine under .1), whatever the angle. Without the real-start rule the angle says little about it once size is fixed. With the rule it does (−.607).
- Status now: stands.
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/push_turn_dino.log, @SP@/push_turn_im_iso_2x4k_wd.log, @SP@/push_turn_im_realstart_2x4k_wd.log

## 11-10 03:15 EST Snow test on CLIP: remove the snow mode, or only its angle
- Question: on CLIP, does removing one or two modes fix the snowy-vehicle errors? Is the error in the mode's size or in its angle?
- Setup: `im_iso_2x4k_wd`; 129 snowy and 350 normal validation photos of 7 vehicle classes. Snow modes ranked by how well they separate snowy from normal photos. Edits: remove, set angle to 0, halve the size. Source `SP/cue_conclusion_im_iso_2x4k_wd.log` (L16253). Repeated on `im_realstart_2x4k_wd` and `im_iso_2x4k` (MR), and on real-start seed 1 (`SP/seed_match_realstart_wd.log`, NR).
- Result: real CLIP snowy .535, normal .643. KAE snowy .302, normal .546. The KAE's wrong answers on snowy photos: snowmobile 40, snowplow 40. Snow mode 344 (separates snowy from normal with AUC .997; λ 4.55 at 7.0°), then 413, 671, 522. Top mode removed: snowy .496 (25 fixed, 0 broken; normal .543). Angle to 0: .302 (0 fixed). Half size: .333. Top 2 removed: .535 (30 fixed), which equals real CLIP. Top 4: .519. Held-out halves: .344 to .484 and .262 to .554. On all validation images, top 4 removed: 658 fixed, 801 broken; angle to 0: 273 fixed, 267 broken. Three runs (MR): `im_iso_2x4k_wd` 30.2 to 49.6 (53.5 with two modes); `im_realstart_2x4k_wd` 27.9 to 48.8; `im_iso_2x4k` 27.1 to 46.5. With the real-start rule (snow mode 493 at 79.8°, AUC .998, λ 3.63): removed .488 (29 fixed, 2 broken); angle to 0 .481 (28 fixed, 2 broken); kept part dropped .287 (NR line 63). Real-start seed 1: removed .512 (33 fixed, 2 broken); angle to 0 .326. In the demo notebook on `im_iso_2x4k` (L20080): held-out 29.7 to 45.3 and 24.6 to 47.7; all validation images 66 fixed, 103 broken.
- Conclusion at the time: removing 1 or 2 modes brings the snowy photos up to real CLIP in every run. Without the real-start rule the error is in the mode's size, not its angle. With the rule the same fix can be made by dropping the angle alone in seed 0, but not in seed 1 (.326). On all images the edit breaks more than it fixes.
- Status now: stands as the one working example of an edit. It is a targeted patch, not a general repair. Not repeated on the CLS-only KAEs.
- Goal: adaptation
- User said (L19415): ""vehicles on snow" is just our internal test to see whether we can fix errors with this. It can't be the only example we show in the paper."
- Files worth reusing: @SP@/report/nb/cell9_0.png, @SP@/report/nb/cell11_1.png, @SP@/cue_conclusion_im_iso_2x4k_wd.log, /home/nsa325/work/koopmann/second_analysis/demo.ipynb

## 11-10 03:15 EST Mode scores: how nameable is one mode compared with one CLS unit or a random direction?
- Question: do a mode's top images share a meaning more than the top images of a single unit of the CLS token or of a random direction? (A score between a floor, random images, and a ceiling, images of one class; judged by CLIP ViT-B/32 embeddings.)
- Setup: `im_dino_w21_wd` (439 modes), and three CLIP KAEs. Sources `SP/ms_score_dino.log`, `SP/ms_score_im_iso_2x4k.log`, `SP/ms_score_im_iso_2x4k_wd.log`, `SP/ms_score_im_realstart_2x4k_wd.log` (L16494).
- Result: DINOv3: floor .483, class ceiling .712. Modes at block 6: median 4.0% of the range; at block 12: 4.6%. CLS units: 0.1% / 0.0%. Random directions with the top 12% of images kept (as TopK does): 6.3%. Strongest 1% of images of a mode: 31.8% (random directions 20.3%). Modes above the 95% point of CLS units: 62% (block 6), 71% (block 12). Modes above the random-direction score: only 13%. Phase test: images with the same phase are more alike than images with opposite phase in 87% of modes (block 6) and 91% (block 12). Score against size of λ .324 (.162 with the other fixed), against angle .218 (.079). CLIP: without weight decay median 1.6%, with weight decay 3.0%, real-start 0.6%. Modes above 5%: 215 (no wd) against 105 (wd). Nameable modes: 61% of 1,022 (about 620) without weight decay, 69% of 479 (about 330) with (MR).
- Conclusion at the time: modes beat single CLS units clearly, but most do not beat a random direction that is given the same top-12% selection. The phase does sort images. Weight decay does not buy cleaner modes; it only removes the weak ones.
- Status now: stands. The comparison with random directions is the honest baseline and most modes fail it. Mode scores on the CLS-only and penalty KAEs were planned (11 Nov 08:36) and never run.
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/ms_score_dino.log, @SP@/ms_score_im_iso_2x4k.log, @SP@/ms_score_im_iso_2x4k_wd.log

## 11-10 03:31 EST Phase wheels: do different phases of one mode show different things?
- Question: if the images of a mode are sorted by phase into six 60° bins, does each bin show its own kind of picture?
- Setup: `im_dino_w21_wd`, 415 modes, then a chosen set. "Gap" = how much more alike images in the same bin are than images in different bins. Sources `SP/phase_wheel.log` (L16907), `SP/phase_wheel_chosen.log` (L17376).
- Result: gap at 10/50/90% of modes: −.004 / .024 / .066. Largest: mode 750, gap +.157 (69% of the floor-to-ceiling range): 0° Shih Tzu, 60° Australian Silky, 180° salamanders. Chosen modes: 667 (snowmobiles at 60°, 3,480 images), 211 (flames at 60°, glowing things at 120°, small birds at 180°), 71 (birds of prey at 0°, bell peppers at 180°), 37 (open water at 120°, 4,472 images; pools at 180°, 6,441 images), 247, 644, 535. Sheets that are clean of the no-show classes: 179, 997, 872, 965, 239, 655, 975, 990, 667, 211, 71.
- Conclusion at the time: in a minority of modes the phase picks between distinct subjects. For the median mode the effect is small (.024).
- Status now: stands (one DINOv3 KAE). Sheets were not shown to the user.
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/phase_wheel.log, @SP@/phase_wheel_chosen.log

## 11-10 03:35 EST Matched edits: is turning a mode different from scaling it?
- Question: for edits that move the state by the same amount, does a turn change more answers than a rescaling? Do they move the state in different directions?
- Setup: `im_dino_w21_wd`, 439 modes. Edits: turn ±30°, scale ×1.52 and ×0.48 (matched to the 30° turn), turn ±60°, scale ×2, remove. Source `SP/matched_edits.log` (L16156).
- Result: median state move: turn ±30° .0248, ×1.52 .0264, ×0.48 .0254; answers changed .0167 to .0183. Turn ±60° .0476 to .0482, ×2 .0515, removed .0476; answers changed .0350 to .0383. A turn changes more answers than the matched scaling in 46% of modes (30°) and 42% (60°); median ratio 1.00. Direction of the state move (median cosine): turn +60° against removal .479; against ×2 −.451; +60° against −60° −.478; ×2 against removed −.949.
- Conclusion at the time: a turn is a different edit from a rescaling (different direction), but not a stronger one.
- Status now: stands.
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/matched_edits.log

## 11-10 03:57 EST Dial test: does turning a mode's phase move an image toward the family that lives at the new phase?
- Question: if a mode shows family A at one phase and family B at another, does turning an A image's phase to B's make the state look more like B?
- Setup: `im_dino_w21_wd`, 439 feature modes; 413 modes have at least 3 "families" (groups of classes at a phase); 11,232 turns tested. Source `SP/dial_test.log`, `SP/dial_test.pt` (L17032).
- Result: the destination family gains most among the non-source families in 45.6% of turns (chance 20.9%). Per mode, share won at 10/50/90%: .35 / .47 / .58; above the mode's own chance in 100% of modes. Cosine of the change of the state with the family's direction: destination +.244, others +.014, source −.263. By turn: 60° wins 25.0%, 120° 54.3%, 180° 73.4%, 240° 51.7%, 300° 24.2%. Removed: source −.320. Doubled: +.271. But the KAE's answer falls in the destination's 3 classes before .012, after .014. Share won against size of λ .169, against angle .038.
- Conclusion at the time: the phase is a real dial for the direction of the state, but turning one mode almost never changes the final answer.
- Status now: stands. See the next entry for what happens with more gain.
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/dial_test.pt, @SP@/dial_test.log

## 11-10 04:08 EST Dial with gain: turning plus boosting a mode
- Question: if the mode is boosted while it is turned, does the answer move to the destination family?
- Setup: `im_dino_w21_wd`, 105 modes, 2,794 cases; gain ×1, ×2, ×4, ×8, with and without the turn. Source `SP/dial_gain.log` (L17690). Photo galleries of the cases in `SP/dial/` (`SP/dial_gallery_x2.log`, `_x4`, `_x8`, 04:13, L17725); four of the twelve sheets were looked at and found clean of the no-show classes (`dial_211_60_180_x2`, `dial_211_60_120_x2`, `dial_71_0_180_x8`, `dial_667_60_120_x4`).
- Result: share of source images answered as the destination family: before .015 (source family .253, ceiling .253). With the turn: ×1 .016, ×2 .022, ×4 .041, ×8 .077 (source falls to .140). Same gain without the turn: .015 / .015 / .017 / .020. Entropy of the answer 1.12 before, 1.96 at ×8 with the turn. Tight families (ceiling at least .6, 265 cases): ×8 with the turn .190, without .041 (before .026, ceiling .705). Ceiling at least .3 (1,006 cases): .167 against .037.
- Conclusion at the time: turn plus a strong boost moves 8% of answers (19% for tight families), four to five times more than the boost alone. It also makes the answer less sure.
- Status now: stands. An effect, but small.
- Goal: meaning, adaptation
- User said: nothing tied to it.
- Files worth reusing: @SP@/dial_gain.log, @SP@/dial/dial_211_60_180_x2.png, @SP@/dial/dial_211_60_120_x2.png, @SP@/dial/dial_71_0_180_x8.png, @SP@/dial/dial_667_60_120_x4.png

## 11-10 04:14 EST The real-start rule (`im_realstart_2x4k_wd`)
- Question: if every mode must start at block 6 as one signed strength (second number zero), does the angle of λ get a fixed meaning, and what does it cost?
- Setup: CLIP, older state, wd 0.1, 8 epochs, flags `--real_start=True --lambda_overlap=0.1`, wandb butbn6t1. Ran 03:10 to 04:14 (`SP/night/runner.log`). Control: `im_iso_2x4k_wd` (wandb 4ujgz6hv).
- Result: accuracy .50898, agreement .61066 against .52166 / .62466 for the control (`SP/night/im_realstart_2x4k_wd.log`, `SP/night/im_iso_2x4k_wd.log`). Per epoch: .4480 .4707 .4783 .4859 .4916 .4989 .5049 .5090 against .4590 .4823 .4871 .4964 .5023 .5106 .5168 .5217. Built part only .2105 | .2364; kept part only .0215 | .0204; no turn .0043 (`SP/kept_built_im_realstart_2x4k_wd.log`). Built share of the output at 25/50/75%: .827 / .914 / .944. 289 modes in use (479 in the control). Angle at 10/50/90%: 27.4 / 44.7 / 71.3°; size of λ .989 / 3.117 / 4.465.
- Conclusion at the time: the rule costs 1.3 points and leaves fewer modes in use. The angles become large (median 45° against about 10°), and most of the answer now comes through the built part.
- Status now: stands. The rule was kept in the plan of 16:59 and tested again without weight decay in `cls_add_real` (cost 0.1 to 0.2 points). The meaning of the angle under the rule with an additive control was undone on 11 Nov (see 11-11 10:31).
- Goal: meaning
- User said (L19415): see the quote under 02:57. Also (16:18, L21422): "Are we still doing the rule where block 6 can't tune phase?"
- Files worth reusing: @SP@/night/im_realstart_2x4k_wd.log, @SP@/kept_built_im_realstart_2x4k_wd.log

## 11-10 04:22 EST Steerability: can a boosted mode steer CLIP's answer to a chosen class?
- Question: what share of modes can push an image's answer to a class when boosted and turned? (Following the idea of Joseph et al.; our protocol differs, so the numbers are not comparable with their 11%.)
- Setup: `im_iso_2x4k_wd` (CLIP), 240 modes, 6 phase directions, 200 images per case, gain ×2, ×4, ×8; a mode counts as steerable if its score is above 0.1. Pilot on 8 modes first. Sources `SP/steerability_test.log`, `SP/steerability_im_iso_2x4k_wd.log` (L17576).
- Result: with the 1,000 class prompts: steerable modes 31.2% (×2), 67.5% (×4), 80.4% (×8); score above 0.5: 5.0 / 25.8 / 49.2%; median best score .054 / .221 / .465. With 19,404 words: 0.4 / 3.8 / 14.2%. Steerable directions per mode 3.8 / 4.6 / 5.7; distinct top classes 3.1 / 3.9 / 4.8; each direction to its own class in 45.0 / 42.3 / 37.5% of modes. Best score at ×8 against size of λ .784, against angle .425. Examples: mode 142 gives Saluki at 0 to 60° and swing at 180 to 300°; mode 355 spoonbill, bald eagle, volleyball; mode 385 jaguar, lion, terrapin.
- Conclusion at the time: most modes can steer the answer to a class if boosted 4 to 8 times, and different phases of one mode steer to different classes. Modes with a large λ steer best.
- Status now: stands for this run. Not repeated on later KAEs.
- Goal: meaning, adaptation
- User said: nothing tied to it.
- Files worth reusing: @SP@/steerability_im_iso_2x4k_wd.log

## 11-10 04:35 EST Angle readable: is a large angle a sign that the mode's content is not yet readable at block 6?
- Question: do modes with a larger angle hold content that a linear read-out cannot yet find in the real block-6 state?
- Setup: seven KAEs; for each mode, how well its block-12 value can be read linearly from the real block-6 state (R²). Rank correlation with the angle, size and usage held fixed. Sources `SP/angle_readable_*.log` (L18154).
- Result: `im_iso_2x4k_wd` −.232 (raw −.278); `im_iso_2x8k_wd` −.150; `im_realstart_2x4k_wd` −.119; `im_iso_2x4k` +.000; `im_iso_2x4k_s1` −.017; `im_iso_2x4k_kl0.1` −.006; `im_dino_w21_wd` +.088. Median R² at block 6 about .75 (CLIP), .42 (DINOv3); at block 12 .996 / .989.
- Conclusion at the time: the reading "angle = how early the concept is readable" shows in the CLIP runs with weight decay and nowhere else. Retracted in the morning report.
- Status now: later corrected: the first reading came from two weight-decay runs and did not replicate in the other five.
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/angle_readable_im_iso_2x4k_wd.log, @SP@/angle_readable_im_iso_2x4k.log

## 11-10 04:40 EST Seed match: which properties of a mode come back in a second training seed?
- Question: if the same KAE is trained twice, do the modes, their sizes and their angles come back?
- Setup: `im_iso_2x4k` (wandb ez6cx5a3) against `im_iso_2x4k_s1` (wandb 32mst724; .52114 / .6216); modes paired by how alike their pushes on the state are; 1,022 pairs. Source `SP/seed_match_s0_s1.log` (L18361).
- Result: likeness of paired modes at 10/25/50/75/90%: .018 / .052 / .275 / .453 / .600 (random pairs −.002). Mutual best matches: 315 (30.8%). Rank correlation across seeds: usage +.548; size of λ +.727 (+.779 with usage fixed); angle +.208 (+.129); shuffled +.002 ± .031. Among mutual best pairs with likeness above .5 (159): angle +.256; angle difference at 25/50/75% 2.1 / 4.3 / 11.6° against 2.4 / 6.8 / 13.5° for random pairs; size ratio 1.06 / 1.13 / 1.27. The phase of z in one seed is a fixed function of the phase in the other: strength median .546 (random .156), mutual best .717, likeness above .7: .869. Sign of the angle agrees in 49.1% (chance 50%).
- Conclusion at the time: the size of λ and the phase of z reproduce across seeds. The angle of λ does not, without a rule that fixes its reference.
- Status now: stands. Only about a third of the modes have a clear twin in the other seed.
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/seed_match_s0_s1.log

## 11-10 04:49 EST Dead modes under weight decay
- Question: how many modes are never used when weight decay is on?
- Setup: count of modes that are kept on no validation image, from the same logs as the angle-readable test (`SP/angle_readable_*.log`).
- Result: without weight decay: 0 dead modes in `im_iso_2x4k`, `im_iso_2x4k_s1` and `im_iso_2x4k_kl0.1` (4 modes are used on 1 to 199 images in one of them). With weight decay: 545 dead of 1,024 (`im_iso_2x4k_wd`), 445 (`im_iso_2x8k_wd`), 735 (`im_realstart_2x4k_wd`), 567 (`im_dino_w21_wd`).
- Conclusion at the time: weight decay 0.1 kills 43 to 72% of the modes.
- Status now: stands. Led to dropping weight decay at 10:28.
- Goal: groundwork
- User said (L19415): "Dead modes are no good, yet weight decay was helping us close the train/val gap."
- Files worth reusing: @SP@/angle_readable_im_iso_2x4k_wd.log, @SP@/angle_readable_im_realstart_2x4k_wd.log

## 11-10 04:50 EST Angle persist: with the real-start rule, a small angle means the block-6 direction survives
- Question: does the angle of λ say how much of a mode's block-6 direction is still there at block 12 in the real states ("persistence")?
- Setup: four KAEs; persistence measured two ways: v1 on the whole state, v2 by regression against the real states without the eigenvalue step or the decoder. Rank correlation with the angle, size and usage held fixed. Sources `SP/angle_persist_*.log` (v1), `SP/angle_persist_reg_*.log` (v2) (L18553, L18659); seed 1 in NR line 202.
- Result (v2): real-start −.470; real-start seed 1 −.444 (raw −.215; seed 0 raw −.223); `im_iso_2x4k_wd` +.065; `im_iso_2x4k` +.012; `im_iso_2x4k_s1` −.071. v1: +.033, +.035, +.070; real-start −.092. Persistence itself is small: median .045 to .062 in v2.
- Conclusion at the time: with the rule, a larger angle goes with less of the block-6 direction surviving, and this replicates in a second seed. Without the rule there is no link.
- Status now: stands for the two real-start runs with weight decay on the older state. Not re-tested on `cls_add_real`. The effect is on a small quantity (median persistence about .06).
- Goal: meaning
- User said: nothing tied to it.
- Files worth reusing: @SP@/angle_persist_reg_im_realstart_2x4k_wd.log
