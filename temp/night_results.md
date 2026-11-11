# Ledger of the night's results (2026-11-10), for the morning report. Append only.

## Pictures checked by eye (no bugs, no arachnids)
- inv/contact_scratch.png (sent 03:57): photo | real | recon | pred for 9 photos. State error left: real .02-.14, pred .09-.26 (12995 .219, 33079 .262, 42917 .182, 32814 .187 are the weak ones); the bird against the sky (32814) is lost in all three columns.
- wheel sheets clean (no reptiles either): largest gap 179, 997, 872, 965, 239; median gap 655, 975, 990; chosen modes 667, 211, 71.
- wheel sheets with snakes / lizards / salamanders (not bugs; hold back): 750, 810, 569; chosen 247 (one snake at 60 deg), 644 (small snake in a hand at 120 deg), 37 (snake at 0 deg), 535 (snakes at 0 deg, Komodo dragon at 240 deg).

## Chosen-mode wheels (modes edited in the inversion), what sits where
- 667: snowmobiles / snow ploughs in snowy woods at 60 deg (3,480 images); poles, towers, bare trees at 120 deg; 0 deg mixed; 180-300 deg fields, grids.
- 211: flames / lighters / volcano at 60 deg; glowing things in the dark (carved pumpkins, lights) at 120 deg; small birds at 180 deg.
- 71: birds of prey against the sky at 0 deg; gulls, parachute, hummingbird at 60 deg; birds at feeders / rotisserie at 120 deg; bell peppers at 180 deg; brain coral at 240 deg.
- 247: fruit bowls at 0 deg; citrus piles at 300 deg; snow sports (ski, hockey, dog sled) at 180-240 deg; cloth / curtain at 60-120 deg.
- 644: bread at 0 deg; radiators at 60 deg; oscilloscopes / dials at 180 deg; white foam, cauliflower, snow peaks at 240 deg; snow and white water at 300 deg.
- 37: open water with animals at 120 deg (4,472); pools, orcas, swimmers at 180 deg (6,441); dogs and birds in snow at 240 deg; ships at 60 deg.
- 535: knitwear at 240-300 deg; furry faces (koala, marmoset) at 180 deg; reptile skin at 0 deg.

## Matched edits (DINOv3, 439 modes): see matched_edits.log. A turn is as strong as a rescaling at matched state move; turn vs removal cosine of class-score change +.247 at 30 deg (+.479 at 60 deg), as plane geometry predicts (sin(theta/2) = .259, .5).

## Dial test (dial_test.py, DINOv3, 413 modes, 11,232 turns; finished 03:58; dial_test.log, dial_test.pt)
- Turn a mode from a source family's direction to a destination family's direction (families = 60 deg slices of the block-6 phase, strongest 120 images), predict block 12, read DINOv3's final embedding.
- The destination gains most among the non-source families in 45.6% of turns (chance 20.9%); above chance in 100% of modes (per mode 10/50/90%: .35/.47/.58).
- Cosine of the embedding's change with a family's place: destination +.244 | other families +.014 | source -.263.
- By turn: 60 deg +.177 | +.006 | -.174 (wins 25.0%); 120 deg +.279 | +.016 | -.303 (54.3%); 180 deg +.315 | +.023 | -.354 (73.4%); 240 deg +.279 | +.017 | -.306 (51.7%); 300 deg +.170 | +.010 | -.180 (24.2%).
- Reading: this is the pattern of a flat circle. If the families sat on a circle in DINOv3's embedding in the order of their phases, a turn by t moves the embedding along the chord, whose cosine with the destination is sin(t/2): .50 at 60 deg, .87 at 120 deg, 1 at 180 deg. Observed relative to 180 deg: .56 and .89 (source side: .49 and .86). The 60 deg turn "loses" only because its chord points at the family 120 deg away, as the circle predicts.
- Rescalings: removed: source -.320 | others +.077; doubled: source +.271 | others -.066 (move along the source family only).
- Class probe: log-probability of the destination's 3 classes +.530, other families +.335, source -.112; removed: source -.125; doubled: source +.216.
- Honest limit: the answer itself hardly changes (answer in the destination's classes .012 before, .014 after): one of about 128 kept modes is turned. dial_gain.py tests the turn with gain x1, x2, x4, x8 against the same gain without a turn.
- Size of lambda and |angle| hardly matter for the dial (rank corr .169 and .038; turns won 45-49% in every fifth of size).

## Dial test with gain (dial_gain.py, DINOv3 im_dino_w21_wd, every 4th mode: 105 modes, 2,794 cases; finished 04:08 EST)
Question: the dial test showed a turn moves the predicted embedding toward the destination family, but the answer hardly changed. Does it change if the turned mode is also made stronger? Control: the same gain without a turn.
Measured: share of the source family's images whose answer (linear probe on the KAE's prediction) is one of the destination family's 3 most common classes. Ceiling = the same share for the destination family's own images (25.3% on average; families are loose).
- before any edit: destination's classes 1.5% | source's classes 25.3%
- turn, gain x1: 1.6% | 23.0%   (no turn: 1.5% | 25.3%)
- turn, gain x2: 2.2% | 21.5%   (no turn: 1.5% | 26.2%)
- turn, gain x4: 4.1% | 18.4%   (no turn: 1.7% | 26.8%)
- turn, gain x8: 7.7% | 14.0%   (no turn: 2.0% | 26.8%)
- entropy of the answer: before 1.12; turn x8 1.96; no turn x8 1.54
- by turn at x8: 60° 9.5%, 120° 7.2%, 180° 5.6%, 240° 6.8%, 300° 9.2% (60° and 300° start at 3.8% because neighbouring slices share classes)
- destination families that agree on their classes (ceiling >= 0.6; 265 cases, ceiling 70.5%): before 2.6%; turn x4 8.3%; turn x8 19.0%; no turn x8 4.1%
Reading: the turn is what brings the destination's classes (4 to 5 times the no-turn control at the same gain) and it takes the source's classes away, which a gain alone does not. But it is a weak switch: one mode of about 128 kept, even 8 times stronger, flips the answer in 8% of images (19% for tight families, a quarter of the ceiling). So: direction of a mode is a dial for the embedding (dial test), and only a nudge for the answer.

## Real-start run im_realstart_2x4k_wd (butbn6t1; CLIP, 8 epochs, wd 0.1, real_start, lambda_overlap 0.1; finished 04:14 EST)
- KAE accuracy / agreement: 50.9 / 61.1 (control im_iso_2x4k_wd: 52.2 / 62.5). One-step FVU .380 (control .366), reconstruction FVU .211 (control .190). Train accuracy 56.7 (control 57.9).
- By epoch it sits 1.1 to 1.2 points under the control throughout (.4480 .4707 .4783 .4859 .4916 .4989 .5049 .5090 against .4590 .4823 .4871 .4964 .5023 .5106 .5168 .5217).
- So fixing where each mode starts (first coordinate only at block 6) costs about 1.3 points of accuracy in an 8-epoch pilot.
- Long run im_dino_w21_wd_e30 started 04:14 EST, wandb id ze4m5akv.

## Retrieval galleries for the dial (dial_gallery.py; DINOv3; gains 2, 4, 8; sheets in dial/)
What it shows: 4 photos of a source family; for each, the val images nearest to the KAE's prediction after the mode is turned to the destination's direction (with gain), against the same gain without a turn.
- Mode 211, 60° to 180° (fire photos): turned predictions retrieve juncos and chickadees already at x2 (3 of 4 photos; the space shuttle goes to stage lights). No-turn control stays fire and candles.
- Mode 211, 60° to 120°: stage lights and guitars in the dark at x2 to x8 (the 120° slice is "glowing things in the dark").
- Mode 71, 0° to 180° (birds of prey against sky): x2 small birds and tennis balls; x4 bell peppers, a rabbit, tennis; x8 bell peppers for all 4. Control stays birds of prey.
- Mode 667, 60° to 120° (snowy woods): kimono group, park bench, camera, street sign: snow-free scenes, no clear destination. The unclear case.
- Caveat: at x8 all 4 photos converge on the same neighbours, so the edit overrides the photo.
- Viewed and sent at 04:13: dial_211_60_180_x2.png, dial_211_60_120_x2.png, dial_71_0_180_x8.png, dial_667_60_120_x4.png (all four checked: no bugs or arachnids). The other sheets were not viewed and not sent.

## Real-start run, analyses (logs *_im_realstart_2x4k_wd.log; control = im_iso_2x4k_wd)
The rule: at block 6 a mode has only its first number (sign and size); the second number at block 12 can only come from the eigenvalue's turn. "Kept part" = Re(λ)·z6 (first number at 12), "built part" = Im(λ)·z6 (second number at 12).
1. K now works through the turn (kept_built.py). Accuracy | agreement: K as trained .5090 | .6107; built part only .2105 | .2364; kept part only .0215 | .0204; no turn, same size .0043. (Control: kept only .2584, built only .0062.) Built share of the predicted latent energy: 10/50/90% .827/.914/.944. Angle of λ against the true mean turn: within .4/.9/1.6° (25/50/75%).
2. Fewer live modes, and less single-meaning. 289 modes are kept on at least 200 val images (control 479). Monosemanticity score, median mode: 0.6% of the floor-to-class range (control 3.0%); strongest 1% of images 17.4% (control 19.9%); modes above the 95% point of CLS units 62% (control 84%). Usage is high: the fifth of modes with the smallest λ is on for a median 80% of images.
3. Angles are large: |angle| 10/50/90% = 27/45/71° (control 2/10/23°). Only 18 of 281 feature modes turn less than 20°.
4. Snow test (cue_conclusion.py): KAE snowy .279, normal .526. Snow mode 493 (AUC .998, λ 3.63, angle 79.8°). Removed: .488 (29 fixed, 2 broken). Built part dropped (angle set to 0): .481 (28 fixed, 2 broken). Kept part dropped: .287. In the control the angle-to-0 edit did nothing. Caveat: mode 493 is nearly all built (80°), so angle-to-0 is close to removal here. Held out: half 0 .297 to .422 (either edit); half 1 .262 to .492 (removed) / .523 (angle to 0). Cost on all ImageNet val (top 4 modes): removed 384 fixed / 604 broken; angle to 0 343 / 533.
5. Words of the kept and built parts (kept_built_words.py, 281 modes). By |angle|: 20-45° 129 modes: kept part's push matches the mode's top images .337, built part's .334, built/kept push size .88. 45-180° 134 modes: kept .095, built .316, size ratio 1.62. Rank correlation of |angle| with "kept part matches top images" -.418 (size and usage fixed: -.405); control +.204 (-.038). Own mode is the best match: whole push 69.9%, kept 44.3%, built 61.9% (control 68.7 / 63.3 / 21.9). Example: mode 493 (79.8°): kept words "philly msnbc umd beijing philadelphia", built words "snow winter winters cold snowy".
6. Class tie (angle_vs_class.py): |angle| with the class tie at block 6 .185; size of λ .818. The angle still does not say object vs look; size does.
7. Phase test (ms_score.py): a mode at block 6 has only a sign; same-sign pairs look more alike than opposite-sign pairs in 99% of modes (gap 2.2% of the range; control 5.1%).
Reading so far: inside the KAE the rule did what it was built for (the turn carries the prediction, the angle knob acts, and the angle separates modes whose kept number still carries the meaning from modes whose meaning is only in the built number). Part of item 5 is geometry: with a large angle the kept part is small. The outside check is the next section.
8. Persistence of a mode's state pattern (push_turn.py, real-start run): cosine of what a mode writes into the state at block 6 against block 12 (through K), median .070. Its rank correlation with |angle|, size and usage fixed: -.607 (control -.124, DINOv3 -.128); with the KAE's code of the TRUE block-12 state: -.440. So under the rule the angle does say how little of a mode's block-6 state pattern is still there at block 12, which is what the overlap loss asks for.

## Two seeds of one setting: what is a property of the feature, what of the run (seed_match.py; im_iso_2x4k seed 0 against im_iso_2x4k_s1 seed 1; CLIP, no weight decay, no rule; log seed_match_s0_s1.log)
Both runs existed already (wandb ez6cx5a3 and 32mst724, same config, seeds 0 and 1). Modes matched one-to-one by how alike they fire over the 50,000 val images (correlation of block-6 sizes).
- Matching: median correlation .275 (random pair -.002); above .5: 19.4%; mutual best pairs 315 of 1,022 (31%). So about a third of the modes come back in the other seed.
- Size of λ agrees across seeds: rank correlation +.73 (all pairs), +.67 to +.78 (mutual best); ratio of the two sizes 1.06 / 1.13 / 1.27 (25/50/75%).
- |Angle| of λ agrees only weakly: +.21 (all pairs), +.24 to +.26 (mutual best; +.20 to +.23 with size and usage fixed; shuffled pairs 0 ± .06). Median difference of the two |angles| 4.3° against 6.8° for two random modes.
- Phase of z is reproducible up to the free convention: for mutual-best pairs the phase in one seed is the phase in the other plus a fixed offset, or its mirror image plus an offset; strength of that relation (0 to 1) median .72 (random pairs .16), .87 for the 38 most alike pairs; above .5 in 93.5%.
- Which way K turns is NOT reproducible: an offset relation should come with eigenvalue angles of the same sign and a mirror relation with opposite signs; that holds in 49.1% of 277 clear pairs (chance 50%).
Reading: across seeds the content of a plane (which image sits at which direction) and the size of λ come back; the turn of λ (how far, which way) mostly does not. My explanation (mine, to be checked): the encoder charts block-6 states and block-12 states in different regions of its input, so it can lay the block-12 circle of a mode at any rotation relative to the block-6 circle; a rotation changes no distance, so the isometric loss does not pin it, and the angle of λ is that free relative rotation. The real-start rule (with the overlap loss) is a way to pin it. Whether it then becomes reproducible needs a second seed of the real-start run.

## Dead modes: weight decay kills half the dictionary (count printed by angle_readable.py)
- im_iso_2x4k (no wd, seed 0): 0 of 1,024 modes kept on no val image (1,022 on 200 or more).
- im_iso_2x4k_s1 (no wd, seed 1): 0 dead.
- im_iso_2x4k_wd (wd 0.1): 545 dead, 479 live. Accuracy the same (52.2 against 52.1).
- im_realstart_2x4k_wd (wd 0.1 + rule): 735 dead, 289 live.
Also: the size of λ depends on the recipe. No wd: 10/50/90% = .26 / .55 / .78 (all under 1). wd: .83 / 3.18 / 4.13. The ordering of modes by size is what carries over, not the level.
Consequence: with wd each live mode is on for about 27% of images (128 of 479) instead of 12.5%. This is a decision for the user (wd was called a net positive on accuracy).

## Steerability (steerability.py; my adaptation of Joseph et al. to KAE modes; CLIP control im_iso_2x4k_wd; every 2nd mode: 240 modes, 6 directions, 200 images)
Steerable = the softmax over the vocabulary moves by a score above 0.10 when the mode is replaced by a large value (2, 4, 8 times its own 99th-percentile size) in its best direction.
- Vocabulary of 1,000 class prompts: 31.2% of modes at x2, 67.5% at x4, 80.4% at x8 (score above 0.5: 5.0 / 25.8 / 49.2%).
- Vocabulary of 19,404 words: 0.4% at x2, 3.8% at x4, 14.2% at x8.
- Code book: of the modes steerable in two or more directions at x4 (156 modes), 4.6 directions are steerable and they lead to 3.9 different top classes; all directions to one class in 6.4%.
- A mode's best score goes with the size of λ (rank correlation .78).
Caveat: the share depends heavily on vocabulary and strength, so it is not comparable with their 11% (5,000 concepts, absolute value 150, SAE features at layer 11) beyond "same order at a comparable vocabulary size".

## Queued behind the long run (04:50 EST; both are second seeds of settings that already ran; to cancel, delete the line from night/queue.txt)
- im_realstart_2x4k_wd_s1: the approved real-start run with seed 1. Question: does the rule make the angle of λ reproducible across seeds (rule-free pair: +.21 to +.26, sign at chance)?
- im_iso_2x4k_wd_s1: the wd control with seed 1. Needed because the rule-free pair has no weight decay and the real-start run has; without it a gain could be credited to weight decay.
About 63 minutes each, starting when im_dino_w21_wd_e30 ends (about 11:15 EST).

## The outside check: does the angle say WHEN a mode's concept becomes readable in the real model? No (angle_readable.py, 7 runs)
Per mode: its concept = mean real final embedding of its 50 top images, less the centre. "Readable at block 6" = held-out R² of a ridge map from the true block-6 state to the concept score (10/50/90% of modes about .65 / .75 / .81; about .995 from block 12). Uses the real model only. Rank correlation with readability; in brackets with the other two of angle, size, usage held fixed.
run | dead of 1,024 / live | |angle| | size of λ | usage
- im_iso_2x4k_wd (CLIP, wd): 545 / 479 | -.278 (-.232) | -.281 (-.023) | +.329 (+.229)
- im_iso_2x8k_wd (CLIP, wd, wide): 445 / 579 | -.198 (-.150) | -.282 (-.077) | +.347 (+.263)
- im_realstart_2x4k_wd (CLIP, wd, rule): 735 / 289 | +.017 (-.119) | -.177 (+.125) | +.290 (+.271)
- im_iso_2x4k (CLIP, no wd, seed 0): 0 / 1,022 | -.121 (+.000) | -.447 (-.381) | +.209 (+.005)
- im_iso_2x4k_s1 (CLIP, no wd, seed 1): 0 / 1,024 | -.093 (-.017) | -.408 (-.377) | +.117 (+.004)
- im_iso_2x4k_kl0.1 (CLIP, no wd, KL 0.1): 0 / 1,020 | -.058 (-.006) | -.399 (-.307) | +.257 (+.078)
- im_dino_w21_wd (DINOv3, wd): 567 / 455 | -.103 (+.088) | -.459 (-.302) | +.569 (+.456)
Reading: a tie between a large angle and a late-built concept shows in the two CLIP runs with weight decay (-.23, -.15) and nowhere else (three CLIP runs without weight decay: 0; DINOv3: +.09; real-start: -.12). Not a stable meaning of the angle; in the runs without weight decay and on DINOv3 the size of λ carries it (a strongly amplified mode stands for a concept block 6 does not yet show). My earlier reading of the real-start angle as "carried over against built late" is NOT confirmed from outside.
Dead modes again: every run with weight decay loses 43 to 72% of its modes; no run without it loses any.

## Does weight decay make the surviving modes more single-meaning? Per mode a little, in total no (ms_score.py on im_iso_2x4k, no wd, against im_iso_2x4k_wd)
Score of a mode = how alike the images that fire it look to DINOv3, as a share of the way from two random images (0%) to two images of one class (100%).
- no wd: 1,006 feature modes; median 1.6% at block 6, 1.9% at block 12; strongest 1% of images 18.6%; above the 95% point of CLS units 71%.
- wd 0.1: 460 feature modes; median 3.0% / 3.7%; strongest 1% 19.9%; above CLS units 84%.
- Counts of modes above a fixed level (block 6; block 12 in brackets): above 5% of the range: no wd 215 (286), wd 105 (165); above 10%: no wd 53 (100), wd 19 (28).
- Phase test: same-direction pairs above opposite pairs in 99% of modes without wd (gap 4.0% of the range) and 100% with (5.1%).
Reading: the median is lower without weight decay because that run keeps about 500 extra weak modes (the fifth with the smallest λ sits at the floor); in absolute numbers it has about twice as many modes above any level. So weight decay does not buy cleaner modes, it removes modes. Accuracy is the same (52.1 against 52.2 on 4096; on 8192 wide wd gave +1.2).

## The snow knob on the run without weight decay (cue_conclusion.py im_iso_2x4k): the demonstration holds in a third run
129 snowy vehicle photos: real CLIP 53.5%; KAE 27.1% (normal photos of the same classes 53.7%). The best snow mode is 344 (AUC .986; on for 98% of snowy and 3% of normal photos; λ size .94, angle 9.6°).
- mode 344 removed: 46.5% (25 answers fixed, 0 broken; normal photos unchanged at 53.7%).
- angle of mode 344 to 0: 29.5% (no effect, as on the wd control).
- removing more modes hurts here (the next candidates are always-on modes).
Three runs now: wd control 30.2 to 49.6%, real-start 27.9 to 48.8%, no wd 27.1 to 46.5%, each by removing one mode of 1,024.

## Decoder-free persistence, version 1: mean state of a mode's top images (angle_persist.py). No tie with the angle anywhere
Persistence = cosine of the mean real block-6 state and the mean real block-12 state of a mode's 50 top images (each less the centre). Median about .10 in every run; repeatable (odd against even top images) .88 to .94.
Rank correlation with |angle|, size and usage fixed: real-start -.09; wd control +.07; no wd seed 0 +.03. Size of λ carries it (no wd: -.53).
Caveat that led to version 2: 128 modes are on per image, so the top images of a mode share many other modes, and the mean pattern is a bundle.

## Decoder-free persistence, version 2 (partial patterns by ridge regression; `angle_persist_reg.py`) — 05:05 EST
A mode's pattern = its partial effect on the real state with the other modes held fixed. Persistence = cosine of its block-6 pattern (U) and what it leads to at block 12 (W). A, B = patterns of the first and second coordinate of the block-12 code.

| Run | modes | persistence median | U with A median | U with B median | persistence vs abs(angle), size and usage fixed |
|---|---|---|---|---|---|
| `im_realstart_2x4k_wd` | 289 | .062 | .200 | .002 | −.470 (raw −.223; halves −.396 / −.533) |
| `im_iso_2x4k_wd` (control) | 479 | .053 | .069 | .002 | +.065 (raw −.174) |
| `im_iso_2x4k` (no wd, seed 0) | 1022 | .045 | .057 | .000 | +.012 (raw −.067) |
| `im_iso_2x4k_s1` (no wd, seed 1) | 1024 | 0.043 | 0.055 | 0.000 | -0.071 (raw -0.051) |

Reading: only under the real-start rule does the angle track how much of a mode's block-6 pattern survives to block 12 in the real model's own states (no K, no decoder), and only there does the first block-12 coordinate carry a visible part of the block-6 pattern (.20 vs .06–.07). In rule-free runs the angle tracks nothing here. Caveats: one seed (second seed queued); in the real-start run angle, size and usage are tangled, so the number that counts is the partial one; persistence itself is small in absolute terms (median .06).

## Inversion edits, numbers (`merge_inv.py`; log `inv/merge.log`, table `inv/edit_numbers.csv`) — 05:05 EST
106 pictures painted and merged into `inv/results.pt`. The sheets (`make_sheets.py`) are NOT made or looked at yet.
- State error left (lower is better): DINOv3's own state .078, KAE in-and-out .134, KAE prediction .160.
- Mode edits arrive in DINOv3's reading of the picture: share of the asked change removed .88, doubled .74, +90° .76, −90° .84 (control mode .73); cosine .69–.82 (control .61).
- Second opinion (CLIP word score for the mode's meaning) does NOT confirm them: removed lowers the word in 6 of 9 photos (below the control in 5), doubled raises it in 5 of 9 (above the control in 4). That is chance. So far no evidence that removing a mode erases the named quality in the picture.
- Colour (5 photos; change in colourfulness against the reference): gray knob −.051, sepia knob −.058, KAE prediction for the gray copy −.118, DINOv3 on the gray copy −.063, on the sepia copy −.094. Strongest on the fruit photo (15059): gray knob −.172, sepia knob −.177. The painting itself carries colour weakly (DINOv3's state of a gray copy still paints with colour .241 on average).

## Mode naming on the run without weight decay (`name_modes.py im_iso_2x4k 0`; log `names_im_iso_2x4k.log`) — 05:10 EST
How often a mode's push on the final embedding matches the top images of its own mode better than those of any other mode:

| Run | modes | whole push | kept part | built part |
|---|---|---|---|---|
| `im_iso_2x4k` (no wd) | 1,022 | 60.6% (best 5: 68.3%) | 59.6% | 12.9% |
| `im_iso_2x4k_wd` (control) | 479 | 68.7% | 63.3% | 21.9% |
| `im_realstart_2x4k_wd` | 289 | 69.9% | 44.3% | 61.9% |

Reading: without weight decay a smaller share of modes is nameable (61% against 69%), but out of twice as many modes, so more modes in total are nameable (about 620 against about 330). As in the control, the push comes from the kept part; the built part names almost nothing without the real-start rule.

## Inversion sheets, looked at (`make_sheets.py`; `inv/sheet_modes_1.png`, `inv/sheet_modes_2.png`, `inv/sheet_colour.png`) — 08:58 EST
Sent to the user at 08:58. Two cells (photo 12995, mode removed and mode turned −90°) are blank: their reading or target had a no-show class among its three best answers. `make_sheets.py` now applies that rule to every cell.
- Painting DINOv3's own block-12 state gives back the scene and usually the object (obelisk .99, snowmobile .89, orange .98, sock 1.00, volcano 1.00); hue is unreliable (oranges come out lemon yellow) and the bird in the sky photo is lost.
- Painting the KAE's prediction gives the right scene (winter woods, snowy woods, citrus pile, open water, striped socks, volcano) with less object detail (the obelisk becomes a trunk, reads as obelisk .17; snowmobile .67; orange .92; sock .97; volcano 1.00). Reading agrees with the state's own answer in 89% of predictions.
- One-mode edits: visible as the mode's quality only for mode 211 (fire or glow): removing it dims the fire in the volcano photo and removes the lights in the dark photo; doubling adds glow. For snow (667, 644), fruit (247), water (37), knit (535) and sky (71) the removal rearranges the picture and the quality stays (snow is still there with the snow mode removed).
- Colour knobs (every mode multiplied by its gray or sepia factor from `alter_factors.pt`): the fruit photo loses colour (.526 to .384 gray, .378 sepia), water with the sepia knob (.231 to .117), sky with the sepia knob (.126 to .038); socks and the volcano do not. The KAE's prediction for the actual gray copy is duller still (.322 fruit, .084 water, .044 sky).
- Limit of the painter: DINOv3's own state of the gray copy still paints in colour (fruit .550), so part of the weak colour effect is the painting, not the KAE.

## Long run finished: `im_dino_w21_wd_e30` (wandb ze4m5akv), 30 epochs, 04:14 to 09:15 EST

Same recipe as `im_dino_w21_wd` (DINOv3 ViT-B/16, ImageNet 3 views + the 21k images, 5,001,140 training states, weight decay 0.1); only the epoch count differs.

| Run | Epochs | KAE accuracy | Agreement | One-step FVU | Train accuracy | Train minus val |
|---|---|---|---|---|---|---|
| `im_dino_w21_wd` | 12 | 61.78 | 66.49 | .4387 | 69.25 | 7.5 |
| `im_dino_w21_wd_e30` | 30 | 62.90 | 67.91 | .4214 | 72.30 | 9.4 |

Real model 83.94, linear map 27.16. Last five epochs of the long run (accuracy / agreement): .6192/.6679, .6206/.6690, .6230/.6718, .6264/.6756, .6290/.6791.
Reading: 2.5 times the epochs buys +1.1 accuracy and +1.4 agreement; train accuracy rose by 3.1, so most of the extra fitting did not carry to val. Training time is not the lever.

## Input-ceiling probe, set up 09:40 EST (results to follow)

Script `ceiling_probe.py`. Three plain MLPs on the KAE's data with the reference run's standardization, Muon (wd 0.1) + AdamW from train.py, linear decay, batch 1024:
- mimic probe: block-6 summary to the block-12 state, 4 hidden layers of 4096 (the KAE's encoder and decoder stacked, no latent), losses = state FVU + embedding + 0.03 KL (the KAE's prediction terms only); read through the real head, so it reports the KAE's own accuracy and agreement.
- label probe at block 6 and at block 12: summary to the 1,000 classes, 2 hidden layers of 4096, cross-entropy on labels.
Queue order: `probe_clip_b6` (8 epochs, reference `im_iso_2x4k_wd`), `probe_dino_b6` (12 epochs + w21, reference `im_dino_w21_wd`), then `im_iso_2x4k_wd_s1`.
CPU code check on 40,000 validation states (width 512, 6 epochs; not a result): all three learn; the block-12 label probe reaches the real model's level, the block-6 probes are far behind at that size.

## Second real-start seed: does the angle reproduce? (10:20 EST; seed_match_realstart_wd.log, angle_persist_reg_im_realstart_2x4k_wd_s1.log)
Run `im_realstart_2x4k_wd_s1` (wjx7afbv, seed 1): KAE accuracy .5107, agreement .6103 (seed 0: .5090 / .6107); 299 live modes (seed 0: 289).
Matched modes across the two seeds (mutual best pairs; rank correlation, in brackets with size and usage of both runs held fixed):
| | pairs | size of λ | |angle of λ| | chance for the angle | twins' |angle| difference, median | random modes' difference |
|---|---|---|---|---|---|---|
| real-start rule, wd (seed 0 vs 1) | 108 | +.794 (+.780) | **+.603 (+.529)** | +.002 ± .099 | 8.9° | 21.5° |
| no rule, no wd (im_iso_2x4k vs _s1) | 315 | +.673 (+.656) | +.244 (+.200) | +.006 ± .055 | 4.3° | 6.8° |
- Under the rule the |angle| is a property of the feature to a fair degree (+.60), not only of the run (+.24 without the rule). NOT a clean comparison: the rule-free pair has no weight decay (the control seed with wd is queued).
- The sign of the angle stays arbitrary (45.3% agreement, chance 50%): flipping the second coordinate flips it. Only |angle| can mean something.
- Caveat on the flagship example: snow mode 493 (79.8°, size 3.63) has its twin in seed 1 at mode 1022 with 33.9° (size 4.43), fired alike .94. The snow mode's angle did NOT reproduce. cue_conclusion.py on seed 1 started 10:22 to see what the edits do there.
- At block 6 a mode has only a sign, and twins agree on it up to a global flip: relation strength, mutual best pairs, 10/50/90% .632 / .939 / .999.
Persistence check against the real states (no K, no decoder), seed 1: rank correlation of |angle| with persistence, size and usage fixed, **-.444** (seed 0: -.470; three rule-free runs +.065 / +.012 / -.071). Replicates. Raw correlation -.215 (seed 0 -.223). Split halves of modes -.467 / -.392.

## Weight decay: what it does to the train/val gap (wandb, last epoch; CLIP)
| run | train accuracy | val accuracy | gap | live modes |
|---|---|---|---|---|
| width 4096, no wd (ez6cx5a3) | .6161 | .5211 | 9.5 | 1,022 |
| width 4096, wd 0.1 (4ujgz6hv) | .5794 | .5217 | 5.8 | 479 |
| width 8192, no wd (l4sdh1k1), epoch 5 of 8, then crashed | .6205 | .5243 | 9.6 | |
| width 8192, wd 0.1 (882lfgmo), epoch 5 of 8 | .5702 | .5202 | 5.0 | |
| width 8192, wd 0.1, final | .6091 | .5360 | 7.3 | 579 |
- Weight decay closes the gap by lowering training accuracy (-3.7 points at width 4096); validation accuracy does not move (+.06).
- Width 8192, same epoch: no wd is .4 ahead. The earlier "+1.2 for wd at width 8192" compared the wd run's last epoch with the crashed run's epoch 5. Per epoch the no-wd run had slowed to +.2, the wd run was still gaining +.5 to +.6, so wd probably ends ahead by under a point; unconfirmed.
- Snow test on seed 1 (cue_conclusion_im_realstart_2x4k_wd_s1.log), snow mode 1022 (33.9°): removal .512 (33 fixed, 2 broken; normal photos .526); built part dropped (angle to 0) .326 (7 fixed); kept part dropped .426 (21 fixed). Held out: half 0 .297 to .391 (removed) / .281 (angle to 0) / .391 (kept dropped); half 1 .246 to .554 / .308 / .477. So the REMOVAL knob replicates on the new seed, but "dropping only the built part takes back the snow error" does not: in seed 0 that edit was nearly a removal (80°), in seed 1 the snow mode's effect sits mostly in its kept part. Cost on all ImageNet val, top 4 modes removed: 654 fixed / 966 broken.

## 10:30 EST: weight decay dropped (user: "yeah we should drop weight decay then")
New KAE runs use no weight decay. Queue after the CLIP probe: `im_iso_2x4k_w21` (no wd, + w21 images; compare with `im_iso_2x4k`: 52.1 / 62.2, gap 9.5, 1,022 live modes), `probe_dino_b6`, `im_realstart_2x4k` and `im_realstart_2x4k_s1` (the real-start rule without wd, two seeds; compare with the rule-free no-wd pair `im_iso_2x4k`, `_s1`). The wd control seed `im_iso_2x4k_wd_s1` was taken off the queue (moot).

## 10:38 EST: doubled learning rate queued first (user: "yes go for it")
`im_iso_2x4k_lr2`: Muon .015, AdamW .0015, everything else as `im_iso_2x4k` (52.1 / 62.2, no wd, seed 0). Tests whether the rate is too low. Probe after 1 of 8 epochs: mimic 49.3 / 58.6; label probe block 6 55.7, block 12 73.5.

## Demo notebook (user: "clean up the notebooks ... i want to see a demo of what works")
`/home/nsa325/work/koopmann/second_analysis/demo.ipynb` + `demo_utils.py` (+ `cache/`: zero-shot classes, snowy states); built by `make_demo.py` here, runs on the CPU in under a minute. The 8 old notebooks were moved to `second_analysis/archive/` (not deleted). New numbers from it on `im_iso_2x4k`: snow mode 344 picked on either half; held-out snowy 29.7 -> 45.3 and 24.6 -> 47.7, normal unchanged; all val: 66 fixed, 103 broken of 2,405 images; |λ| vs class tie +.79 (+.83 usage fixed) over 1,022 modes; CPU timing 256 images: blocks 7-12 about 1.5 s, KAE about 50 ms (about 30x). Smallest-|λ| modes' top images share nothing visible.

## 11:43 EST: input-ceiling probe on CLIP finished (`probe_clip_b6`, e2gcy7w9)
Plain MLP (4 hidden layers of 4096, no latent, no TopK, no K), same input, losses and data as the KAE, wd 0.1, 8 epochs:
mimic probe 53.03 / 63.59 (state FVU .3586, train accuracy 58.93). KAE: 52.2 / 62.5 (wd), 52.1 / 62.2 (no wd); MLP-in-place-of-K 53.8 / 64.2; real CLIP 63.3.
Per epoch: 49.3, 50.2, 51.0, 51.3, 51.6, 52.3, 52.5, 53.0 (still +.2 to +.7 per epoch at the end, with the rate decaying to 0).
Label probes (supervised, 2 hidden layers): from the block-6 summary 62.3 (train 79.9); from the block-12 summary 78.2 (train 97.6).
Reading: the KAE's structure costs under 1 point. The cap near 52-54 is shared by every network on this input and recipe, so it is the input or the training, not the KAE.
`im_iso_2x4k_lr2` (9ck676yj) started 11:43.

## 12:45 EST: user's direction for the day; online trainer queued
User: close the gap; stay near the architecture; extra context enters as a CONTROL input; goals = speed, adaptability to a new domain, meaning of phase/scale; aim high, train longer if promising; code short and simple.
`im_iso_2x4k_lr2` after 7 of 8 epochs: 52.22 (reference at epoch 7: 52.06). Doubling the rate does nothing. Train accuracy keeps rising (61.6) while val is flat: the KAE fits its 3 cached crops.
Earlier grid run `im_grid_2x4k_r2` (quarter-grid patch means as extra state): 52.6 / 62.9, +0.5 only.
New script `control_train.py` (imports train.py's model, losses, evaluation): fresh random crops at every step from the real model, and block-6 patch tokens as a control input (none / add / bilinear). Smoke-tested on CPU; online val states match the cache. Queued first as `on` (runs on_none, on_add, on_bilinear; 30,000 steps of 1,024).

12:46 EST: `im_iso_2x4k_lr2` final 52.39 / 62.02 (reference 52.11 / 62.17; train 62.0 vs 61.6). Doubling the rate changes nothing. `on` job started 12:46.

13:31 EST: `on` job, step 2,500 of 30,000 (~1 step/s, so ~8.5 h in all): on_none 44.7 / 53.2; on_add 54.6 / 65.4; on_bilinear 54.8 / 65.7. Both control KAEs already beat every earlier run (best was 53.8 / 64.5) at 1/12 of training.
13:19 EST: CIFAR-100 adaptation, im_iso_2x4k: KAE 35.5 vs CLIP 64.2 untuned; eigenvalue tuning with 64 shots reaches 41-42; CLIP with scale+shift 69.2. Negative: the ImageNet-only KAE does not transfer.

14:05 EST: user allowed dataset downloads. Got ImageNet-R/Sketch/A/V2 (clip-benchmark shards, not gated) in /scratch/nsa325/datasets/shift; CLIP extraction running on CPU (extract_shift.py -> domain_cache/).
CIFAR-100 with the CC3M-trained KAE (im_cc3m_iso0.1_2x4k_e20): 37.2 untuned, 44.8 at 64 shots; CLIP 64.2 / 69.2. Broader pilot data does not fix transfer.
User idea: a fine-grained animal set (deer/antelope/impala) that DINOv3 confuses; train a KAE there, tune eigenvalues to beat it. Check on ImageNet's own 9 bovid classes (345-353): real DINOv3 85.6%, KAE (im_dino_w21_wd_e30) 62.2%. ImageNet's own classes are too easy for DINOv3; candidate: iNaturalist 2021 mammals (246 species, many antelope and deer). Its download carries terms (non-commercial research), so waiting for the user's OK.

14:10 EST: no download needed for the animal idea: the local ImageNet-21k slice (202 shards) holds 301 hoofed-animal synsets, 26,861 images (animals/scan_w21.py -> animals/w21_hoofed.pt). Built a 49-species set (21 antelope, 12 deer, 9 sheep/goat, 7 cattle; species only, ~115 images each): animals/extract_hoofed.py -> animals/hoofed_<encoder>.pt. Test: animals/adapt_hoofed.py (linear probe on real DINOv3 = the bar; KAE read through that probe; eigenvalue tuning).
imagenetv2 shards hold webp, extract_shift.py fixed and restarted (imagenet-a and v2 done).

14:10 EST: CIFAR-100 with im_w21_wd_2x4k (ImageNet + ImageNet-21k slice): 43.2 untuned (vs 35.5 ImageNet only, 37.2 with CC3M); 64 shots: 50.9 size / 49.4 turn / 52.5 both; CLIP 64.2 / 69.2. Broader labelled-style data helps transfer by ~8 points, still 17-21 behind.

14:15 EST: hoofed species, DINOv3 (im_dino_w21_wd_e30), test accuracy (fit): real DINOv3 + probe 76.9 (99.7); KAE through that probe 38.7; tuned size 41.4 / turn 42.0 / both 41.8 (58.3 fit); KAE + own probe 47.4. Linear probe on the real block-6 summary: 42 (DINOv3), 41 (CLIP); block 12: 77 (DINOv3), 62-63 (CLIP). So the species information is made in blocks 7-12 from patch tokens; the summary-only KAE cannot hold it and tuning eigenvalues cannot add it. This is the test for the patch-control KAEs.
14:30 EST: on job step 5,000: on_none 48.4 / 57.2; on_add 56.4 / 68.1; on_bilinear 56.8 / 68.6. Probing the step-5,000 control KAEs on the CLIP species set (animals/probe_kaes.py).

14:20 EST: CLIP species probe (animals/probe_kaes.py; own probe | real CLIP's probe): real block 6 37.6, block 12 61.1; im_iso_2x4k 40.4 | 36.5; on_none 40.0 | 33.7; on_add 49.7 | 43.6; on_bilinear 51.0 | 43.4 (on_* at step 5,000 of 30,000). Control closes ~half the species gap already. Redo at the end of the on job, then eigenvalue tuning through real CLIP's probe.

14:33 EST: report built and sent: /home/nsa325/work/koopmann/temp/report/report.html (rebuild: report/make_figures.py then report/build_report.py).

15:39 EST: `on` step 10,000: on_none 50.8 / 60.0; on_add 57.8 / 70.2; on_bilinear 57.9 / 70.3 (step 7,500: 49.8/58.9, 57.5/69.5, 57.5/69.8). ~1.04 s/step, ends ~21:25 EST.
15:36 EST: bilinear control shifts (step 7,500, hoofed set, 442 modes): shared λ near 1 (|λ| median 0.98, angle ~2°); |δ|/|λ| median 0.59; turn spread across images ~15° median; species explains ~15% of size change and of turn (10% for |z6|). Mean δ per mode is not 0: fold into λ after training.

16:00 EST: constant modes (step 7,500). Bilinear: none constant (control rewrites λ; mean image eigenvalue size 0.41/0.81/1.74). on_add and on_none: 16 modes each constant in real CLIP on ImageNet val (animals/constant_val.py). on_add: high usage, class explains 0.05-0.12, follow CLS. on_none: usage ~0.1, class explains 0.4-0.74, follow patch mean. TODO: look at photos (hidden-class filter).
Tuning on_bilinear, hoofed, real probe (60.3): untuned 45.6, size 46.4, turn 47.7, both 44.6 (fit 73.5, overfit).
User asked: why patch mean/std in the state with control? TODO: CLS-only state + bilinear control run next after `on` (needs a tokens option in a copy of control_train.py; do not change the running job's file semantics).

16:15 EST: looked at constant-mode photos (animals/constant_gallery_{on_add,on_none}.png; figure report/constant.png). on_none constants = recognisable content (big cats, gymnastics, rugs, birds on branches, printed text, ships, cars, terriers, typewriters). on_add constants = photo conditions (snow/pale background, motion blur, shallow depth of field, framed photos); some rows unclear.

16:20 EST: tune_control (step-7,500 checkpoints, hoofed, real CLIP probe):
real CLIP block 12, its own probe: 60.3
on_bilinear, untuned: test 45.6 (fit 62.1) | largest |size| 0.00, |turn| 0.00
on_bilinear, size: test 46.4 (fit 67.9) | largest |size| 5.43, |turn| 0.00
on_bilinear, turn: test 47.7 (fit 65.5) | largest |size| 0.00, |turn| 3.31
on_bilinear, both: test 44.6 (fit 73.5) | largest |size| 5.70, |turn| 3.42
on_add, untuned: test 46.1 (fit 63.6) | largest |size| 0.00, |turn| 0.00
on_add, size: test 46.4 (fit 68.4) | largest |size| 5.49, |turn| 0.00
on_add, turn: test 47.4 (fit 66.9) | largest |size| 0.00, |turn| 3.37
on_add, both: test 46.1 (fit 72.9) | largest |size| 5.56, |turn| 3.48
on_none, untuned: test 35.8 (fit 47.4) | largest |size| 0.00, |turn| 0.00
on_none, size: test 36.8 (fit 54.5) | largest |size| 3.80, |turn| 0.00
on_none, turn: test 37.1 (fit 51.4) | largest |size| 0.00, |turn| 3.26
on_none, both: test 36.3 (fit 61.2) | largest |size| 3.74, |turn| 3.17
finished

16:22 EST: tune_control with CLIP zero-shot head over 49 species names (step-7,500 checkpoints):
real CLIP, zero-shot: test 43.7 (fit 45.7)
on_add, untuned: test 31.6 (fit 34.9) | largest |size| 0.00, |turn| 0.00
on_add, size: test 34.7 (fit 39.0) | largest |size| 5.68, |turn| 0.00
on_add, turn: test 34.8 (fit 39.3) | largest |size| 0.00, |turn| 3.39
on_add, both: test 35.0 (fit 45.2) | largest |size| 5.99, |turn| 4.10
on_bilinear, untuned: test 32.6 (fit 34.2) | largest |size| 0.00, |turn| 0.00
on_bilinear, size: test 34.2 (fit 39.2) | largest |size| 5.69, |turn| 0.00
on_bilinear, turn: test 34.9 (fit 38.9) | largest |size| 0.00, |turn| 3.68
on_bilinear, both: test 35.8 (fit 45.7) | largest |size| 5.52, |turn| 3.58
on_none, untuned: test 25.6 (fit 27.1) | largest |size| 0.00, |turn| 0.00
on_none, size: test 28.6 (fit 32.5) | largest |size| 3.79, |turn| 0.00
on_none, turn: test 27.9 (fit 32.8) | largest |size| 0.00, |turn| 3.48
on_none, both: test 26.9 (fit 40.5) | largest |size| 3.65, |turn| 3.34
finished
Also: label probe (2x4096 MLP, supervised) on block-6 summary: 62.3% ImageNet val; block 12: 78.2%; CLIP head straight on block 6: 0.2%.

 17:25 EST: DECISION (user: "make a decision on how to proceed"): one plan. New script cls_train.py (state = CLS token only; block-6 patch tokens as additive control; arms add / add_real (real-start rule) / mlp baseline; no weight decay). Smoke test first, then stop `on` and launch.
 Frozen-patch test on real CLIP (ceiling_frozen.py, 5,000 val images, CPU): real 63.2; real blocks 7-12 with patch tokens held at block 6: 3.3 (agreement 3.7). So it is NOT a ceiling: the real blocks cannot be reused with stale patches; a trained map is needed (our on_add: 57.8 at step 10,000).
 17:30 EST: `on` stopped at step ~15,100 (step 15,000: none 52.2/61.6, add 58.7/71.4, bilinear 58.8/71.8; best checkpoints kept). Started `cls` (cls_train.py, 30,000 steps): cls_add 2u0pkn5x, cls_add_real fqr98z82, cls_mlp 4dbf2wa3. Old queue parked in night/queue_parked.txt.
 18:25 EST: report updated (report/report.html): on run through step 15,000, cls run table, frozen-patch fact, block-6 probe note, species by zero-shot head, constant modes section, future steps per prong with sources. cls step 2,500: add 54.0/64.8, add_real 53.6/64.5, mlp 55.3/66.5.
 19:00 EST: user: long run (300k steps, with/without mixup) is for later on a better GPU. Do NOT launch it here after cls ends.
 01:58 EST (11 Nov): `cls` FINISHED (30,000 steps). Final accuracy / agreement / state FVU: cls_add 59.08 / 72.66 / 0.277; cls_add_real 58.94 / 72.24 / 0.282; cls_mlp 60.46 / 74.89 / 0.248. At step 15,000: cls_add 58.09, cls_add_real 57.87, cls_mlp 59.55 (on_add was 58.7, so CLS-only state costs 0.6). Real-start rule costs 0.1-0.2. Eigenvalue step + TopK cost 1.4 vs the MLP baseline. Real CLIP 63.3.
 02:30 EST: angle_tests.py on the final cls checkpoints (50,000 val images).
  Eigenvalue ablation, accuracy (cls_add / cls_add_real): as trained 59.09 / 58.94; angle dropped (λ→|λ|) 58.76 / 58.05; size dropped (λ→λ/|λ|) 53.23 / 56.32; λ=1 53.28 / 55.39; λ=0 (control only) 40.65 / 40.44; control dropped 0.25 / 0.20.
  Share of z12's squared size that is λ·z6: 6.7% / 8.3%. THE CONTROL DOMINATES THE LATENT.
  Angle of λ: median 3.6° / 4.4°, 90th pct 11.4° / 11.7°. Modes used by 50+ images: 238 / 292.
  Size/phase swap on shared modes (87 / 73 of 128): size A + phase B gives B's answer 27.4% / 21.5%, A's answer 0.2% / 0.4%; A or B alone on shared modes keeps its answer 87.6% / 84.8%.
 02:08 EST: launched `share` (cls_train.py --name=share --arms=real_w01,real_w03,real_w1, 30,000 steps, PID 3349485, log night/share.log): real-start KAE + penalty weight x control's share of z12's squared size. wandb 1j7duyt9, ngi5gija, twx9r064. Report rebuilt with cls finals + angle tests.
 03:30 EST: share step 5000 (acc / control share): w01 56.40 / 0.092; w03 56.18 / 0.024; w1 55.18 / 0.006 (cls_add_real at 5000: 56.06, share ~0.92 at end). Penalty nearly free so far; must check with the ablation at the end that the share is not gamed by scale.

 09:20 EST 11 Nov: angle_tests on share checkpoints at step 22,500 (CPU; full log night/angle_share_22500.log). w01 / w03 / w1:
 as trained 59.01 / 59.04 / 59.13; angle dropped 17.78 / 9.18 / 8.73; size dropped 51.78 / 44.76 / 39.73; λ=1 46.43 / 35.54 / 27.58;
 λ=0 (control only) 27.17 / 23.08 / 32.45; control dropped 0.54 / 1.20 / 1.02. λ·z6 is ~100% of z12's squared size.
 Swap (size A + phase B): B's answer 10.3 / 14.1 / 20.8%, A's 1.8 / 3.5 / 4.0%. Own answer on shared modes 94.1 / 90.2 / 89.4%.
 Reading: eigenvalues now matter (were near-decoration), control alone gets less, but control is still required (expected: block-6 CLS has no answer).
 Angle dropped is worse than λ=0, so part of the drop is the decoder breaking on inputs it never saw, not lost information.

 10:27 EST 11 Nov: share run FINISHED (30,000 steps). acc / agreement / FVU / control share: w01 59.10 / 72.75 / 0.277 / 0.019; w03 59.44 / 72.82 / 0.279 / 0.010; w1 59.34 / 72.40 / 0.281 / 0.002. (cls_add_real 58.94 / 72.24 / 0.282.) Saved checkpoint = best: w01 is step 25,000 (59.14), w03 and w1 are step 30,000.
 10:40 EST: angle_tests on share finals in night/angle_share_final.log. w01/w03/w1: angle dropped 13.89/3.42/4.66; size dropped 50.41/38.46/33.96; λ=1 44.19/28.34/23.53; λ=0 26.11/20.45/31.17; control dropped 0.48/1.00/0.89; swap B's answer 10.0/13.1/20.3, A's 1.8/3.9/4.3.
 10:55 EST: where_control.py (new) on cls_add_real / share w01 / w03 / w1. Share of z12's squared size:
   λ·z6 first coords 8.28 / 99.72 / 99.58 / 99.22; λ·z6 second 0.03 / 1.02 / 0.96 / 0.48; control first 69.93 / 2.92 / 0.98 / 0.17; control second 5.84 / 0.09 / 0.04 / 0.04.
   Accuracy: as trained 58.94 / 59.14 / 59.44 / 59.34; λ·z6 second removed 58.07 / 14.05 / 3.47 / 4.71; control second removed 3.67 / 18.95 / 23.69 / 26.65;
   control first removed 11.58 / 40.29 / 40.15 / 42.10; all second removed 2.99 / 0.88 / 0.50 / 0.23.
 Reading: the decoder reads mostly the second coordinates (about 1% of size). Under real start b·x is a fixed multiple of a·x, so it carries no new information:
 the penalty moved the dependence, the size share was gamed by scale. Angle "mattering" in the share KAEs is not evidence of meaning.
