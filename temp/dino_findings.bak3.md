##### What we found

Everything is measured on ImageNet's 50,000 validation images unless a line says otherwise.

**Accuracy.** The KAE reaches 61.8% probe accuracy and gives DINOv3's answer on 66.5% of images. DINOv3 itself reaches 83.9%. Almost all of the 22-point gap opens at K: block 12 pushed through the encoder and decoder alone scores 81.0%, and 79.5% when it has to use the modes picked at block 6. K's prediction scores 61.8%. Giving every mode its true angle at block 12 (keeping K's size) lifts that to 74.2%, and giving it its true size (keeping K's angle) to 73.1%. These are ceilings, as they use block 12 itself.

**The spectrum.** 457 of the 1,025 modes are ever kept. They fall into four groups:
- 2 **growth** modes (331 and 77), on for every image, with `|λ|` of 18.9 and 14.4 and angles of 10° and 8°. They hold 78% of the latent energy at block 12. Removing either costs 7 to 10 points.
- 10 **offset** modes, on for every image with nearly the same value, so they carry the average state. They turn by 14° to 174° (median 82°) and hold 63% of the latent energy at block 6. The large angles are used: setting mode 338's angle to 0 at the same size costs 7.6 points.
- 4 **inert** modes that sit at `|λ|` ≈ 1 and do nothing.
- 441 **feature** modes, on for some images only. 305 of them have `|λ|` between 5 and 10 and turn by 5° to 45°. The other 136 have `|λ|` under 5 and hardly turn.

Unlike the CLIP runs, nearly every kept mode grows.

**Do modes follow their eigenvalue?** On average, yes: the measured gain tracks `|λ|` (correlation 0.97) and the measured turn is within 1° of the angle of `λ` for half the modes. Image by image, only partly: `λ·z₆` explains 58% of the variation of `z₁₂` for the median feature mode, and 16% for the modes that turn by under 5°. That unexplained part is where the 18 points at K go.

**Snow.** DINOv3 is not fooled by snow: 86.0% on the 129 snowy photos against 82.9% on normal images of the same classes. The KAE is: 34.1% against 67.7%, answering snowplow (35 images) and snowmobile (22). So here snow is a failure of the KAE, not of the model it stands in for.
- Snow is spread over several modes. The top ones fire on winter woods (667), on snow and white water (644), on open water (37) and, at the top of its range, on piles of fruit and balls (247).
- Removing the top 4 takes snowy accuracy to 57.4% (31 images fixed, 1 broken) and leaves the normal vehicles unchanged. Picking the modes on one half of the test and scoring the other gives 56% and 58%. It stays far below DINOv3's 86%.
- The cost: of the 25,815 validation images that keep one of the 4 modes, removal fixes 312 and breaks 574.
- Turning against removing: for mode 667 alone, a turn of +90° fixes 15 snowy images where removal fixes 6. With 4 or more modes, removal and a 180° turn tie near 58%, and the turns break more validation images.
- K is not wrong about the snow modes. Giving them their true block-12 value changes little (38.8%), while giving every other mode its true value recovers 69.8%. On snowy vehicles the snow modes really do grow by about `|λ|`, as they do on snowplows. So removing them deletes evidence that is truly there, to make up for the vehicle evidence K fails to produce. It is a patch, not a repair.

**Modes.** 95% of the modes kept on at least 0.5% of images have top images more alike than chance, and most group images across classes: round fruit and balls, shaggy dogs, birds against the sky, antelope herds, knitted fabric. Modes drawn at random look much the same. Removing a mode nearly always hurts: summed over modes, removal fixes 51,909 answers and breaks 97,332. The modes that matter most are dog modes.

**Size against turn.** Over 40 feature modes, turning a mode by 45° moves DINOv3's feature in a direction that is mostly new (cosine -0.36 with the move made by scaling), while a turn of 180° acts like scaling down (-0.93). So a turn reaches outputs that a change of size cannot, but on this model no turn improves accuracy on the images that keep the mode.

**Limits.** The snow test has 129 images, so one image is 0.8 points. The snow modes and the best turns are picked on the images they are scored on, except in the two held-out checks. Passing the coherence bar means "more alike than chance", not "interpretable". The model was still improving when training stopped at 12 epochs.
