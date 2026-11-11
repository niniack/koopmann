##### What we found

Everything is measured on ImageNet's 50,000 validation images unless a line says otherwise.

**Accuracy.** The KAE reaches 61.8% probe accuracy and gives DINOv3's answer on 66.5% of images. DINOv3 itself reaches 83.9%. Almost all of the 22-point gap opens at K: block 12 pushed through the encoder and decoder alone scores 81.0%, and 79.5% when it has to use the modes picked at block 6. K's prediction scores 61.8%. Giving every mode its true angle at block 12 (keeping K's size) lifts that to 74.2%, and giving it its true size (keeping K's angle) to 73.1%. These are ceilings, as they use block 12 itself.

**The spectrum.** 457 of the 1,025 modes are ever kept. They fall into four groups:
- 2 **growth** modes (331 and 77), on for every image, with `|λ|` of 18.9 and 14.4 and angles of 10° and 8°. They hold 78% of the latent energy at block 12. Removing either costs 7 to 10 points.
- 10 **offset** modes, on for every image with nearly the same value, so they carry the average state. They turn by 14° to 174° (median 82°) and hold 63% of the latent energy at block 6. The large angles are used: setting mode 338's angle to 0 at the same size costs 7.6 points.
- 4 **inert** modes that sit at `|λ|` ≈ 1 and do nothing.
- 441 **feature** modes, on for some images only. 305 of them have `|λ|` between 5 and 10 and turn by 5° to 45°. The other 136 have `|λ|` under 5 and hardly turn.

Unlike the CLIP runs, nearly every kept mode grows.

**The modes every image keeps.** 16 of each image's 128 modes are the same 16 for every image.
- The 10 offset modes carry the average state, and their large angles come from the shared standardization. With one mean and std for both blocks, the average block-6 state and the average block-12 state are opposite (cosine -1.00) and equally long, while block-12 states are spread 10.7 times wider around their average than block-6 states. So K has to flip the average at the same length and grow the rest about tenfold. The offset modes do the first job: they are among the slowest-growing kept modes (`|λ|` of 0.6 to 3.5, against a median of 7.6 for feature modes) and hold 9 of the 10 largest angles in the spectrum. Mode 861 is close to a multiplication by -1.2. Giving them the same value for every image costs 0.5 points. Setting their angles to 0 costs 9.7 points, more than removing them (5.1).
- The 2 growth modes are the two fastest-growing modes, with ordinary turns (10° and 8°, under the feature median of 12°). Each is a point on a plane that every image has, and the direction it points in says what type of image it is. For mode 331, going round the plane: small objects on plain backgrounds, a single subject against an empty field (a shark, a fruit), close-ups that fill the frame (pets, food), reefs, animals in vegetation, outdoor machines and gatherings, street vehicles, office furniture and electronics. The direction matters more than the size: giving every image mode 331's average size costs 1.3 points, giving it the average direction costs 8.8.
- The 4 inert modes do nothing: no edit of them changes any answer.

**Do modes follow their eigenvalue?** On average, yes: the measured gain tracks `|λ|` (correlation 0.97) and the measured turn is within 1° of the angle of `λ` for half the modes. Image by image, only partly: `λ·z₆` explains 58% of the variation of `z₁₂` for the median feature mode, and 16% for the modes that turn by under 5°. That unexplained part is where the 18 points at K go.

**Snow.** DINOv3 is not fooled by snow: 86.0% on the 129 snowy photos against 82.9% on normal images of the same classes. The KAE is: 34.1% against 67.7%, answering snowplow (35 images) and snowmobile (22). So here snow is a failure of the KAE, not of the model it stands in for.
- Snow is spread over several modes. The top ones fire on winter woods (667), on snow and white water (644), on open water (37) and, at the top of its range, on piles of fruit and balls (247).
- Removing the top 4 takes snowy accuracy to 57.4% (31 images fixed, 1 broken) and leaves the normal vehicles unchanged. Picking the modes on one half of the test and scoring the other gives 56% and 58%. It stays far below DINOv3's 86%.
- The cost: of the 25,815 validation images that keep one of the 4 modes, removal fixes 312 and breaks 574.
- Turning against removing: for mode 667 alone, a turn of +90° fixes 15 snowy images where removal fixes 6. With 4 or more modes, removal and a 180° turn tie near 58%, and the turns break more validation images.
- K is not wrong about the snow modes. Giving them their true block-12 value changes little (38.8%), while giving every other mode its true value recovers 69.8%. On snowy vehicles the snow modes really do grow by about `|λ|`, as they do on snowplows. So removing them deletes evidence that is truly there, to make up for the vehicle evidence K fails to produce. It is a patch, not a repair.

**DINOv3's own mistakes.** DINOv3 is wrong on 8,032 validation images. For 80% of them the label is among its top 5 answers, and the most common mix-ups are look-alike or overlapping labels: tights and one-piece bathing suit, laptop and notebook computer, collie and Shetland Sheepdog. The KAE inherits them: on these images it is right 12.9% of the time, against 71.1% on the rest.
- Some modes flag these images beyond what their classes explain: 28 modes have a score above 3.7 where chance gives none, and the scores repeat across two halves of the images (correlation 0.67).
- Editing the flagged modes does not fix the mistakes. Removing one breaks at least about as many answers as it fixes, through K and on the true block 12, and turns break more. Picking the modes to remove on one half of the images and scoring the other half gives no gain.
- So on ImageNet itself the modes can say where DINOv3 is likely wrong, and cannot repair it. The weak spots the DINOv3 paper reports for this model lie outside ImageNet's photos: drawings and other renditions (76.7% on ImageNet-R) and ObjectNet (64.1%).

**Modes.** 95% of the modes kept on at least 0.5% of images have top images more alike than chance, and most group images across classes: round fruit and balls, shaggy dogs, birds against the sky, antelope herds, knitted fabric. Modes drawn at random look much the same. Removing a mode nearly always hurts: summed over modes, removal fixes 51,909 answers and breaks 97,332. The modes that matter most are dog modes.

**Size against turn.** Over 40 feature modes, turning a mode by 45° moves DINOv3's feature in a direction that is mostly new (cosine -0.36 with the move made by scaling), while a turn of 180° acts like scaling down (-0.93). So a turn reaches outputs that a change of size cannot, but on this model no turn improves accuracy on the images that keep the mode.

**Limits.** The snow test has 129 images, so one image is 0.8 points. The snow modes and the best turns are picked on the images they are scored on, except in the two held-out checks. Passing the coherence bar means "more alike than chance", not "interpretable". The model was still improving when training stopped at 12 epochs.
