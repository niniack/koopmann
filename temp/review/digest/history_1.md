# History digest 1: 6 and 7 November 2026

This is the record of the first two days: what was tried, what came out, and whether it still holds.
It is written from `review/history/talk_11-06.txt` and `talk_11-07.txt` (both read in full), with numbers checked
against the training logs in the scratchpad where a log exists.

**How to read it**

- Times are EST. Entries are in time order. A source written as `full_11-06.txt line N` means the entry headed
  `[11-06 HH:MM | line N]` in that file (`grep -n "line N\]"` jumps to it).
- **KAE** = Koopman autoencoder: an encoder, one linear step K in a wider "latent" space, and a decoder.
- **FVU** = fraction of variance unexplained. 0 is a perfect prediction, 1 is no better than predicting the average.
  "One-step FVU" is measured on the decoded state. "Latent FVU" is measured inside the latent space.
  "Rollout FVU" starts at block 1 and applies K again and again.
- **Mode** = one eigen-direction of K with its eigenvalue λ. **TopK** = each image may keep only its k strongest latent entries.
- **Jordan form** = K is built as one real eigenvalue plus 2×2 blocks `[[a, −b], [b, a]]`, so every block is one mode with eigenvalue a ± ib.
- Goals: **speed** (a faster stand-in for CLIP), **meaning** (eigenvalues and modes should mean something),
  **adaptation** (fix a domain by tuning a few eigenvalues), **groundwork** (anything needed before those).

**Things a reader must know before reusing anything**

- The notebooks `06` to `24` that most entries mention were deleted on 11-07 23:13 at the user's request
  ("remove all the bullshit notebooks. condense down to one"). Only the condensed one survives:
  `/home/nsa325/work/koopmann/second_analysis/archive/jordan_topk_modes.ipynb`.
- Checkpoints named below are in `/home/nsa325/work/koopmann_data/weights_cache/tokae/` as `<name>.safetensors` plus `<name>_stats.pt` (checked with ls).
- File times shown by `ls` in the scratchpad are EST + 9 hours.
- Accuracy was judged with a **fitted linear classifier** until 11-07 15:24. From then on it is CLIP's own zero-shot head. Every entry says which one it used. Numbers from the fitted classifier are not comparable with later ones.
- Photo files in the scratchpad from this period (`gallery_*.jpg`, `mode_images/`, `sparse_images/`, `feature_images/`, `mode_views/`, `final_images/`, `survey_*.png`) were **not checked against the no-bug rule** (one gallery showed a "monarch butterfly" mode). Do not show them without filtering.
- Most wandb runs of 11-06 were deleted on the user's orders that night (24 runs cleared on 11-07 00:14). Their ids are given only as a record.

---

# 6 November

## 11-06 13:25 Starting point: the Class Vectors paper
- Question: what does "Class Vectors" (Kim and Do, arXiv 2510.11268) do, and is classifier editing a use case for the KAE?
- Setup: reading only. Repo was on branch `2026`; the user's unfinished work sat in `stash@{0}`.
- Result: a class vector is the mean feature of the fine-tuned model minus the mean feature of the pretrained model. Forgetting uses −1.5 times that vector. Reported gains: Snowy ImageNet 55.2 to 72.7 (ViT-B/16), typographic attack 48.9 to 98.9 (`full_11-06.txt line 265`). Deep Repurposing (arXiv 2610.01453) forgotten-class accuracy on CIFAR-10 / CIFAR-100 / Flowers102 / ImageNet: base 0.90 / 0.78 / 0.82 / 0.68, Class Vectors 0.17 / 0.66 / 0.76 / 0.66, Deep Repurposing 0.00 on all four (`full_11-06.txt line 460`).
- Conclusion at the time: editing a classifier is a plausible venue; the open point was what a KAE adds.
- Status now: stands as background. Only the snow part of that paper was kept (see 16:13).
- Goal: groundwork
- User said: (14:25) "Make a new branch . Try out your ideas . If you end up training please use muon optimizer and please log to wandb. Any new code should be written in my style of code It should be very simple"
- Files worth reusing: none.

## 11-06 14:43 Mistake: the assistant wrote its own Muon optimizer
- Question: none; this is a correction.
- Setup: 12 runs had been launched with a hand-written Muon.
- Result: replaced with `torch.optim.Muon`; torch index moved from cu121 to cu130, torch `>=2.9`; the 12 runs relaunched in wandb project `new_koopmann` (`full_11-06.txt line 628, 632, 710`).
- Conclusion at the time: use library code; do not reimplement.
- Status now: stands. Every later run uses `torch.optim.Muon` on 2D weights and AdamW on biases and vectors.
- Goal: groundwork
- User said: "Why the hell did you implement muon? https://docs.pytorch.org/docs/2.15/generated/torch.optim.Muon.html". On cleaning wandb afterwards: "No, don't mess with old stuff. Can you tell me any results we have from experiments?"
- Files worth reusing: none.

## 11-06 14:57 Experiment 1: deleting a class (Yin-Yang, then MNIST)
- Question: can editing K delete a class better than simple baselines?
- Setup: small classifiers on Yin-Yang and MNIST. Methods: retraining, masking the class's logit, Class Vectors, EMMET (a closed-form weight edit) on the host model's last layer, on a plain linear map, and on the KAE's K. Scores: accuracy on the forgotten class, accuracy on the kept classes, and two leak tests (`conf_auc`, `probe_auc`: how well the forgotten class can still be detected; lower is better).
- Result, Yin-Yang, forgotten / kept / conf_auc / probe_auc, mean of 9 edits (`full_11-06.txt line 798`): original 0.985 / 0.985 / 0.609 / 0.999; retrained 0.000 / 0.994 / 0.599 / 0.932; logit mask 0.000 / 0.992 / 0.754 / 0.999; Class Vectors 0.110 / 0.969 / 0.673 / 0.997; EMMET on host 0.027 / 0.895 / 0.595 / 0.929; EMMET on linear map 0.000 / 0.731 / 0.591 / 0.878; **EMMET on KAE 0.091 / 0.899 / 0.673 / 0.970**. Unedited KAE keeps 0.977 accuracy, linear map 0.807.
- Result, MNIST, forgotten / kept (`full_11-06.txt line 914`): logit mask 0.000 / 0.986; EMMET on host 0.000 / 0.982; EMMET on KAE 0.000 / 0.976; EMMET on linear map 0.000 / 0.918; Class Vectors 0.070 / 0.934.
- Conclusion at the time: masking one logit deletes a class for free, and EMMET on K only ties EMMET on the host layer. Class deletion gives the KAE no advantage.
- Status now: stands; class deletion was dropped as a use case and never revisited.
- Goal: adaptation (closed)
- User said: "None of this seems to be an attempt to extend my framework though, right? You seem to be running some baselines to get context?"
- Files worth reusing: none (code was in uncommitted `scripts/edit_kae/`, which no longer exists).

## 11-06 15:21 Experiment 2: predicting fine-tuned features
- Question: can a KAE predict how features move under fine-tuning better than a class vector or a linear map?
- Setup: R² on held-out fine-tuned features, for a model pretrained on FashionMNIST and for a randomly initialised one.
- Result (`full_11-06.txt line 914`), pretrained / random init: class vector 0.148 / −1.07; linear map 0.762 / 0.652; **KAE 0.962 / 0.915**.
- Conclusion at the time: the KAE fits the shift far better, but no use was built on it.
- Status now: never checked again.
- Goal: adaptation
- User said: (15:29) "Fucking chaos. Let's discuss a meaningful direction to take before running more bullshit, half-baked experiment"
- Files worth reusing: none.

## 11-06 15:04 Decision: stop running, discuss the direction
- Question: what problem should the KAE be given?
- Setup: discussion only.
- Result: no numbers.
- Conclusion at the time: the benefit of a KAE is the linear step in the middle; the search is for a problem where that matters.
- Status now: stands as the framing of the whole project.
- Goal: groundwork
- User said: "We need to be methodical and not burn time chasing down random stuff. I still don't get where class vectors fails, what others do to improve it, and where our benefits might sit". Then: "We are inherently looking for a problem given a solution (how to apply KAE). This is our flaw but we must live with this. I'd like to give the KAE a meaningful problem to work on. Can task repurposing fit our framework? ... the goal is to give KAE a good reason to exist. Our main benefit with a KAE structure is the linear predictor in the middle, which allow some basic steering with LQR, some eigendcomposition stuff, etc". Also: "Stop fucking looking at other work running on this machine or in other repos. It's irrelevant to you."
- Files worth reusing: none.

## 11-06 15:41 What the three papers do (reading)
- Question: what problems do Class Vectors, Deep Repurposing and Santurkar et al. (arXiv 2112.01008) each solve?
- Setup: reading the papers and appendices.
- Result: Class Vectors on snow uses `κ(snow+object) − κ(object)` at scale −1.0; ViT-B/16 original 55.2, latent edit 69.7, weight-mapped 72.7, DirMatch 72.0 (`full_11-06.txt line 998`). It edits the encoder's final output, gated by cosine above 0.5; the mapped version trains `resblocks.11.mlp` and `ln_post`, 4.7M parameters (`line 1078`). Deep Repurposing versus masking a logit (its Table 14, CIFAR-10): kept accuracy 0.91 vs 0.91; obsolete class still predicted 0.00 vs 0.00; raw probability of the obsolete class 0.88 vs 0.00; cost free vs 1.16 s (`line 1047`). Santurkar's VGG16 edit costs 92.27% to 91.05% on normal images (`line 1134`).
- Conclusion at the time: snow is a different kind of problem from deletion. The class is still valid and the model is drawn to a wrong class, so masking cannot help. That is where an edit of the representation is needed.
- Status now: stands; this is why snow became the adaptation test.
- Goal: adaptation
- User said: "So, i don't care about any of the adversarial/backdoor stuff int he class vectors paper. I do care about adaptation and class unlearning. I'm still not sure what exactly the deep repurposing paper does. However, it seems to all point towards modifying the latent representations to change how the model treats the class right?"
- Mistake inside this entry: a keyword search missed the appendix that compares Deep Repurposing with logit suppression; it was found on the second pass (`line 1012 to 1047`).
- Files worth reusing: paper folders `/home/nsa325/work/koopmann/temp/santurkar/` and `.../scratchpad/editing_classifiers/`.

## 11-06 16:56 CLIP's snow failure reproduced, no training
- Question: do we get the paper's numbers on the snowy vehicle photos, and how does CLIP fail on them?
- Setup: the snow test set from `MadryLab/EditingClassifiers` (`data/snow/test/`): **129 images, not 140** (school bus 22, motor scooter 21, racing car 20, fire truck 20, car wheel 20, army tank 17, traffic light 9; no license stated). Zero-shot CLIP, 1000 ImageNet classes, 80 prompt templates, three encoders.
- Result (`full_11-06.txt line 1310`), class-mean accuracy ± std: ViT-B/32 **53.4 ± 29.9**; ViT-B/16 55.2 ± 24.6; ViT-L/14 60.2 ± 22.5. These equal the paper's "Pretrained" row. If the answer is limited to the 7 true classes: 97.6, 98.4, 100.0. Per class (B/32, B/16, L/14): racing car 10.0, 30.0, 20.0; motor scooter 19.0, 23.8, 57.1; car wheel 45.0, 55.0, 55.0; traffic light 55.6, 44.4, 55.6; army tank 58.8, 52.9, 52.9; fire truck 90.0, 85.0, 90.0; school bus 95.5, 95.5, 90.9. Wrong answers: 60 / 56 / 50 images, of which "snowplow" or "snowmobile" 47 / 30 / 32.
- Conclusion at the time: the failure is real and is mostly two snow classes stealing the answer.
- Status now: stands. The 53.4 (class mean) and the later 53.5% (image mean, `full_11-07.txt line 5263`) for real CLIP on these photos are the reference for all later snow work.
- Goal: adaptation
- User said: "so surely we can very quickly reproduce the numbers right? can you please do so in a notebook? get the snowy dataset and show me, for all 129 images, how it fails. use all three encoders. don't train anything"
- Mistakes fixed here: OpenAI weights need QuickGELU (the warning was hidden by an output filter); class 670 is named `vespa` in CLIP's list; a quoting error.
- Files worth reusing: data `/scratch/nsa325/datasets/snowy_imagenet/` (7 class folders); `/home/nsa325/work/koopmann/temp/snow_probe.py`. Notebook `06_snowy_clip.ipynb` deleted.

## 11-06 17:07 Ceiling: how far could a snow fix go?
- Question: if every obvious snow mistake were fixed by hand, what accuracy would result?
- Setup: same 129 photos; replace wrong "snowplow"/"snowmobile" answers by the next answer, then do the same for all snow-themed classes.
- Result (`full_11-06.txt line 1354`), class means: B/32 53.4, then **88.6**, then **90.2** (Class Vectors on the same model: latent 72.2, mapped 76.2, DirMatch 73.9). B/16 55.2, 78.2, 88.4 (69.7, 72.7, 72.0). L/14 60.2, 84.7, 85.4 (71.3, 78.3, 74.6). 12 / 13 / 17 images stay wrong for other reasons.
- Conclusion at the time: there is room up to about 90%; the published method reaches about 72 to 78.
- Status now: stands as the upper reference. No KAE edit has come near it (best later: the KAE's own snow failure repaired to 53.5%, real CLIP's level; memory note `project-angle-meaning-search.md`).
- Goal: adaptation
- User said: "i assume that if we corrected all the obvious snow mistakes just by hand, we would arrive at their results? i suspect that we can't recover 100% accuracy because not all the mistakes are snow related"
- Files worth reusing: none beyond the data.

## 11-06 17:13 The user's design idea: modes that carry "snow"
- Question: (design direction, no experiment)
- Setup: none.
- Result: none.
- Conclusion at the time: the assistant raised three risks: features must be made to sit in modes; the edit should act across depth; a baseline without a KAE is needed.
- Status now: this is still the project's central idea. The "nudge each intermediate step" part was dropped on 11-07 00:06.
- Goal: meaning, adaptation
- User said: "I think of each sample as traveling through some high dimensional path to arrive at its class. In that perspective, there is something about its initial/early representation that draws it towards the wrong class. A fix would be to improve its early layer representation (fix the embedding) or we could slightly nudge each intermediate one, like they do in llm steering. the KAE has the benefit that perhaps we can map these representations to some latent space. then each representation can be seen as an evolution of many modes. Hopefully some of those modes carry "snow" as a feature. if we dampen or excite those eigenvalues then maybe that representation will travel differently and arrive at a differen toutput." And at 17:26: "I was thinking we could use sparse koopman autoencoders to do that. The idea would be kind of similar to SAEs. We would project into a wide space and hopefully select a small handful modes to evolve. ... honestly, i need to justify and show kaes can matter."
- Files worth reusing: none.

## 11-06 17:31 Baseline with no KAE: remove one "snow" direction from CLIP's final embedding
- Question: how much of the snow problem does the simplest possible edit fix, and what does it break?
- Setup: direction = mean embedding of 5 snowy minus 5 clean photos per class; project it out of CLIP's final image embedding. Mean of 20 random draws. Also tested on real snowmobile and snowplow photos and on random ImageNet images.
- Result (`full_11-06.txt line 1427`), snowy before, after; real snowmobile; real snowplow; random ImageNet: B/32 52.9, **73.2**; 87.2 to 51.7; 91.3 to 46.0; 65.2 to 64.7. B/16 55.9, 76.0; 100 to 81.3; 89.1 to 51.4; 68.3 to 68.4. L/14 60.6, 74.2; 100 to 93.1; 91.3 to 74.6; 75.4 to 75.4. Removing 2 to 7 directions does not help further.
- Conclusion at the time: one direction, no training, matches Class Vectors. The price is 17 to 45 points on the classes where snow is real evidence. The Class Vectors paper does not report that price; Santurkar and Deep Repurposing do report damage (`line 1447`).
- Status now: stands, and it is the baseline any KAE snow edit must beat. Never rerun against the later KAE edits on the same footing.
- Goal: adaptation
- User said: "You're welcome to try the non-KAE baseline, if its easy"
- Files worth reusing: none (notebook `07_snow_direction.ipynb` deleted).

## 11-06 18:01 Decision: train a KAE on CLIP's CLS token
- Question: can a KAE learn how the CLS token changes from block to block?
- Setup: decision only.
- Result: none.
- Conclusion at the time: start small (Imagenette), wandb project `tokae`.
- Status now: `tokae` is still the wandb project; `/home/nsa325/work/koopmann/scripts/tokae/train.py` is still the trainer.
- Goal: groundwork
- User said: "okay so we need to now train a kae on the clip encoder's cls token. how does that sound? we can try something very basic (encoder, decoder, linear operator). make the wandb project name "tokae". for now maybe lets train on something small like imagenette? i think that's a limitation of my approach is not sure how many samples i have to train on to generalize 😕 anyways, i kind of just want to see if we can even learn the dynamics of the cls class in a kae with this approach. we shoul dhave a reconstruction loss and a one step latent prediction loss ... then we can test on the validation set of imagenette to see how close it is in predicting all the intermediate steps". Also: "please use muon"; "the validation and training logs have to show for each transition man, otherwise hard to tell where training is going wrong and which transitions are hard to learn."
- Files worth reusing: `/home/nsa325/work/koopmann/scripts/tokae/train.py` (much changed since).

## 11-06 18:15 First KAE: one K for all 11 block-to-block steps
- Question: does one shared K predict the CLS token from each block to the next?
- Setup: CLIP ViT-B/32, CLS token after blocks 1 to 12 (11 steps). Imagenette train, 9,469 images; validation 3,925. Encoder 768 to 1024 to 1024, mirrored decoder, leaky ReLU. One full 1024×1024 K, started at the identity. Losses: reconstruction + one-step latent prediction. 200 epochs, batch 256, Muon lr 0.005 (4.7M parameters), AdamW lr 0.001, linear decay. One standardization for all layers. wandb `pzz09i9f` (deleted).
- Result (`full_11-06.txt line 1601, 1614`), validation: reconstruction FVU 0.014; latent 0.348; one-step **0.418**; rollout **1.554**. Reference models in closed form: one shared linear map 0.483 one-step, a separate linear map per step 0.196, shared-map rollout about 1.47. Per-step one-step FVU: 1→2 0.596, 2→3 0.521, 3→4 0.427, 4→5 0.386, 5→6 0.377, 6→7 0.459, 7→8 0.445, 8→9 0.397, 9→10 0.408, 10→11 0.351, 11→12 0.236. Rollout by step rises from 0.596 to 2.459. The spread across images grows about 400 times from block 1 to block 12 (0.01 to 4.2).
- Conclusion at the time: train and validation agree (0.407 vs 0.418), so it is not overfitting. One K is too weak for 11 different steps.
- Status now: superseded. The "rollout" target was abandoned on 11-07 00:06 when a closed-form check showed block 1 holds too little to predict block 12 at all.
- Goal: groundwork
- User said: "holy shit rollout and one step fvu looks terrible! is it overfitting?"
- Files worth reusing: none.

## 11-06 18:22 Standardize each layer separately
- Question: does the huge growth in scale across layers hurt?
- Setup: same model, each block standardized on its own. wandb `jhw2tnnm` (deleted); checkpoint `kae_clip_cls_ViT-B-32_layer_norm_seed_0` exists.
- Result (`full_11-06.txt line 1678`), validation one-step / rollout: no change (persistence) 0.811 / 1.745; shared linear map 0.398 / 0.847; **KAE 0.377 / 0.831**; separate map per step 0.212 / 0.719.
- Conclusion at the time: the KAE beats a shared linear map by only about 5%, and one K is 1.8 times worse than separate maps. Good reconstruction is trivial because the latent (1024) is wider than the state (768).
- Status now: superseded by the block-6-to-block-12 setup.
- Goal: groundwork
- User said: "please standardize and run again. btw are your results fvu?" and "what is the size of the latent? is it bigger? that would explain the fact that reconstruction is good"
- Files worth reusing: `/home/nsa325/work/koopmann_data/weights_cache/tokae/kae_clip_cls_ViT-B-32_layer_norm_seed_0.safetensors`.

## 11-06 18:30 Shared modes with a different eigenvalue per layer (real and complex)
- Question: can all layers share one set of modes, with only the eigenvalues changing from layer to layer?
- Setup: K diagonal in a learned basis; eigenvalues per layer; a real version (`kc9fs3dr`) and a complex version (`2x1vk7t9`), both deleted.
- Result (`full_11-06.txt line 1736`), validation latent / one-step / rollout: real 0.568 / 0.715 / 1.126; complex 0.536 / 0.705 / 1.132; shared full K 0.351 / 0.377 / 0.831. Both stalled at about 0.71 from epoch 20. Closed-form check (`line 1758`), one-step / rollout: random basis + per-layer eigenvalues 0.628 / 0.948; PCA basis 0.589 / 0.927; eigenbasis of the shared map 0.390 / 0.844. Only 0.1 to 0.4% of each layer map's energy lies on the diagonal in a shared basis; condition number 563.
- Conclusion at the time: the layer maps do not share an eigenbasis. The trained complex model (0.705) was worse than a random basis (0.628), so the training also failed.
- Status now: stands as a negative result. Per-layer eigenvalues were not tried again; later models use one step only.
- Goal: meaning
- User said: "lets try shared modes"
- Files worth reusing: none.

## 11-06 18:40 Does snow sit in a few modes of K? No
- Question: is the difference between snowy and clean photos carried by a few eigenmodes of the trained K?
- Setup: model `jhw2tnnm` (full shared K). Compare how much of the snowy-minus-clean difference is explained by the top 2 / 10 / 50 / 200 directions of three kinds: K's eigenmodes, PCA of the latent, random.
- Result (`full_11-06.txt line 1821`), share explained at block 12: K modes −1.09 / −10.7 / −33.3 / −20.5; PCA 0.21 / 0.50 / 0.81 / 0.95; random 0.02 / 0.08 / 0.28 / 0.64. Snow detection AUC from the top 2 directions at block 12: K modes 0.70, PCA 0.99, random 0.94, full direction 1.00. K's eigenvector condition number is about 19,000; median size of λ 0.47. Follow-up with K's singular directions (`line 1875`): top 10 explain 0.08 at every block (random 0.07 to 0.09), PCA of the latent 0.94 / 0.80 / 0.42 / 0.50 at blocks 3 / 6 / 9 / 12.
- Conclusion at the time: the eigenmodes of an unconstrained K are almost parallel and cancel each other, so they are useless as features. Singular directions behave like random ones.
- Status now: stands for a full, unconstrained K. It is the reason for the later Jordan form and TopK. One caveat was added at 18:44: the snowy photos come from Flickr and the clean ones from ImageNet, so the "snow direction" may partly be a source difference.
- Goal: meaning
- User said: "yes lets check, but lets please remove any bullshit runs/code"; "I didn't quite understand these results"; "why do you say "snow is in the latent space"?"; "what about the principal directions? if you say that the eigenmodes are too similar, why not use the SVD?"
- Files worth reusing: none (notebook `08_snow_modes.ipynb` deleted).

## 11-06 18:58 Decision: stay on Imagenette, add a state loss, then sparsity, look for class modes
- Question: (decision)
- Setup: none.
- Result: none.
- Conclusion at the time: snow is set aside until the model is better.
- Status now: superseded; snow came back on 11-07 13:08 on ImageNet.
- Goal: groundwork
- User said: "Let's stick to imagenette for now and not chase the snow idea so much. i think we should try adding state loss as well in addition and see if it improves. then lets add sparsity too. We should try and see if we can get some modes that are class relevant perhaps?"
- Files worth reusing: none.

## 11-06 19:05 One-step state loss
- Question: does a loss on the decoded next state improve prediction?
- Setup: same 12-block model plus a loss `decode(K φ(x_l))` against `x_(l+1)`. Checkpoint `kae_clip_cls_ViT-B-32_state_1.0_dim_1024_k_0_seed_0`.
- Result (`full_11-06.txt line 1961`), validation without / with: one-step 0.377 / **0.282**; latent 0.351 / 0.702; rollout 0.831 / **12.6** (45 at the last layer); reconstruction 0.000 / 0.007.
- Conclusion at the time: one-step is 29% better than a shared linear map, but the decoder does the work and repeated application explodes. The loss was one-step only, which is why rollout did not improve.
- Status now: the state loss was kept in every later model. The rollout problem went away when the model became one step.
- Goal: speed
- User said: "i would have thought that the state loss would help rollout, i'm so confused"
- Files worth reusing: `/home/nsa325/work/koopmann_data/weights_cache/tokae/kae_clip_cls_ViT-B-32_state_1.0_dim_1024_k_0_seed_0.safetensors`.

## 11-06 19:14 First TopK: keep 32 of 1024, with state loss
- Question: does sparsity give class-specific latents or modes?
- Setup: as above with TopK 32 of 1024. wandb `p8cvfijh` (deleted); checkpoint `kae_clip_cls_ViT-B-32_state_1.0_dim_1024_k_32_seed_0`.
- Result (`full_11-06.txt line 1975`), train / validation: reconstruction 0.252 / 0.306; latent 0.343 / 0.437; one-step 0.288 / 0.386; rollout 4.35 / 6.37.
- Conclusion at the time: sparsity costs reconstruction and prediction.
- Status now: superseded by later TopK models.
- Goal: meaning
- User said: "just run it now man"
- Files worth reusing: the checkpoint named above.

## 11-06 19:20 Class tests on the dense and the TopK-32 model
- Question: do K's modes, or the sparse latents, pick out Imagenette's 10 classes?
- Setup: for each class, the best single direction's detection AUC (mean over 10 classes); then a stricter test: 10-way accuracy from only 5 / 10 / 20 directions; and "selectivity" of each direction.
- Result (`full_11-06.txt line 2006`), best single direction AUC, latents / K modes / PCA / random / full class direction. Dense, block 9: 0.812 / 0.882 / 0.814 / 0.806 / 0.965; block 12: 0.927 / 0.907 / 0.977 / 0.919 / 0.999. TopK 32, block 9: 0.912 / 0.922 / 0.897 / 0.872 / 0.977; block 12: 0.988 / 0.944 / 0.998 / 0.982 / 0.999. Stricter test (`line 2113`), 10-way accuracy from 5 / 10 / 20 directions, dense block 9: K modes .62 / .74 / .79, PCA .56 / .71 / .80; dense block 12: K modes .63 / .73 / .77, PCA .89 / .97 / .98. Selectivity of sparse latents 0.68 / 0.84 / 0.89 at blocks 6 / 9 / 12; of K modes 0.39 to 0.45.
- Conclusion at the time: K's modes are compact at mid-depth but each mixes classes. Sparse latents are class-specific, but a latent is a mode only if K is diagonal.
- Status now: the mid-depth advantage of K's modes was **withdrawn at 20:52** (it did not hold on `var25`: K modes 0.765 vs PCA 0.874 at block 9, `line 2643`). The point "sparse latents are modes only if K is diagonal" stands and led to the Jordan form.
- Goal: meaning
- User said: "hmm but doesn't sparse latents only select a few modes?"
- Files worth reusing: none (notebook `09_class_modes.ipynb` deleted).

## 11-06 19:44 Wider and less sparse: keep 1024 of 4096
- Question: does a wide latent with a mild TopK keep prediction and still separate classes?
- Setup: latent 4096, TopK 1024, full K, no state loss, wandb `q5k0d03g` (deleted); the version with state loss (`eu4jky0d`) was stopped by the user. Checkpoint `kae_clip_cls_ViT-B-32_state_0.0_dim_4096_k_1024_seed_0`.
- Result (`full_11-06.txt line 2188`), train / validation: reconstruction 0.000 / 0.001; latent 0.344 / 0.354; one-step 0.333 / 0.341; rollout 0.863 / 0.865. Share of latents that are distinct to a class at blocks 6 / 9 / 12 (`line 2240`): TopK 32 of 1024: 24.8 / 72.7 / 83.9%; TopK 1024 of 4096: 0.1 / 0.6 / 3.8%, with about 1,140 latents on for every image.
- Conclusion at the time: prediction is fine, but a quarter of the latent is a fixed always-on core, so there is no class separation. An earlier remark that this model had "397 near-pure latents" was corrected: they sit on that fixed core.
- Status now: stands as a negative result for mild TopK.
- Goal: meaning
- User said: "are you doing 32 on 4096? that's too constrained. I think we should still expand, maybe allow 1024 in 4096"; "is there a way to compute what percent of the latents are shared? and what percent are distinct? per class"
- Files worth reusing: the checkpoint named above.

## 11-06 19:54 First Jordan-form K
- Question: if K is block-diagonal, so that every kept latent pair is a mode, does it still predict?
- Setup: latent 4097 (1 real eigenvalue + 2,048 two-by-two blocks), TopK keeps 512 modes, no state loss, 12 blocks, shared K. wandb `b0qx57kg` (deleted); checkpoint `kae_clip_cls_ViT-B-32_jordan_0.0_dim_4097_k_1024_seed_0`.
- Result (`full_11-06.txt line 2260`), train / validation: latent 0.606 / 0.610; one-step 0.678 / 0.680; rollout 1.120 / 1.123. Median block angle 0.019 rad; median size of λ 0.87.
- Conclusion at the time: K learned roughly 0.87 times the identity. One block-diagonal K cannot serve 11 different steps.
- Status now: superseded. Jordan form worked once the model became a single step (11-07 01:57) and is the form of every model since.
- Goal: meaning
- User said: "okay please try out training diagonal K plus TopK at 1024 of 4096, without state loss. you should use block jordan form. 1 real eigenvalue, the rest should be 2d blocks"
- Files worth reusing: the checkpoint named above.

## 11-06 20:00 Decision: one clean setup, then a TopK sweep
- Question: (decision on the training setup and on wandb hygiene)
- Setup: dense K, latent 1024, reconstruction + one-step latent + multi-step latent losses, no state loss; then the same with TopK 32, 128, 256, 512. Runs `dense`, `k32`, `k128`, `k256`, `k512` (`k256` crashed at epoch 4, GPU out of memory, and has no checkpoint).
- Result: see the next entry; all of them collapsed.
- Conclusion at the time: config goes in wandb's config and tags, not in the run name.
- Status now: the naming and logging rules stand.
- Goal: groundwork
- User said: "our wandb is getting quite messy. I want to see one run. remove everything 1. no state loss, yes latent, yes latent multi step, yes reconstruction. 2. 1024 dim, dense K. 3. wandb should show the same stats it shows now, great job. 4. the tables suck ass and they need to go. 5. naming should be short and simple. 6. the details about the run should be logged (log the weights for each loss, log sparsity/k, you can also add tags,etc. don't use the name to do that bookkeeping). then we should run another run with 1024 dim but k=32, 128, 256, 512"
- Files worth reusing: checkpoints `dense`, `k32`, `k128`, `k512` in the weights cache.

## 11-06 20:09 Finding: the latent space collapses
- Question: why does the latent loss fall while nothing else improves?
- Setup: `dense`, `k32`, `k128`, `k512`. Measures: variance of the latent over training; effective rank (how many directions the latent really uses); share of variance in the top direction; a "noise test" (state FVU when the latent is replaced by noise of the same size).
- Result: latent variance of `dense` 1.22 at epoch 1, 0.19 (5), 0.012 (20), 0.0018 (50), 0.0009 (100), about 1,500 times smaller (`full_11-06.txt line 2327`). After dividing the latent loss by the variance (`line 2419`): `dense` latent FVU 0.015, reconstruction 0.021, one-step 0.458, rollout 0.822. Collapse table (`line 2514`), effective rank; share in top direction; state FVU on step 6→7 from the true latent / K's prediction / noise: `dense` 1.0; 99.6%; 0.008 / 0.499 / 6.96. `k512` 1.0; 99.5%; 0.033 / 0.543 / 4.98. `k128` 1.0; 99.5%; 0.203 / 0.545 / 3.69. `k32` 1.1; 98.9%; 0.366 / 0.596 / 3.39.
- Conclusion at the time: first the latent shrank in scale (fixed by the relative loss); then it collapsed onto one direction. A latent prediction loss alone rewards collapse. TopK size made no difference to this.
- Status now: stands. This was the main obstacle of the evening and drove the next ten runs.
- Goal: groundwork
- User said: "it seems that despite latent loss going down, our latent fvu doesn't go down. it feels like the latent space continuously shrinks!"; "is it cheating again? latent fvu is improving but nothing else seems to improve. if reconstruction is good and latent is good, you would naturally expect better rollout"; "you need to look at how koopman vae solved this problem please"; "i'm confused what you are suggesting. right now the problem is that our latent space clearly collapses. i want to fix that"
- Not done: adding noise to the encoder's output (Koopman-VAE style) was discussed and never run.
- Files worth reusing: paper folders `/home/nsa325/work/koopmann/temp/koopman_vae/` and `.../scratchpad/vicreg_papers/`.

## 11-06 20:37 Variance and covariance losses, weight 1 (`vicreg`)
- Question: does a VICReg-style penalty (keep each latent dimension's spread up, keep dimensions uncorrelated) stop the collapse?
- Setup: `dense` plus variance and covariance terms at weight 1. wandb `7ukiinul` (deleted); checkpoint `vicreg`.
- Result (`full_11-06.txt line 2569`): effective rank 30.4 pooled over layers, 3.3 to 9.2 inside one layer; latent FVU 0.034; one-step 0.407; rollout 0.818; noise test 2.27.
- Conclusion at the time: helps a little; still nearly collapsed inside each layer.
- Status now: superseded.
- Goal: groundwork
- User said: "sure lets try it out with dense"
- Files worth reusing: checkpoint `vicreg`.

## 11-06 20:49 Variance weight 25 (`var25`)
- Question: does a strong variance term fix it?
- Setup: variance weight 25. wandb `6gd0x2mp` (deleted); checkpoint `var25`.
- Result (`full_11-06.txt line 2612`): rank 984 pooled, 376 to 464 inside a layer; reconstruction 0.126; latent 0.642; one-step 0.410; rollout 0.786; noise test 0.64. Class tests on `var25` (`line 2643`): best single direction AUC, K modes / PCA: block 9 0.765 / 0.874; block 12 0.735 / 0.965.
- Conclusion at the time: collapse is gone, but prediction is no better (0.410) and the latent is now hard to predict (0.642). K's modes are not class-specific; the earlier mid-depth advantage was withdrawn.
- Status now: stands as a negative result. The "K condition" numbers reported with it (7,740 for `dense`, 2,270 for `var25`) are probably wrong: they were computed after a `torch.linalg.eig` call had silently overwritten K (found 11-07 01:15; not recomputed).
- Goal: groundwork
- User said: "raise the variance weight"; "isn't vicreg for like jepa wehere they want invariance between augmented versions? whats our invariance?"; later "var25 seems quite rough tbh"
- Files worth reusing: checkpoint `var25`.

## 11-06 21:00 Published weightings (`vic5`, `pldm`)
- Question: do the weightings other papers use do better?
- Setup: `vic5` = weights 5, 5, 5, 1 (`3m9ychlr`); `pldm` = 1, 1, 4, 6.9 (`8op48uev`). Both deleted from wandb; checkpoints exist.
- Result (`full_11-06.txt line 2707`): `vic5` rank 59, latent 0.060, one-step 0.454, rollout 0.825, noise 4.53. `pldm` rank 193 (392 / 316 / 45 / 7 at blocks 2 / 6 / 9 / 12), latent 0.497, one-step 0.430, rollout 0.794, noise 0.65.
- Conclusion at the time: neither beats `var25` on prediction; `pldm` still collapses in late layers.
- Status now: stands as a negative result.
- Goal: groundwork
- User said: "i feel like 25 was quite high, covariance and latent (i guess invariance?) didn't really improve too much. let's see what other people use and try that"
- Files worth reusing: checkpoints `vic5`, `pldm`.

## 11-06 21:19 Learning-rate sweep
- Question: is the learning rate too low?
- Setup: `var25` settings with Muon lr 0.005, 0.01, 0.02, 0.05. Checkpoints `lr0.01`, `lr0.02`, `lr0.05`.
- Result (`full_11-06.txt line 2741`), latent / one-step / rollout / reconstruction: 0.005: 0.642 / 0.410 / 0.786 / 0.126; 0.01: 0.622 / 0.405 / 0.782 / 0.135; 0.02: 0.599 / 0.405 / 0.780 / 0.152; 0.05: 0.334 / 0.475 / 0.817 / 0.293.
- Conclusion at the time: one-step is stuck near 0.40 whatever the rate; 0.05 is too high.
- Status now: stands. (A later sweep on ImageNet also found no effect of doubling the rate: memory note `project-pilot-run-results.md`.)
- Goal: groundwork
- User said: "maybe the learning rate is too low?"; "yeah lr sweep"
- Files worth reusing: none.

## 11-06 21:28 A separate K for every step (`layer_k`)
- Question: is the shared K the limit?
- Setup: 11 separate K matrices. wandb `dyn31f4u` (deleted); checkpoint `layer_k`.
- Result (`full_11-06.txt line 2804`), train / validation: one-step 0.237 / 0.295; rollout 0.527 / 0.909; latent 0.348 / 0.480. Validation rollout was 0.777 at epoch 20 and got worse.
- Conclusion at the time: better one-step, but it overfits on 9,469 images.
- Status now: stands; not tried again.
- Goal: groundwork
- User said: "lets stick to shared, i think it avoids overfitting. but we need to find ways to improve things. i think try var 10"
- Files worth reusing: checkpoint `layer_k`.

## 11-06 21:33 Variance weight 10 (`var10`)
- Question: is there a middle setting?
- Setup: variance weight 10. wandb `xg6ddqyq` (deleted); checkpoint `var10`.
- Result (`full_11-06.txt line 2839`), train / validation: reconstruction 0.120 / 0.135; latent 0.562 / 0.600; one-step 0.381 / 0.403; rollout 0.734 / 0.781; rank inside a layer 390 / 346 / 422 / 378; noise test 0.60.
- Conclusion at the time: about the same as `var25`.
- Status now: stands; variance terms were dropped altogether on 11-07 01:04.
- Goal: groundwork
- User said: "lower std target"; "actually what about this with sparsity, isn't that effectively lowering the latent?"; "additionally, shouldn't we have every layer be proportional in its variance?"
- Mistake: a hand check of `var10`'s variance per layer was wrong and was withdrawn (`line 2967`).
- Files worth reusing: checkpoint `var10`.

## 11-06 21:52 Orthogonal K (`ortho`)
- Question: does forcing K to be a pure rotation give well-separated modes at an acceptable cost?
- Setup: `var10` with K constrained to be orthogonal. wandb `ey7ovgjs` (deleted); checkpoint `ortho`.
- Result (`full_11-06.txt line 2967`), validation, `ortho` vs `var10`: reconstruction 0.192 vs 0.135; latent 0.632 vs 0.600; one-step 0.428 vs 0.403; rollout 0.850 vs 0.781; condition number 1.0. Latent variance grew with depth (0.50, 0.56, 0.93, 1.31, 1.78 at blocks 2, 3, 6, 9, 12).
- Conclusion at the time: worse on everything. A rotation cannot grow or shrink anything, but the real states grow a lot with depth.
- Status now: stands; not tried again.
- Goal: meaning
- User said: "sure go for orthogonal K on var 10"; "is vicreg with standardization stupid?"
- Files worth reusing: checkpoint `ortho`.

## 11-06 21:59 Isometric loss with natural scales (`iso`)
- Question: if the latent is asked to keep the same distances between images as the real states (instead of a fixed spread), and layers keep their natural scale, does prediction improve?
- Setup: one global standardization; isometric loss in place of the variance term; covariance kept; relative reconstruction; all five weights 1; Muon lr 0.01. wandb `2y1r86v3` (deleted); checkpoint `iso`.
- Result (`full_11-06.txt line 3037`), validation: one-step **0.369** (shared linear map 0.483, separate maps 0.196); rollout 0.752 (shared linear map 1.47); reconstruction 0.066; latent variance follows the state's (ratio 0.85 to 0.97). But the latent rank is 4.2 / 4.4 / 3.8 / 3.0 at blocks 1 / 6 / 9 / 12 against 28 / 70 / 160 / 154 for the states; the top 10 directions hold 99.5%; K's error on step 6→7 is 0.009 on the top direction, 0.52 on directions 11 to 100, 1.22 on the rest. Condition number 10,700 (may be affected by the same eig bug).
- Conclusion at the time: best shared-K prediction so far, and scales are natural, but the latent is again almost collapsed and K is only right on the top few directions.
- Status now: the isometric loss was kept and is still in the models (runs named `im_iso_*`). The collapse it left was not solved by losses; it stopped mattering in the one-step setup.
- Goal: groundwork
- User said: "okay lets try that"
- Files worth reusing: checkpoint `iso`.

## 11-06 22:08 More covariance (`cov10`, `cov25`)
- Question: does a stronger covariance penalty spread the latent out?
- Setup: `iso` with covariance weight 10 (`ylhl8jld`) and 25 (`wvzqc65r`); both deleted; checkpoints exist.
- Result (`full_11-06.txt line 3078`), validation, `iso` / `cov10` / `cov25`: rank at blocks 1 / 6 / 9 / 12: 4 / 4 / 4 / 3; 1 / 1 / 1 / 1; 164 / 10 / 3 / 10. Reconstruction 0.066 / 0.158 / 0.272. One-step 0.369 / 0.461 / 0.550. Rollout 0.752 / 0.768 / 0.782. During training the rank falls: `cov10` 807 / 250 / 5 / 3 at epochs 20 / 50 / 100 / 200; `cov25` 898 / 806 / 391 / 29.
- Conclusion at the time: worse on every count. The pooled covariance penalty has a loophole: it is small if each layer uses a different few directions (`cov10`: penalty 0.005 at rank 1).
- Status now: stands as a negative result.
- Goal: groundwork
- User said: "what does higher cov achieve?"
- Files worth reusing: checkpoints `cov10`, `cov25`.

## 11-06 22:17 Multi-step state loss (`state`)
- Question: does a loss on decoded states several steps ahead help rollout?
- Setup: `iso` plus `decode(K^h φ(x_l))` against `x_(l+h)`. wandb `m1kcxfcs` (deleted); checkpoint `state`.
- Result (`full_11-06.txt line 3133`), train / validation, `iso` vs `state`: one-step 0.357 / 0.369 vs 0.272 / **0.293**; rollout 0.652 / 0.752 vs 0.442 / 0.823; reconstruction 0.060 / 0.066 vs 0.083 / 0.092. Validation rollout was 0.733 at epoch 20, then rose to 0.823. Rank fell 107 to 8. Median size of λ went 0.55 to 0.88, with 353 above 1.
- Conclusion at the time: best one-step with a shared K (40% better than a shared linear map). Rollout on validation overfits. Collapse not fixed.
- Status now: the reading "rollout fails because of overfitting" was **overturned on 11-07 00:13**: from block 1 even the best linear map only reaches FVU 0.908, because block 1 does not hold the information.
- Goal: speed
- User said: "sure go for it"
- Files worth reusing: checkpoint `state`.

## 11-06 22:24 Is the isometric loss doing anything? (`no_iso`)
- Question: ablate the isometric loss.
- Setup: `state` without it. wandb `m2d2543m` (deleted); checkpoint `no_iso`.
- Result (`full_11-06.txt line 3168`), `state` vs `no_iso`: one-step validation 0.293 vs 0.294; reconstruction 0.092 vs 0.083; rollout validation 0.823 vs 0.873; rank inside a layer 3 to 4 vs 1; ratio of latent to state variance 0.90 to 1.00 vs 9.8 to 67; noise test on step 11→12 1.91 vs 3.96.
- Conclusion at the time: it does not help prediction. It keeps the latent's scale natural and a little healthier, so it was kept.
- Status now: stands. The same question came back on ImageNet later (runs `im_shared_noiso_2x4k`, `im_shared_iso0.1_2x4k`, outside this period).
- Goal: groundwork
- User said: "ablate iso, is it really even helping?"
- Files worth reusing: checkpoint `no_iso`.

## 11-06 22:30 Class tests on `iso` and `state`, and a claim that was later withdrawn
- Question: with rank this low, are the modes class-specific, and is class information lost?
- Setup: same class tests as 19:20, on `var10`, `iso`, `state`.
- Result (`full_11-06.txt line 3211`): full class direction AUC, `var10` / `iso` / `state`: block 6 0.962 / 0.838 / 0.831; block 12 0.999 / 0.861 / 0.885. `state`, best single direction (K modes / latents / PCA / random): block 9 0.822 / 0.901 / 0.850 / 0.896; block 12 0.754 / 0.895 / 0.878 / 0.920. Selectivity 0.34 to 0.45 everywhere.
- Conclusion at the time: "low rank costs class information".
- Status now: **withdrawn at 22:55.** A linear classifier on all latent dimensions gives the same 10-way accuracy as raw CLIP states (raw / `var10` / `state` / `relvar`: block 6 0.899 / 0.883 / 0.893 / 0.874; block 12 0.992 / 0.991 / 0.991 / 0.991; `full_11-06.txt line 3372`). The class information is all there; single directions just do not line up with classes.
- Goal: meaning
- User said: "however, isn't effective rank being so low still a problem? you didn't show me how the iso version does in terms of our earlier tests to check whether modes are class speciifc?"
- Files worth reusing: none (notebook `12_state_modes.ipynb` deleted).

## 11-06 22:42 Per-layer covariance (`rank`): collapse onto four coordinates
- Question: does applying the covariance penalty inside each layer close the loophole?
- Setup: per-layer covariance weight 1, batch 1024, 400 epochs. Checkpoint `rank`.
- Result (`full_11-06.txt line 3282`): mean rank 610 at epoch 46, 54 at epoch 96, about 3 at epoch 196, 2.4 at the end. **Four latent coordinates (196, 478, 738, 853) hold 99% of the variance at every layer.** Final validation: one-step 0.323, rollout 0.812, reconstruction 0.154, latent 0.041.
- Conclusion at the time: failed. A covariance penalty alone cannot stop a collapse onto a few big coordinates (the VICReg paper says so); the variance term had been removed.
- Status now: stands as a negative result. A first table in that message had wrong epoch labels and was corrected in the same hour.
- Goal: groundwork
- User said: "sure lets try to fix rank"; "seems like its failing"
- Files worth reusing: checkpoint `rank`.

## 11-06 22:55 Relative variance floor (`relvar`)
- Question: does a floor on each dimension's spread, relative to its layer, stop the four-coordinate collapse?
- Setup: `rank` plus a hinge: every latent dimension's std at least 0.5 times that layer's typical std; weight 10. wandb `3rc2sbpp` (deleted); checkpoint `relvar`.
- Result (`full_11-06.txt line 3372`), train / validation, `state` / `rank` / `relvar`: rank inside a layer 3 to 4 / 2 to 3 / **13 to 17**; coordinates that hold 99%: not measured / 4 / **984**; one-step 0.272 / 0.293, 0.299 / 0.323, 0.298 / 0.330; rollout validation 0.823, 0.812, 0.823; reconstruction 0.083 / 0.092, 0.140 / 0.154, 0.159 / 0.183. Selectivity of modes still 0.33 to 0.44.
- Conclusion at the time: the latent now uses all coordinates and a rank of 13 to 17, at a cost of about 0.04 in one-step FVU. Modes are still not class-specific.
- Status now: superseded; this hinge and the covariance term were switched off on 11-07 01:04 with no loss.
- Goal: groundwork
- User said: "go for it"
- Files worth reusing: checkpoint `relvar`.

## 11-06 23:15 TopK on top of `relvar` (`sparse32`, `sparse128`)
- Question: with a healthy latent, does TopK now give class-specific latents?
- Setup: `relvar` + TopK 32 or 128 of 1024. wandb `9hnju8b6`, `985k0vwj` (deleted); checkpoints exist.
- Result (`full_11-06.txt line 3407`), train / validation, `relvar` / `sparse128` / `sparse32`: reconstruction 0.159 / 0.183, 0.257 / 0.308, 0.322 / 0.399; one-step 0.298 / 0.330, 0.351 / 0.401, 0.387 / 0.454; rollout validation 0.823, 0.804, 0.823. At block 12 (`sparse128` / `sparse32`): share of latents distinct to a class 0% / 0%; latents above 0.5 purity 0 / 85; above 0.9: 0 / 0; linear classifier 0.971 / 0.943. About 95% of the energy of K's output is in its top-k entries, but only about 35% of those entries match the true next state's top-k.
- Conclusion at the time: no class-specific latents; sparsity only cost prediction. Suspected cause, not tested: the variance floor forces every coordinate to be used by every image.
- Status now: superseded. The question changed the next day from "class" to "feature", and TopK with a Jordan K did give single-feature modes.
- Goal: meaning
- User said: "sparsity"
- Files worth reusing: checkpoints `sparse32`, `sparse128`.

