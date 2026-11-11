import json, os
os.chdir("/home/nsa325/work/koopmann/second_analysis")
nb = json.load(open("22_imagenet_long.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]
exec(cells[1], globals())
exec(cells[3], globals())
exec(cells[5].split("bundles = {name")[0], globals())
exec(cells[11], globals())
exec(cells[16], globals())

head_path = Path("/home/nsa325/work/koopmann/temp/zero_shot_head.pt")
if head_path.exists():
    head = torch.load(head_path)
else:
    clip.to("cuda")
    head = open_clip.build_zero_shot_classifier(
        clip, tokenizer, open_clip.IMAGENET_CLASSNAMES, open_clip.OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10, device="cuda", use_tqdm=False,
    ).cpu()
    clip.to("cpu")
    torch.save(head, head_path)


def zero_shot(states):
    return (to_embeddings(states) @ head).argmax(dim=1)


def kae_forward(name, top_k, jordan, width=1024):
    kae = load_model(name, top_k, jordan, width)
    return lambda s: kae.decode(kae.koopman_forward(kae.encode(s), 1))


surrogates = {
    "linear": lambda s: apply_linear(linear, s),
    "jordan + topk, 1024, 20 epochs": kae_forward("im_jordan128", 128, True),
    "jordan, 1024": kae_forward("im_jordan_e100", 0, True),
    "dense, 1024": kae_forward("im_dense_e100", 0, False),
    "dense, 2048": kae_forward("im_dense_2k", 0, False, 2048),
}

true_val = zero_shot(val_x[:, 1])
true_snow = zero_shot(x[:, 1])
rows = {"real CLIP": {
    "zero-shot": (true_val == val_y).double().mean().item(),
    "agrees with CLIP": 1.0,
    "ridge classifier": (predict_class(val_x[:, 1]) == val_y).double().mean().item(),
    "snowy": (true_snow == y)[is_snow].double().mean().item(),
    "normal": (true_snow == y)[~is_snow].double().mean().item(),
}}
for name, forward in surrogates.items():
    val_pred, snow_pred = forward(val_x[:, 0]), forward(x[:, 0])
    labels = zero_shot(val_pred)
    snow_labels = zero_shot(snow_pred)
    rows[name] = {
        "zero-shot": (labels == val_y).double().mean().item(),
        "agrees with CLIP": (labels == true_val).double().mean().item(),
        "ridge classifier": (predict_class(val_pred) == val_y).double().mean().item(),
        "snowy": (snow_labels == y)[is_snow].double().mean().item(),
        "normal": (snow_labels == y)[~is_snow].double().mean().item(),
    }
print(pd.DataFrame(rows).T.round(3).to_string())

# The snow edit, scored with CLIP's own head
edits = {
    "linear ICA 14": (get_ica_bundle(lambda s: apply_linear(linear, s), lambda s: s), "ica 14"),
    "dense 2048 ICA 96": (get_kae_ica("im_dense_2k", 0, False, 2048), "ica 96"),
    "jordan + topk mode 414": (get_mode_bundle("im_jordan128", 128, True), "mode 414"),
}
rows = {}
for name, ((features, edit, predict, labels), label) in edits.items():
    index = labels.index(label)
    base, edited = zero_shot(predict(x[:, 0])), zero_shot(edit(x[:, 0], index))
    rows[name] = {
        "AUC": get_auc(features(x[:, 0])[:, index : index + 1], is_snow)[0].item(),
        "snowy before": (base == y)[is_snow].double().mean().item(),
        "snowy after": (edited == y)[is_snow].double().mean().item(),
        "normal before": (base == y)[~is_snow].double().mean().item(),
        "normal after": (edited == y)[~is_snow].double().mean().item(),
    }
print()
print(pd.DataFrame(rows).T.round(3).to_string())

# Rough compute: a matrix multiply costs 2 FLOPs per weight per token
tokens, width = 50, 768
blocks = clip.visual.transformer.resblocks
block_weights = sum(p.numel() for p in blocks[0].parameters() if p.ndim == 2)
block_flops = 2 * block_weights * tokens + 4 * tokens**2 * width
patch_flops = 2 * clip.visual.conv1.weight.numel() * 49
kae = load_model("im_dense_2k", 0, False, 2048)
kae_weights = sum(p.numel() for p in kae.parameters() if p.ndim == 2)
print()
print(f"One block: {block_flops / 1e9:.3f} GFLOPs, {block_weights / 1e6:.1f}M weights")
print(f"Whole image encoder: {(patch_flops + 12 * block_flops) / 1e9:.2f} GFLOPs")
print(f"Blocks 7-12: {6 * block_flops / 1e9:.2f} GFLOPs, {6 * block_weights / 1e6:.1f}M weights")
print(f"Dense KAE, 2048 wide: {2 * kae_weights / 1e9:.3f} GFLOPs, {kae_weights / 1e6:.1f}M weights")
print(f"Surrogate image encoder: {(patch_flops + 6 * block_flops + 2 * kae_weights) / (patch_flops + 12 * block_flops):.3f} of the original compute")
