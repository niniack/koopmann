"""The hoofed-species set through CLIP with control_train's Teacher: states after blocks 6 and 12, and block-6 patch tokens for the control."""

import sys
from pathlib import Path

import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from control_train import Teacher  # noqa: E402
from extract_hoofed import HERE, Hoofed  # noqa: E402
from extract_shift import extract  # noqa: E402

if __name__ == "__main__":
    torch.set_num_threads(16)
    teacher = Teacher("ViT-B-32", "cpu")
    preprocess = transforms.Compose([*teacher.resize, teacher.to_rgb, transforms.PILToTensor()])
    torch.save(extract(teacher, Hoofed(preprocess)), HERE / "hoofed_clip_patches.pt")
