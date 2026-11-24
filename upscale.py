"""Upscale an image 4x with a LocalSR model.

    python upscale.py 4xLocalSRPhoto_PULSE-64x8.safetensors photo.jpg photo_x4.png [--tile 256] [--device cpu]

Needs torch, safetensors and Pillow. Large images are processed in tiles with a 32 px overlap, which is wider than the
network's receptive field, so the result equals a single full-image pass.
"""
import argparse

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from pulse_arch import PULSE

OVERLAP = 32   # receptive field: 20 px LR for Quick (8 blocks), 28 px for Best (12 blocks)


def load_model(path, device):
    state = load_file(path)
    dim = state["stem.weight"].shape[0]
    blocks = len({k.split(".")[1] for k in state if k.startswith("body.")})
    model = PULSE(dim=dim, scale=4, n_blocks=blocks)
    model.load_state_dict(state)
    return model.eval().to(device)


@torch.inference_mode()
def upscale(model, x, tile=256):
    """x: 1x3xHxW in [0, 1] -> 1x3x4Hx4W."""
    _, _, h, w = x.shape
    if max(h, w) <= tile:
        return model(x).clamp(0, 1)
    out = torch.empty(1, 3, 4 * h, 4 * w, device=x.device)
    for y in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y + tile, h), min(x0 + tile, w)
            ya, xa = max(0, y - OVERLAP), max(0, x0 - OVERLAP)
            yb, xb = min(h, y1 + OVERLAP), min(w, x1 + OVERLAP)
            part = model(x[:, :, ya:yb, xa:xb])
            out[:, :, 4 * y:4 * y1, 4 * x0:4 * x1] = part[:, :, 4 * (y - ya):4 * (y1 - ya), 4 * (x0 - xa):4 * (x1 - xa)]
    return out.clamp(0, 1)


def main():
    p = argparse.ArgumentParser(description="Upscale an image 4x with a LocalSR model.")
    p.add_argument("weights")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--tile", type=int, default=256, help="input tile size in px")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()
    model = load_model(a.weights, a.device)
    img = np.asarray(Image.open(a.input).convert("RGB"), dtype=np.float32) / 255
    x = torch.from_numpy(img).permute(2, 0, 1)[None].to(a.device)
    y = upscale(model, x, a.tile)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray((y * 255 + .5).astype(np.uint8)).save(a.output)


if __name__ == "__main__":
    main()
