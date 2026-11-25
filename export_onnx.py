"""Export LocalSR models to ONNX (fp32 and fp16) and check them against PyTorch.

    python export_onnx.py weights/4xLocalSRPhoto_PULSE-64x8.safetensors [--out onnx/] [--check photo.jpg]

Needs torch, safetensors, onnx, onnxruntime, numpy and Pillow. The graph takes any input size: input "input" is
1x3xHxW RGB in [0, 1], output "output" is 1x3x4Hx4W (not clamped, clamp to [0, 1] yourself). The fp16 file has fp16
input and output.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from safetensors import safe_open

from upscale import load_model

OPSET = 17
KEEP = ("model", "author", "license", "attribution", "arch", "scale", "dim", "n_blocks", "datasets")


def export(model, path, half, meta):
    model = model.half() if half else model.float()
    dummy = torch.rand(1, 3, 64, 64, dtype=torch.float16 if half else torch.float32)
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["output"], opset_version=OPSET,
                      dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
                      dynamo=False)
    proto = onnx.load(str(path))
    onnx.helper.set_model_props(proto, {**meta, "precision": "fp16" if half else "fp32", "output_clamp": "none"})
    proto.doc_string = meta.get("attribution", "")
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))


def psnr(a, b):
    mse = float(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1 / mse)


def check(model, path, x, half):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        ref = model.float()(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x.astype(np.float16) if half else x})[0].astype(np.float32)
    return float(np.abs(got - ref).max()), psnr(got, ref)


def main():
    p = argparse.ArgumentParser(description="Export a LocalSR model to ONNX.")
    p.add_argument("weights")
    p.add_argument("--out", default="onnx")
    p.add_argument("--check", nargs="*", default=[], help="images to compare ONNX Runtime with PyTorch on")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    stem = Path(a.weights).stem
    with safe_open(a.weights, "pt") as f:
        meta = {k: v for k, v in (f.metadata() or {}).items() if k in KEEP}
    meta["source_sha256"] = hashlib.sha256(Path(a.weights).read_bytes()).hexdigest()
    for half in (False, True):
        path = out / f"{stem}{'_fp16' if half else ''}.onnx"
        export(load_model(a.weights, "cpu"), path, half, meta)
        print(f"{path}  {path.stat().st_size / 1e6:.2f} MB")
        for img in a.check:
            arr = np.asarray(Image.open(img).convert("RGB"), dtype=np.float32) / 255
            x = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
            diff, db = check(load_model(a.weights, "cpu"), path, x, half)
            print(f"  {Path(img).name} {x.shape[3]}x{x.shape[2]}: max abs diff {diff:.2e}, PSNR vs PyTorch {db:.1f} dB")


if __name__ == "__main__":
    main()
