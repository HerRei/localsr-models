# LocalSR Photo models

**Small 4x photo upscalers that run entirely on Apple's Neural Engine, trained only on CC0 photos.**

By Hermes Reisner · **[Technical report (PDF)](report/localsr-models.pdf)** ·
[Download (release v1)](https://github.com/HerRei/localsr-models/releases/tag/v1) ·
[Hugging Face](https://huggingface.co/HerRei/localsr-models) · Weights CC BY 4.0, code MIT

I made these models for [LocalSR](https://github.com/HerRei/local-upscale), my free app that upscales photos and
videos on your own machine. They run on the NPU, so you can upscale in the background while you keep using your
computer.

**In short:**
- **Photo S** is as fast as or faster than the model LocalSR used until now for its Quick preset, and better.
- **Photo M** is 5 to 11 times faster than much larger models like NomosWebPhoto and Real-ESRGAN, for a little less
  quality.
- Both stay fast on old computers without an NPU. That is one of their biggest strengths.

## The models

| | **Photo S** | **Photo M** |
|---|---|---|
| File | [`4xLocalSRPhoto_PULSE-64x8`](https://github.com/HerRei/localsr-models/releases/download/v1/4xLocalSRPhoto_PULSE-64x8.safetensors) | [`4xLocalSRPhoto_PULSE-96x12`](https://github.com/HerRei/localsr-models/releases/download/v1/4xLocalSRPhoto_PULSE-96x12.safetensors) |
| Size | 2.5 MB, 0.62M parameters | 8.1 MB, 2.03M parameters |
| Neural Engine (M1 Pro) | 188 ms per input megapixel | 485 ms per input megapixel |
| Good for | everyday photos, batches | soft camera photos, single images |
| In LocalSR | the Quick preset | the Best Quality preset |

The names give the scale (4x), the purpose (photos), the network (PULSE), and its size (channels x blocks). Every
operation of the Core ML export runs on the Neural Engine. Two experimental video variants of Photo S are in the same
release.

## Fast on older computers

Seconds per input megapixel, lower is better:

| Model | 2017 laptop, i7-8550U (CPU) | Same laptop, Intel UHD 620 (iGPU) | 2012 Mac mini, i7-3720QM (CPU) |
|---|---|---|---|
| **Photo S** | 9.1 | 15.0 | 22.3 |
| SPAN NomosUni (LocalSR's Quick until now) | 10.1 | 15.5 | 22.1 |
| **Photo M** | 25.1 | 48.5 | 58.9 |
| NomosWebPhoto RealPLKSR (LocalSR's Best Quality until now) | 200.8 | 340.9 | 317.1 |
| Real-ESRGAN x4plus | 272.1 | 462.5 | 554.0 |

- **Photo S** is faster than SPAN on the laptop and just as fast on the Mac mini.
- **Photo M** is 5 to 8 times faster than NomosWebPhoto and 9 to 11 times faster than Real-ESRGAN. On the 2017 laptop
  a 1-megapixel photo to 16 megapixels takes 25 seconds with Photo M, against 3.3 minutes with NomosWebPhoto and 4.5
  minutes with Real-ESRGAN.

Measured with one 256x256 input (1024x1024 output) in fp32, 5 interleaved rounds, the median, and each model at its
fastest thread count. The iGPU runs through ONNX Runtime DirectML 1.24.4, as in LocalSR on Windows, and its output
matches the CPU's for every model.

## Quality

**Sealed hold-out:** 99 DIV2K photos as web copies, scored once in a single final run. Every file is at its own
operating point, so fidelity is not matched. Lower LPIPS and DISTS are better.

| Model | Parameters | PSNR-Y | LPIPS | DISTS |
|---|---|---|---|---|
| **Photo S** | 0.62M | 28.40 | 0.259 | 0.148 |
| **Photo M** | 2.03M | 28.40 | 0.245 | 0.139 |
| SPAN NomosUni | 2.24M | 26.97 | 0.276 | 0.159 |
| NomosWebPhoto RealPLKSR | 7.4M | 25.78 | 0.249 | 0.137 |
| Real-ESRGAN x4plus | 16.7M | 26.22 | 0.242 | 0.128 |

- **Photo S** beats SPAN on LPIPS and DISTS, at 1.4 dB higher fidelity.
- **Photo M** shows no clear difference to NomosWebPhoto on LPIPS and DISTS (the paired intervals include zero), at
  2.6 dB higher fidelity and with about a quarter of the parameters.
- **Real-ESRGAN** has the lowest DISTS, at 2.2 dB lower fidelity.

**Real camera photos** (RealSR and DRealSR crops, A-FINE, lower is better):

| Model | Parameters | RealSR | DRealSR |
|---|---|---|---|
| **Photo S** | 0.62M | 70.8 | 69.1 |
| **Photo M** | 2.03M | 65.1 | 67.2 |
| NomosWebPhoto RealPLKSR | 7.4M | 58.6 | 70.2 |
| Real-ESRGAN x4plus | 16.7M | 39.5 | 40.0 |

Photo M is clearly better than Photo S on camera photos, because it learned the blur that real cameras leave (a
lens-blur regime in the training degradations). That held with a second training seed. The much larger Real-ESRGAN is
still clearly ahead by A-FINE, most of all on soft and noisy photos.

## Video variants (experimental)

Run frame by frame on video, a GAN-trained upscaler re-invents fine texture from each frame's grain, and that texture
flickers. The video variants are Photo S, fine-tuned to give the same result for two noisy, slightly shifted copies of
the same crop.

| File | Flicker, still shot | Flicker, camera shake | Sharpness (LPIPS, lower is better) |
|---|---|---|---|
| `4xLocalSRPhoto_PULSE-64x8` | 2.42 | 4.01 | 0.328 |
| `4xLocalSRPhoto_PULSE-64x8_video-light-exp` | −10 % | −5 % | 0.322 |
| `4xLocalSRPhoto_PULSE-64x8_video-calm-exp` | −39 % | −37 % | 0.383 |

*Light* is a little calmer and just as sharp. *Calm* removes over a third of the flicker and is softer on a paused
frame. Neither passed the rule I set before training, and in a side-by-side eye test I could not tell them apart from
Photo S. So for video I recommend plain Photo S.

## Quick start

```sh
pip install torch safetensors pillow
python upscale.py 4xLocalSRPhoto_PULSE-64x8.safetensors photo.jpg photo_x4.png
```

`pulse_arch.py` is the network (plain PyTorch, about 80 lines) and `upscale.py` a small command line around it. Large
images are processed in tiles whose overlap is wider than the network's receptive field, so tiling never changes the
result. Check downloads against `SHA256SUMS`.

## How they were made

- **Data:** [lucid-cc0-v2-hc](https://huggingface.co/datasets/Phips/lucid-cc0-v2-hc), photos from pxhere that the
  collection's publisher releases as CC0. Nothing else was used for training.
- **Degradations:** my own phone-photo pipeline (blur, zoom, resampling, sensor noise, noise reduction, sharpening,
  JPEG, WebP, HEIC, AVIF), plus a lens-blur regime for Photo M.
- **Training:** an L1 phase, then a GAN phase with MS-SSIM, DINOv2 features, a sliced Wasserstein texture term, and a
  PatchGAN. The released files are blends of both phases, with a head that is corrected against checkerboard texture.
- **Evaluation:** 166 web copies of my own photos, the RealSR and DRealSR camera sets, and the sealed DIV2K hold-out.
  Differences are paired per photo, with bootstrap intervals. The details are in the
  [report](report/localsr-models.pdf).

## Limits

- **Soft and noisy photos:** Real-ESRGAN restores clearly more there.
- **The judges disagree:** A-FINE ranks Real-ESRGAN first everywhere, and on the hold-out Real-ESRGAN also has the
  lowest DISTS. LPIPS and DISTS favour these files over the models they are meant to replace. There is no rater study
  yet.
- **Photos only:** not trained for illustrations, anime, or pixel art.

## Licence and credits

- **Weights:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (`LICENSE-WEIGHTS`). You can use them
  freely, commercially too, as long as you credit them, for example: *LocalSR models by Hermes Reisner
  (https://github.com/HerRei/localsr-models), CC BY 4.0*.
- **Code** (`pulse_arch.py`, `upscale.py`): MIT (`LICENSE`).
- **Credits:**
  - The network layout follows Philip Hofmann's [PULSE](https://huggingface.co/Phips/PULSE) specification (MIT).
    `pulse_arch.py` is an independent reimplementation, and the weights come from my own runs, not his checkpoints.
  - The training data is Philip Hofmann's lucid-cc0-v2-hc, CC0 photographs published on pxhere.
  - DINOv2 (Apache-2.0) was used as a training loss only and is not part of the weights.

## Citation

```bibtex
@misc{reisner2026localsr,
  author       = {Hermes Reisner},
  title        = {Small, Licence-Clean 4x Photo Upscalers for the Apple Neural Engine, and What Frame-by-Frame
                  Video Does to Them},
  year         = {2026},
  howpublished = {\url{https://github.com/HerRei/localsr-models}}
}
```
