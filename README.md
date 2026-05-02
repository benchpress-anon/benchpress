# BenchPress — Video Object Removal Pipeline

SAM3 text-prompted segmentation + DiffuEraser temporal-consistent inpainting for counterfactual video editing.

## Setup

```bash
# Clone with the DiffuEraser submodule
git clone --recurse-submodules <repo-url>
cd benchpress

# (or if already cloned without --recurse-submodules:)
# git submodule update --init

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (SAM3 support is in transformers 5.3.0 on PyPI)
pip install -r requirements.txt

# DiffuEraser weights — download into third_party/DiffuEraser/weights/:
#   - stable-diffusion-v1-5
#   - sd-vae-ft-mse
#   - diffuEraser checkpoint
#   - ProPainter weights
#   - PCM weights
# See third_party/DiffuEraser/README.md for download links.
```

## Dataset

BenchPress is the generation pipeline for the **NExT-CF dataset**, a counterfactual video evaluation benchmark derived from NExT-QA.

### Input data

- **Original videos**: NExT-QA validation set (570 videos), available from the [NExT-QA project page](https://github.com/doc-doc/NExT-QA). Place under a directory and pass via `--video-dir` / `--dataset-root`.
- **Prompt mapping**: `data.json` (included in this repo) — maps each `video_id` to the `prompt_word` (entity to remove).

### Output / Reference dataset

The edited videos produced by this pipeline constitute the **NExT-CF dataset**, released at:

  https://huggingface.co/datasets/nextcf-anon-2026/nextcf

Use this URL for the canonical edited videos and metadata (Croissant manifest, attribution, license). Reviewers can verify reproducibility by either:

1. Re-running this pipeline against NExT-QA validation videos and comparing outputs to the released dataset; or
2. Inspecting the released dataset directly without re-running the GPU pipeline.

The dataset is distributed under CC-BY-NC-SA-4.0 (videos) and CC-BY-4.0 (metadata).

## Usage

### Single video
```bash
python pipeline.py \
  --video input.mp4 \
  --prompt "dog" \
  --edit remove \
  --out runs/my_run
```

### Reuse masks from a previous run
```bash
python pipeline.py \
  --video input.mp4 \
  --prompt "dog" \
  --edit remove \
  --reuse-masks runs/prev_run/masks \
  --out runs/new_run
```

### Batch processing across GPUs
```bash
python batch_runner.py \
  --video-dir /path/to/videos/ \
  --prompt-file data.json \
  --num-gpus 3 \
  --out /path/to/output/
```

## Pipeline Flow

1. **Video decode** — Load frames from input video (`video_io.py`)
2. **SAM3 segmentation** — Text-prompted anchor detection + bidirectional video tracker propagation (`sam3_segmenter.py`)
3. **Mask overlay** — Visualization of segmentation masks (`overlay_render.py`)
4. **DiffuEraser inpainting** — BrushNet + UNet (SD 1.5) + ProPainter temporal prior (`diffueraser_backend.py`)
5. **Video encode** — Write edited frames to output video

## Hardware

- **GPU:** 8+ GB VRAM minimum, 24 GB recommended
- **Multi-GPU:** SAM3 distributes image model, tracker, and DINOv3 across GPUs
- **Turing GPUs (RTX 6000, SM 7.5):** bf16 breaks SDPA — backends auto-detect and use fp16

## Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--edit-backend` | `diffueraser` | Inpainting backend; `telea` is a cv2 fallback for failed clips |
| `--sam3-video-tracker` | ON | Bidirectional mask propagation (recommended) |
| `--no-sam3-video-tracker` | — | Per-frame text segmentation fallback |
| `--dino-verify` | OFF | DINOv3 verification after tracker |
| `--multi-object` | OFF | Track multiple objects separately |
| `--reuse-masks` | — | Skip segmentation, reuse saved masks |
| `--skip-inpaint` | — | Only run segmentation + overlay |
| `--max-frames` | — | Limit frame count |
| `--seed` | 42 | Random seed |

## Output Structure

```
runs/my_run/
├── overlay.mp4           # Mask visualization
├── masks/
│   ├── png/              # Per-frame mask PNGs
│   └── metadata.json     # Mask metadata
├── remove/
│   ├── edited.mp4        # Final edited video
│   └── diffueraser_results/
└── logs.jsonl            # Pipeline logs
```
