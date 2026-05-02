"""
pipeline.py - Main Orchestration for Counterfactual Video Editor

Orchestrates:
1. Video loading
2. SAM 3 segmentation
3. Mask overlay rendering
4. DiffuEraser inpainting
5. Final video output
6. Comprehensive logging

Usage:
    python -m pipeline --video input.mp4 --prompt "dog" --edit remove --out runs/run1
    python -m pipeline --video input.mp4 --prompt "dog" --edit swap --swap_target "cat" --out runs/run2
    python -m pipeline --video input.mp4 --prompt "dog" --edit remove --edit_backend diffueraser --out runs/run3
"""

import os
import sys
import json
import shutil
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

# Local imports
from video_io import decode_video_to_pil, encode_video, get_video_info
from sam3_segmenter import SAM3VideoSegmenter, SegmentationResult, save_masks, load_masks
from overlay_render import save_overlay_video, render_overlay_video
from diffueraser_backend import (
    DiffuEraserBackend,
    DiffuEraserConfig,
    InProcessDiffuEraserRunner,
    run_telea_fallback,
    run_swap_stub,
    check_diffueraser_available,
)


# Backend types
BACKEND_DIFFUERASER = "diffueraser"
BACKEND_TELEA = "telea"

ALL_BACKENDS = (
    BACKEND_DIFFUERASER,
    BACKEND_TELEA,
)


class JSONLLogger:
    """JSONL logger for audit trail."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        # Clear existing log
        open(log_path, 'w').close()
    
    def log(self, record: Dict[str, Any]):
        """Append a record to the log."""
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    
    def log_clip(
        self,
        video_id: str,
        input_video_path: str,
        text_prompt: str,
        edit_type: str,
        swap_target: Optional[str],
        sam3_num_objects: int,
        selected_object_id: int,
        clip_seed: int,
        num_frames: int,
        mean_mask_area_frac: float,
        overlay_video_path: str,
        edited_video_path: str,
        backend: str = "diffusion",
    ):
        """Log clip-level metadata."""
        self.log({
            "record_type": "clip",
            "video_id": video_id,
            "input_video_path": input_video_path,
            "text_prompt": text_prompt,
            "edit_type": edit_type,
            "swap_target": swap_target,
            "sam3_num_objects": sam3_num_objects,
            "selected_object_id": selected_object_id,
            "clip_seed": clip_seed,
            "num_frames": num_frames,
            "mean_mask_area_frac": mean_mask_area_frac,
            "overlay_video_path": overlay_video_path,
            "edited_video_path": edited_video_path,
            "backend": backend,
            "timestamp": datetime.now().isoformat(),
        })
    
    def log_frame(
        self,
        frame_idx: int,
        object_id: int,
        frame_seed: int,
        mask_area_px: int,
        mask_area_frac: float,
        inpaint_steps: int,
        guidance_scale: float,
        strength: float,
        prompt: str,
        negative_prompt: str,
    ):
        """Log per-frame generation details."""
        self.log({
            "record_type": "frame",
            "frame_idx": frame_idx,
            "object_id": object_id,
            "frame_seed": frame_seed,
            "mask_area_px": mask_area_px,
            "mask_area_frac": mask_area_frac,
            "inpaint_steps": inpaint_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        })


def _free_gpu_after_inpainting() -> None:
    """Release GPU memory after an inpainting backend finishes."""
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_prebuilt_video(edit_backend: str, remove_dir: Path) -> Optional[str]:
    """Return the path to the pre-built output video for a given backend."""
    if edit_backend == BACKEND_DIFFUERASER:
        _alt = str(remove_dir / "edited.mp4")
        _pri = str(remove_dir / "diffueraser_results" / "diffueraser_result.mp4")
        return _alt if os.path.exists(_alt) else _pri
    return None
def generate_video_id(video_path: str) -> str:
    """Generate a unique ID for a video based on path and name."""
    basename = Path(video_path).stem
    path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    return f"{basename}_{path_hash}"


def run_pipeline(
    input_video_path: str,
    text_prompt: str,
    edit_type: str,
    output_dir: str,
    swap_target: Optional[str] = None,
    max_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
    uniform_subsample: bool = False,
    clip_seed: int = 42,
    overlay_alpha: float = 0.5,
    sam3_model_id: str = "facebook/sam3",
    skip_inpaint: bool = False,
    skip_postprocess: bool = False,
    reuse_masks_dir: Optional[str] = None,
    # Backend selection
    edit_backend: str = BACKEND_DIFFUERASER,
    # DiffuEraser options
    diffueraser_path: str = "third_party/DiffuEraser",
    diffueraser_device: str = "auto",
    mask_erode_px: int = 0,
    mask_dilate_px: int = 0,
    mask_feather_sigma: float = 0.0,
    # Multi-object
    multi_object: bool = False,
    merge_proximity_px: int = 50,
    # SAM3 video tracker + optional DINOv3 verification
    use_dino_tracker: bool = False,
    dino_verify: bool = False,
    # Pre-loaded models (for batch mode -- avoids reloading per video)
    _preloaded_segmenter: Optional["SAM3VideoSegmenter"] = None,
    _preloaded_diffueraser_runner: Optional["InProcessDiffuEraserRunner"] = None,
    _preloaded_dino_tracker: Optional["DINOv3Tracker"] = None,
) -> Dict[str, Any]:
    """
    Run the full video editing pipeline.
    
    Args:
        input_video_path: Path to input video
        text_prompt: Object to segment (e.g., "dog")
        edit_type: "remove" or "swap"
        output_dir: Directory for outputs
        swap_target: Target object for swap (required if edit_type="swap")
        max_frames: Limit frames for testing
        target_fps: Resample video to a target FPS
        clip_seed: Base seed for reproducibility
        overlay_alpha: Mask overlay opacity
        sam3_model_id: SAM 3 model ID
        skip_inpaint: If True, only run segmentation + overlay
        reuse_masks_dir: Path to a previously saved masks directory. If set,
            SAM 3 segmentation is skipped and masks are loaded from disk.
        diffueraser_path: Path to DiffuEraser repo
            mask_erode_px: Mask erosion for preprocessing
        mask_dilate_px: Mask dilation for preprocessing
        mask_feather_sigma: Mask feather sigma
        multi_object: Keep all detected instances. Nearby masks are merged
            into one group (single inpainting run); distant masks become
            separate groups inpainted sequentially.
        merge_proximity_px: Max pixel gap for two masks to be merged into
            one group (only used when multi_object=True).
                                
    Returns:
        Dictionary with output paths and metadata
    """
    # Validate inputs
    if edit_type not in ("remove", "swap"):
        raise ValueError(f"edit_type must be 'remove' or 'swap', got: {edit_type}")
    
    if edit_type == "swap" and not swap_target:
        raise ValueError("swap_target is required when edit_type='swap'")
    
    if edit_backend not in ALL_BACKENDS:
        raise ValueError(
            f"edit_backend must be one of: {', '.join(ALL_BACKENDS)}"
        )

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Video not found: {input_video_path}")
    
    # Setup output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    remove_dir = output_dir / "remove"
    remove_dir.mkdir(exist_ok=True)
    
    # Output paths
    overlay_path = str(output_dir / "overlay.mp4")
    edited_path = str(remove_dir / "edited.mp4") if edit_type == "remove" else str(output_dir / "edited.mp4")
    log_path = str(output_dir / "logs.jsonl")
    
    # Initialize logger
    logger = JSONLLogger(log_path)
    video_id = generate_video_id(input_video_path)
    
    print("=" * 60)
    print("Counterfactual Video Editor Pipeline")
    print("=" * 60)
    print(f"Input: {input_video_path}")
    prompts = [p.strip() for p in text_prompt.split(",")]
    is_multi_prompt = len(prompts) > 1
    print(f"Prompt: '{text_prompt}'" + (f"  ({len(prompts)} prompts)" if is_multi_prompt else ""))
    print(f"Edit type: {edit_type}")
    print(f"Backend: {edit_backend}")
    if swap_target:
        print(f"Swap target: '{swap_target}'")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Step 1: Load video frames
    print("\n[1/6] Loading video frames...")
    video_info = get_video_info(input_video_path)
    print(f"  Video: {video_info['width']}x{video_info['height']}, "
          f"{video_info['fps']:.1f} fps, {video_info['frame_count']} frames")
    
    frames, fps, width, height = decode_video_to_pil(
        input_video_path, max_frames=max_frames, target_fps=target_fps,
        uniform_subsample=uniform_subsample,
    )
    print(f"  Loaded {len(frames)} frames")
    
    # Free DiffuEraser GPU memory before segmentation — both models can't
    # coexist on 24 GB (DiffuEraser ~15 GB + SAM3 tracker ~6 GB > 24 GB).
    # Models stay in CPU RAM and are reloaded to GPU before inpainting.
    if _preloaded_diffueraser_runner is not None and hasattr(
        _preloaded_diffueraser_runner, "offload_to_cpu"
    ):
        _preloaded_diffueraser_runner.offload_to_cpu()

    # Step 2: Run SAM 3 segmentation (or reload from disk)
    print("\n[2/6] Running SAM 3 segmentation...")
    segmenter: Optional[Any] = None
    if reuse_masks_dir:
        reuse_path = Path(reuse_masks_dir)
        if not (reuse_path / "metadata.json").exists() or not (reuse_path / "masks.npz").exists():
            raise FileNotFoundError(
                f"--reuse-masks directory is missing metadata.json or masks.npz: {reuse_masks_dir}"
            )
        print(f"  Reusing saved masks from: {reuse_masks_dir}  (SAM 3 skipped)")
        segmentation = load_masks(reuse_masks_dir)
        # Allow using a subset of saved masks when --max-frames is smaller than the saved count
        if segmentation.num_frames > len(frames):
            obj_id = segmentation.selected_object_id
            segmentation.masks_by_frame = {
                i: segmentation.masks_by_frame[i]
                for i in range(len(frames))
            }
            segmentation.num_frames = len(frames)
            print(f"  Subsetting saved masks to first {len(frames)} frames.")
        elif segmentation.num_frames < len(frames):
            raise ValueError(
                f"Saved masks only have {segmentation.num_frames} frames but video loaded "
                f"{len(frames)} frames. Use --max-frames <= {segmentation.num_frames}."
            )
        print(f"  Loaded {segmentation.num_frames} masks for object '{segmentation.text_prompt}'")
        print(f"  Selected object ID: {segmentation.selected_object_id}")
        print(f"  Mean mask area: {segmentation.mean_mask_area_frac:.4f}")
    else:
        segmenter = _preloaded_segmenter or SAM3VideoSegmenter(model_id=sam3_model_id)

        if use_dino_tracker and not is_multi_prompt:
            tracker = None
            if dino_verify:
                from dino_tracker import DINOv3Tracker
                dino_dev = getattr(segmenter, "dino_device", None)
                tracker = _preloaded_dino_tracker or DINOv3Tracker(
                    device=dino_dev,
                )
                if not tracker._loaded:
                    tracker.load_model()
            segmentation = segmenter.segment_video_dino_tracked(
                frames, text_prompt,
                dino_tracker=tracker,
            )
        elif is_multi_prompt:
            segmentation = segmenter.segment_video_multi_prompt(
                frames, prompts,
                merge_proximity_px=merge_proximity_px,
            )
        else:
            segmentation = segmenter.segment_video(
                frames, text_prompt,
                multi_object=multi_object,
                merge_proximity_px=merge_proximity_px,
            )

        print(f"  Object groups: {len(segmentation.object_ids)}")
        print(f"  Selected object ID: {segmentation.selected_object_id}")
        print(f"  Mean mask area: {segmentation.mean_mask_area_frac:.4f}")
        if segmentation.merge_info:
            mi = segmentation.merge_info
            if mi.get("tracker") == "dinov3":
                print(f"  DINOv3 tracker: anchor=frame {mi['anchor_frame']} "
                      f"(score={mi['anchor_score']:.3f}), "
                      f"sim_threshold={mi['similarity_threshold']}")
            elif mi.get("multi_prompt"):
                print(f"  Multi-prompt: {mi['prompts']} → "
                      f"{mi['num_groups']} group(s) "
                      f"(proximity={mi['merge_proximity_px']}px)")
            elif "num_raw_detections" in mi:
                print(f"  Multi-object: {mi['num_raw_detections']} detections → "
                      f"{mi['num_groups']} group(s) "
                      f"(proximity={mi['merge_proximity_px']}px)")

        # Save masks so they can be reused later
        print("  Saving masks to disk...")
        save_masks(segmentation, str(masks_dir))

    
    # Step 3: Render overlay video
    print("\n[3/6] Rendering overlay visualization...")
    save_overlay_video(frames, segmentation, overlay_path, fps, overlay_alpha)
    
    # Prepare mask list(s) for inpainting
    obj_id = segmentation.selected_object_id
    num_groups = len(segmentation.object_ids)
    is_multi_group = num_groups > 1

    masks = [segmentation.get_union_mask(i) for i in range(len(frames))]

    if skip_inpaint:
        print("\n[4/6] Skipping inpainting (--skip-inpaint flag)")
        print("[5/5] Skipping video encoding")
        
        # Log clip info
        logger.log_clip(
            video_id=video_id,
            input_video_path=input_video_path,
            text_prompt=text_prompt,
            edit_type=edit_type,
            swap_target=swap_target,
            sam3_num_objects=len(segmentation.object_ids),
            selected_object_id=segmentation.selected_object_id,
            clip_seed=clip_seed,
            num_frames=len(frames),
            mean_mask_area_frac=segmentation.mean_mask_area_frac,
            overlay_video_path=overlay_path,
            edited_video_path="(skipped)",
            backend=edit_backend,
        )
        
        return {
            "video_id": video_id,
            "overlay_video": overlay_path,
            "edited_video": None,
            "masks_dir": str(masks_dir),
            "log_path": log_path,
            "segmentation": segmentation,
        }
    
    # Convert frames to numpy for inpainting
    frames_np = [np.array(f) for f in frames]

    # Step 3.5: Shrink masks where disoccluded content is already visible
    # in nearby frames.  Don't modify frames — just reduce the mask so the
    # inpainter has a smaller target and preserves more real content.
    # Swap GPU: offload SAM3, reload DiffuEraser for inpainting
    if _preloaded_diffueraser_runner is not None and hasattr(
        _preloaded_diffueraser_runner, "reload_to_gpu"
    ):
        _preloaded_diffueraser_runner.reload_to_gpu()

    # Scene-conditioned inpainting: detect & track all known objects,
    # build composite reference frames, pass to inpainter as Zone B guidance.
    _scene_composites = None
    _scene_overlap_masks = None
    _scene_tracks = []
    _scene_device = "cuda:0"
    # Step 4: Run inpainting based on backend
    if is_multi_group:
        print(f"\n[4/6] Running inpainting ({edit_type}, backend={edit_backend}, "
              f"{num_groups} groups - sequential)...")
    else:
        print(f"\n[4/6] Running inpainting ({edit_type}, backend={edit_backend})...")
    
    frame_infos: List[Dict[str, Any]] = []
    
    if edit_backend == BACKEND_DIFFUERASER:
        diffueraser_cfg = DiffuEraserConfig(
            mask_erode_px=mask_erode_px,
            mask_dilate_px=mask_dilate_px,
            mask_feather_sigma=mask_feather_sigma,
            diffueraser_path=diffueraser_path,
            device=diffueraser_device,
        )
        
        available, msg = check_diffueraser_available(diffueraser_cfg)
        if not available:
            print(f"WARNING: DiffuEraser not available. Falling back to Telea inpainting.")
            print(msg)
            edit_backend = BACKEND_TELEA
            edited_frames, backend_info = run_telea_fallback(
                frames_np, masks, str(remove_dir), radius=5
            )
        elif is_multi_group:
            current_frames = frames_np
            for g_idx, gid in enumerate(segmentation.object_ids):
                group_masks = [
                    segmentation.get_mask(i, gid) for i in range(len(frames))
                ]
                has_content = any(m.any() for m in group_masks)
                if not has_content:
                    print(f"  Group {gid}: all masks empty, skipping")
                    continue

                group_dir = str(remove_dir / f"group_{gid}")
                Path(group_dir).mkdir(exist_ok=True)
                print(f"  Group {gid}/{num_groups}: inpainting...")

                backend = DiffuEraserBackend(diffueraser_cfg)
                current_frames, _ = backend.run_remove_inpaint_frames(
                    current_frames, group_masks, group_dir,
                    inprocess_runner=_preloaded_diffueraser_runner,
                    scene_reference_frames=_scene_composites,
                    scene_overlap_masks=_scene_overlap_masks,
                )
            edited_frames = current_frames
        else:
            backend = DiffuEraserBackend(diffueraser_cfg)
            runner = _preloaded_diffueraser_runner
            if runner is None and (_scene_composites is not None):
                print("  Creating in-process DiffuEraser runner...")
                runner = InProcessDiffuEraserRunner(diffueraser_cfg)

            edited_frames, backend_info = backend.run_remove_inpaint_frames(
                frames_np, masks, str(remove_dir),
                inprocess_runner=runner,
                scene_reference_frames=_scene_composites,
                scene_overlap_masks=_scene_overlap_masks,
            )

            if runner is not None and runner is not _preloaded_diffueraser_runner:
                if hasattr(runner, 'cleanup'):
                    runner.cleanup()
                del runner
                _free_gpu_after_inpainting()
        
        frame_infos = [{"backend": edit_backend, "frame_idx": i} for i in range(len(edited_frames))]

    elif edit_backend == BACKEND_TELEA:
        if is_multi_group:
            current_frames = frames_np
            for g_idx, gid in enumerate(segmentation.object_ids):
                group_masks = [
                    segmentation.get_mask(i, gid) for i in range(len(frames))
                ]
                has_content = any(m.any() for m in group_masks)
                if not has_content:
                    continue
                group_dir = str(remove_dir / f"group_{gid}")
                Path(group_dir).mkdir(exist_ok=True)
                print(f"  Group {gid}/{num_groups}: Telea inpainting...")
                current_frames, _ = run_telea_fallback(
                    current_frames, group_masks, group_dir, radius=5
                )
            edited_frames = current_frames
        else:
            edited_frames, backend_info = run_telea_fallback(
                frames_np, masks, str(remove_dir), radius=5
            )
        frame_infos = [{"backend": "telea", "frame_idx": i} for i in range(len(edited_frames))]

    else:
        raise ValueError(f"Unknown edit_backend: {edit_backend}")

    # Step 6: Encode edited video
    print("\n[5/5] Encoding edited video...")
    
    # Save edited frames if requested
    edited_frames_dir = str(remove_dir / "edited_frames") if edit_type == "remove" else None
    if edited_frames_dir:
        Path(edited_frames_dir).mkdir(exist_ok=True)
        for i, frame in enumerate(edited_frames):
            Image.fromarray(frame).save(os.path.join(edited_frames_dir, f"{i:06d}.png"))
    
    encode_video(edited_frames, edited_path, fps)
    print(f"  Saved to: {edited_path}")
    
    
    # Log clip-level info
    logger.log_clip(
        video_id=video_id,
        input_video_path=input_video_path,
        text_prompt=text_prompt,
        edit_type=edit_type,
        swap_target=swap_target,
        sam3_num_objects=len(segmentation.object_ids),
        selected_object_id=segmentation.selected_object_id,
        clip_seed=clip_seed,
        num_frames=len(frames),
        mean_mask_area_frac=segmentation.mean_mask_area_frac,
        overlay_video_path=overlay_path,
        edited_video_path=edited_path,
        backend=edit_backend,
    )
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Overlay video: {overlay_path}")
    print(f"Edited video:  {edited_path}")
    print(f"Masks:         {masks_dir}")
    print(f"Logs:          {log_path}")
    print("=" * 60)
    
    return {
        "video_id": video_id,
        "overlay_video": overlay_path,
        "edited_video": edited_path,
        "masks_dir": str(masks_dir),
        "log_path": log_path,
        "segmentation": segmentation,
    }


def run_batch(
    video_paths: List[str],
    text_prompt: str,
    edit_type: str,
    output_root: str,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Process multiple videos with models loaded once.

    Loads SAM3 once; preloads DiffuEraser in-process only when
    ``edit_backend`` is ``diffueraser``.  Then iterates over videos.
    """
    from sam3_segmenter import SAM3VideoSegmenter

    sam3_model_id = kwargs.pop("sam3_model_id", "facebook/sam3")
    edit_backend = kwargs.get("edit_backend", BACKEND_DIFFUERASER)

    print("=" * 60)
    print(f"Batch mode: {len(video_paths)} videos")
    print("=" * 60)

    # --- Pre-load SAM3 ---
    print("\n[Batch] Loading SAM3 model (once) ...")
    segmenter = SAM3VideoSegmenter(model_id=sam3_model_id)
    segmenter.load_model()

    # --- Pre-load DiffuEraser in-process (if applicable) ---
    runner: Optional[InProcessDiffuEraserRunner] = None
    if edit_backend == BACKEND_DIFFUERASER:
        diffueraser_path = kwargs.get("diffueraser_path", "third_party/DiffuEraser")
        weights_path = os.path.join(diffueraser_path, "weights")
        cfg = DiffuEraserConfig(
            diffueraser_path=diffueraser_path,
            weights_path=weights_path,
            device=kwargs.get("diffueraser_device", "auto"),
        )
        runner = InProcessDiffuEraserRunner(cfg)
        try:
            runner.load()
        except Exception as exc:
            print(f"[Batch] WARNING: in-process DiffuEraser load failed ({exc}); "
                  "falling back to subprocess per video.")
            runner = None

    # --- Pre-load DINOv3 tracker (only if verification requested) ---
    dino_tracker = None
    if kwargs.get("dino_verify", False):
        from dino_tracker import DINOv3Tracker
        print("\n[Batch] Loading DINOv3 tracker for verification ...")
        dino_tracker = DINOv3Tracker()
        dino_tracker.load_model()

    print(f"\n[Batch] Models ready. Processing {len(video_paths)} videos ...\n")

    results: List[Dict[str, Any]] = []
    for idx, vpath in enumerate(video_paths, 1):
        vname = os.path.basename(vpath)
        out_dir = os.path.join(output_root, Path(vpath).stem)
        print(f"\n{'='*60}")
        print(f"[Batch {idx}/{len(video_paths)}] {vname}")
        print(f"{'='*60}")

        try:
            r = run_pipeline(
                input_video_path=vpath,
                text_prompt=text_prompt,
                edit_type=edit_type,
                output_dir=out_dir,
                sam3_model_id=sam3_model_id,
                _preloaded_segmenter=segmenter,
                _preloaded_diffueraser_runner=runner,
                _preloaded_dino_tracker=dino_tracker,
                **kwargs,
            )
            results.append(r)
        except Exception as exc:
            print(f"[Batch] FAILED on {vname}: {exc}")
            results.append({"video": vpath, "error": str(exc)})

    passed = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*60}")
    print(f"Batch complete: {passed}/{len(video_paths)} succeeded")
    print(f"{'='*60}")
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Counterfactual Video Editor - SAM 3 + Inpainting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove object (default: DiffuEraser)
  python -m pipeline --video input.mp4 --prompt "dog" --edit remove --out runs/run1

  # DiffuEraser (BrushNet + ProPainter)
  python -m pipeline --video input.mp4 --prompt "dog" --edit remove --edit_backend diffueraser --out runs/run2

  # Swap object
  python -m pipeline --video input.mp4 --prompt "dog" --edit swap --swap_target "cat" --out runs/run3
  
  # Multi-prompt: segment dog and leash separately, merge if close
  python -m pipeline --video input.mp4 --prompt "dog,leash" --edit remove --edit_backend diffueraser --out runs/run4

  # Quick test with limited frames
  python -m pipeline --video input.mp4 --prompt "ball" --edit remove --out test_run --max-frames 10
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--video", "-v",
        required=True,
        nargs="+",
        help="Path to input video file(s). Pass multiple for batch mode."
    )
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Text prompt(s) for segmentation. Comma-separated for multiple (e.g., 'dog,leash')"
    )
    parser.add_argument(
        "--edit", "-e",
        required=True,
        choices=["remove", "swap"],
        help="Edit type: 'remove' or 'swap'"
    )
    parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output directory path"
    )
    
    # Backend selection (--edit-backend is an alias; batch_runner uses hyphens)
    parser.add_argument(
        "--edit_backend", "--edit-backend",
        default=BACKEND_DIFFUERASER,
        choices=list(ALL_BACKENDS),
        help=f"{BACKEND_DIFFUERASER} (default) or {BACKEND_TELEA} (cv2 fallback)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--swap_target", "-t",
        default=None,
        help="Target object for swap mode (e.g., 'cat')"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames (for testing)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Resample video to target FPS (e.g., 10 or 15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Mask overlay opacity 0-1 (default: 0.5)"
    )
    parser.add_argument(
        "--sam3-model",
        default="facebook/sam3",
        help="SAM 3 model ID (default: facebook/sam3)"
    )
    parser.add_argument(
        "--skip-inpaint",
        action="store_true",
        help="Only run segmentation and overlay (no inpainting)"
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip post-processing steps (output raw inpainting result)"
    )
    parser.add_argument(
        "--reuse-masks",
        default=None,
        metavar="MASKS_DIR",
        help=(
            "Path to a previously saved masks/ directory to skip SAM 3. "
            "Example: runs/test/masks"
        )
    )
    
    # SAM3 video tracker + optional DINOv3 verification
    parser.add_argument(
        "--sam3-video-tracker",
        "--use-dino-tracker",
        dest="use_dino_tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable SAM3 Tracker Video (recommended): text-search an anchor frame, "
            "then propagate the mask through all frames with SAM3's video tracker "
            "(avoids per-frame misses / attention drift). "
            "Without this flag, SAM3 runs independent text segmentation per frame "
            "and uses an empty mask on any frame with no detection. "
            "Alias: --use-dino-tracker (name is historical; DINO is optional via "
            "--dino-verify only)."
        ),
    )
    parser.add_argument(
        "--dino-verify",
        action="store_true",
        default=False,
        help=(
            "Enable DINOv3 verification pass after SAM3 Tracker Video propagation. "
            "Flags and zeros frames where the tracked mask doesn't overlap with "
            "the DINOv3 target signature. Requires --sam3-video-tracker. "
            "Adds ~3s per 100 frames."
        ),
    )

    # Multi-object options
    parser.add_argument(
        "--multi-object",
        action="store_true",
        default=False,
        help=(
            "Keep all detected instances. Nearby masks are merged into one "
            "group (single inpainting run); distant masks become separate "
            "groups inpainted sequentially."
        ),
    )
    parser.add_argument(
        "--merge-proximity-px",
        type=int,
        default=50,
        help=(
            "Max pixel gap for two masks to be merged into one group "
            "(only used with --multi-object, default: 50)"
        ),
    )
    
    # DiffuEraser options
    parser.add_argument(
        "--diffueraser-path",
        default="third_party/DiffuEraser",
        help="Path to DiffuEraser repo"
    )
    parser.add_argument(
        "--diffueraser-device",
        default="auto",
        help=(
            "Device for DiffuEraser inference. "
            "'auto' picks CUDA > MPS > CPU (default). "
            "'cpu' always uses CPU (slow but stable). "
            "'mps' or 'cuda' forces that device with no fallback."
        )
    )
    parser.add_argument(
        "--mask_erode_px",
        type=int,
        default=0,
        help="Mask erosion in pixels (default: 0)"
    )
    parser.add_argument(
        "--mask_dilate_px",
        type=int,
        default=0,
        help="Mask dilation in pixels (default: 0)"
    )
    parser.add_argument(
        "--mask_feather_sigma",
        type=float,
        default=0.0,
        help="Mask feather sigma (default: 0)"
    )
    









    args = parser.parse_args()
    
    # Validate swap mode
    if args.edit == "swap" and not args.swap_target:
        parser.error("--swap_target is required when --edit=swap")

    shared_kwargs = dict(
        swap_target=args.swap_target,
        max_frames=args.max_frames,
        target_fps=args.fps,
        clip_seed=args.seed,
        overlay_alpha=args.overlay_alpha,
        sam3_model_id=args.sam3_model,
        skip_inpaint=args.skip_inpaint,
        skip_postprocess=args.skip_postprocess,
        reuse_masks_dir=args.reuse_masks,
        edit_backend=args.edit_backend,
        multi_object=args.multi_object,
        merge_proximity_px=args.merge_proximity_px,
        diffueraser_path=args.diffueraser_path,
        diffueraser_device=args.diffueraser_device,
        mask_erode_px=args.mask_erode_px,
        mask_dilate_px=args.mask_dilate_px,
        mask_feather_sigma=args.mask_feather_sigma,
        use_dino_tracker=args.use_dino_tracker,
        dino_verify=args.dino_verify,
    )

    is_batch = len(args.video) > 1

    try:
        if is_batch:
            run_batch(
                video_paths=args.video,
                text_prompt=args.prompt,
                edit_type=args.edit,
                output_root=args.out,
                **shared_kwargs,
            )
        else:
            run_pipeline(
                input_video_path=args.video[0],
                text_prompt=args.prompt,
                edit_type=args.edit,
                output_dir=args.out,
                **shared_kwargs,
            )
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
