"""
diffueraser_backend.py - DiffuEraser Video Inpainting Backend

Uses DiffuEraser for temporally consistent object removal:
- Deterministic neutral prefill + diffusion polish
- Temporal consistency across frames
- No hallucinated objects

Reference: 
- GitHub: https://github.com/lixiaowen-xw/DiffuEraser
- HuggingFace: https://huggingface.co/lixiaowen/diffuEraser
- Paper: https://arxiv.org/abs/2501.10018
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from tqdm import tqdm


# Default paths for DiffuEraser
DEFAULT_DIFFUERASER_PATH = "third_party/DiffuEraser"
DEFAULT_WEIGHTS_PATH = "third_party/DiffuEraser/weights"

# HuggingFace model ID
HUGGINGFACE_MODEL_ID = "lixiaowen/diffuEraser"

# Required weight folders according to DiffuEraser README
REQUIRED_WEIGHTS = [
    "diffuEraser",  # Main model weights (brushnet + unet_main)
    "stable-diffusion-v1-5",  # Base SD model
    "PCM_Weights",  # PCM weights
    "propainter",  # ProPainter prior model
    "sd-vae-ft-mse",  # VAE weights
]


@dataclass
class DiffuEraserConfig:
    """Configuration for DiffuEraser inference."""
    # Mask preprocessing
    mask_erode_px: int = 0
    mask_dilate_px: int = 0
    mask_feather_sigma: float = 0.0

    # Device selection: "auto", "cpu", "mps", "cuda"
    # "auto" picks CUDA > MPS > CPU in priority order
    device: str = "auto"

    # Paths
    diffueraser_path: str = DEFAULT_DIFFUERASER_PATH
    weights_path: str = DEFAULT_WEIGHTS_PATH

    # Resolution (DiffuEraser supports 1280x720, 960x540, 640x360)
    # Lower resolution = less VRAM, faster inference
    target_resolution: Optional[Tuple[int, int]] = None  # None = keep original

    # Background locking: overwrite unmasked latents each denoising step
    lock_background: bool = False

    # KV-Lock temporal stabilization
    kvlock: bool = False
    kvlock_alpha: float = 0.6
    kvlock_max_alpha: float = 0.9
    kvlock_dynamic_cfg: bool = True

    # Debug
    save_debug_frames: bool = True

    @classmethod
    def default(cls) -> "DiffuEraserConfig":
        return cls()


def download_weights_from_huggingface(
    weights_path: str,
    model_id: str = HUGGINGFACE_MODEL_ID,
) -> bool:
    """
    Download DiffuEraser weights from HuggingFace.
    
    Args:
        weights_path: Where to save weights
        model_id: HuggingFace model ID
        
    Returns:
        True if successful
    """
    try:
        from huggingface_hub import snapshot_download
        
        print(f"Downloading DiffuEraser weights from HuggingFace ({model_id})...")
        
        # Download the diffuEraser folder
        snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(weights_path, "diffuEraser"),
            local_dir_use_symlinks=False,
        )
        
        print(f"Weights downloaded to: {weights_path}/diffuEraser")
        return True
        
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"Failed to download from HuggingFace: {e}")
        return False


def check_diffueraser_available(cfg: DiffuEraserConfig) -> Tuple[bool, str]:
    """
    Check if DiffuEraser repo and checkpoints are available.
    
    Returns:
        (available, message): Tuple of availability status and message
    """
    # Use absolute paths
    diffueraser_path = Path(cfg.diffueraser_path).resolve()
    weights_path = Path(cfg.weights_path).resolve()
    
    if not diffueraser_path.exists():
        return False, f"""
DiffuEraser not found at: {diffueraser_path.absolute()}

To set up DiffuEraser:

1. Clone the repo:
   git clone https://github.com/lixiaowen-xw/DiffuEraser.git {diffueraser_path}

2. Create conda environment:
   conda create -n diffueraser python=3.9.19
   conda activate diffueraser
   cd {diffueraser_path}
   pip install -r requirements.txt

3. Download weights (see step 4 below)
"""
    
    # Check for inference script
    inference_script = diffueraser_path / "run_diffueraser.py"
    if not inference_script.exists():
        return False, f"DiffuEraser inference script not found at: {inference_script}"
    
    # Check for weights directory
    if not weights_path.exists():
        return False, f"""
DiffuEraser weights not found at: {weights_path.absolute()}

Download weights:

Option A - From HuggingFace (recommended):
   pip install huggingface_hub
   python -c "from diffueraser_backend import download_weights_from_huggingface; download_weights_from_huggingface('{weights_path}')"
   
   Or visit: https://huggingface.co/lixiaowen/diffuEraser

Option B - From ModelScope:
   Visit: https://modelscope.cn/models/lixiaowen/diffuEraser

Required folder structure:
   {weights_path}/
   ├── diffuEraser/
   │   ├── brushnet/
   │   └── unet_main/
   ├── stable-diffusion-v1-5/
   ├── PCM_Weights/
   ├── propainter/
   └── sd-vae-ft-mse/
"""
    
    # Check for required weight folders
    missing = []
    for folder in REQUIRED_WEIGHTS:
        if not (weights_path / folder).exists():
            missing.append(folder)
    
    if missing:
        return False, f"""
Missing required weight folders in {weights_path}:
  - {chr(10).join('  - ' + m for m in missing)}

Download from:
  - HuggingFace: https://huggingface.co/lixiaowen/diffuEraser
  - ModelScope: https://modelscope.cn/models/lixiaowen/diffuEraser

Additional required models:
  - stable-diffusion-v1-5: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
  - PCM_Weights: https://huggingface.co/wangfuyun/PCM_Weights
  - propainter: https://github.com/sczhou/ProPainter
  - sd-vae-ft-mse: https://huggingface.co/stabilityai/sd-vae-ft-mse
"""
    
    return True, "DiffuEraser is available"


def preprocess_masks(
    masks: List[np.ndarray],
    erode_px: int = 0,
    dilate_px: int = 0,
    feather_sigma: float = 0.0,
) -> List[np.ndarray]:
    """
    Preprocess masks with morphological operations.
    
    Args:
        masks: List of binary masks (H, W)
        erode_px: Erosion radius
        dilate_px: Dilation radius
        feather_sigma: Gaussian blur sigma for feathering
        
    Returns:
        Processed masks (0/255 uint8)
    """
    processed = []
    
    for mask in masks:
        m = mask.astype(np.float32)
        
        # Normalize to 0-1 if needed
        if m.max() > 1:
            m = m / 255.0
        
        # Erode
        if erode_px > 0:
            struct = np.ones((erode_px * 2 + 1, erode_px * 2 + 1))
            m = binary_erosion(m > 0.5, structure=struct).astype(np.float32)
        
        # Dilate
        if dilate_px > 0:
            struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1))
            m = binary_dilation(m > 0.5, structure=struct).astype(np.float32)
        
        # Feather (blur)
        if feather_sigma > 0:
            m = gaussian_filter(m, sigma=feather_sigma)
        
        # Binarize for DiffuEraser (0 or 255)
        m_binary = (m > 0.5).astype(np.uint8) * 255
        processed.append(m_binary)
    
    return processed


def frames_to_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 25.0,
) -> str:
    """Convert frames to MP4 video (required format for DiffuEraser)."""
    if len(frames) == 0:
        raise ValueError("No frames to convert")
    
    height, width = frames[0].shape[:2]
    
    # Use ffmpeg for reliable encoding
    try:
        import imageio
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec='libx264',
            quality=8,
        )
        for frame in frames:
            # Convert RGB to BGR if needed, then back
            if frame.ndim == 3 and frame.shape[2] == 3:
                writer.append_data(frame)
            else:
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
        writer.close()
    except Exception:
        # Fallback to OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            if frame.ndim == 3:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(bgr)
        out.release()
    
    return output_path


def masks_to_video(
    masks: List[np.ndarray],
    output_path: str,
    fps: float = 25.0,
) -> str:
    """Convert masks to lossless MP4 video so thin structures survive encoding."""
    if len(masks) == 0:
        raise ValueError("No masks to convert")

    rgb_frames = []
    for mask in masks:
        if mask.ndim == 2:
            rgb_frames.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
        else:
            rgb_frames.append(mask)

    height, width = rgb_frames[0].shape[:2]

    try:
        import imageio
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec='libx264',
            output_params=['-crf', '0'],
        )
        for frame in rgb_frames:
            writer.append_data(frame)
        writer.close()
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in rgb_frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)
        out.release()

    return output_path


def video_to_frames(video_path: str) -> List[np.ndarray]:
    """Load video frames from MP4."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames


def run_diffueraser_inference(
    input_video_path: str,
    input_mask_path: str,
    output_dir: str,
    cfg: DiffuEraserConfig,
) -> str:
    """
    Run DiffuEraser inference via subprocess.
    
    DiffuEraser expects:
    - input_video: MP4 video file
    - input_mask: MP4 video file (same fps as input_video)
    
    Args:
        input_video_path: Path to input video (MP4)
        input_mask_path: Path to mask video (MP4)
        output_dir: Output directory
        cfg: Configuration
        
    Returns:
        Path to output video
    """
    available, msg = check_diffueraser_available(cfg)
    if not available:
        raise RuntimeError(msg)
    
    # Convert to absolute paths
    diffueraser_path = Path(cfg.diffueraser_path).resolve()
    weights_path = Path(cfg.weights_path).resolve()
    run_script = diffueraser_path / "run_diffueraser.py"
    output_dir_abs = Path(output_dir).resolve()
    
    if not run_script.exists():
        raise RuntimeError(f"DiffuEraser script not found: {run_script}")
    
    # Create output directory for DiffuEraser results
    results_dir = output_dir_abs / "diffueraser_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running DiffuEraser inference...")
    print(f"  DiffuEraser path: {diffueraser_path}")
    print(f"  Weights path: {weights_path}")
    print(f"  Input video: {input_video_path}")
    print(f"  Input mask: {input_mask_path}")
    print(f"  Output: {results_dir}")
    
    # DiffuEraser uses argparse, so we call it with command-line arguments
    cmd = [
        sys.executable,
        str(run_script),
        "--input_video", str(input_video_path),
        "--input_mask", str(input_mask_path),
        "--save_path", str(results_dir),
        "--base_model_path", str(weights_path / "stable-diffusion-v1-5"),
        "--vae_path", str(weights_path / "sd-vae-ft-mse"),
        "--diffueraser_path", str(weights_path / "diffuEraser"),
        "--propainter_model_dir", str(weights_path / "propainter"),
        "--video_length", "50",  # Max frames to process per chunk
        "--max_img_size", "512",  # Proven stable at 25 frames on MPS (test5/test6)
    ]
    
    def _build_env(force_cpu: bool) -> dict:
        e = os.environ.copy()
        e["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        e["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        if force_cpu:
            e["DIFFUERASER_FORCE_CPU"] = "1"
        else:
            e.pop("DIFFUERASER_FORCE_CPU", None)
        return e

    def _run_subprocess(env: dict) -> None:
        """Launch DiffuEraser subprocess, stream stdout+stderr live, raise on failure."""
        import threading, sys as _sys
        proc = subprocess.Popen(
            cmd,
            cwd=str(diffueraser_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        stderr_buf = []

        def _stream_stderr():
            for line in proc.stderr:
                stderr_buf.append(line)
                # tqdm writes carriage-return lines (\r); print them as-is
                _sys.stderr.write(line)
                _sys.stderr.flush()

        t = threading.Thread(target=_stream_stderr, daemon=True)
        t.start()

        for line in proc.stdout:
            print(line, end="", flush=True)

        t.join()
        try:
            proc.wait(timeout=7200)  # 2-hour cap
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError("DiffuEraser timed out after 2 hours")
        if proc.returncode != 0:
            raise RuntimeError(
                f"DiffuEraser failed (exit {proc.returncode}): "
                f"{''.join(stderr_buf)[-600:]}"
            )

    device = cfg.device.lower()
    print(f"  Device mode: {device}")

    try:
        print(f"  Command: {' '.join(cmd)}")
        if device == "cpu":
            print("  Forcing CPU (as requested).")
            _run_subprocess(_build_env(force_cpu=True))

        elif device in ("mps", "cuda"):
            print(f"  Using {device.upper()} (as requested — no CPU fallback).")
            _run_subprocess(_build_env(force_cpu=False))

        else:  # "auto"
            print("  Auto mode: trying GPU first...")
            try:
                _run_subprocess(_build_env(force_cpu=False))
                print("  GPU run succeeded.")
            except RuntimeError as gpu_err:
                err_str = str(gpu_err)
                if "exit -6" in err_str or "exit 134" in err_str or "CUDA out of memory" in err_str:
                    print(
                        "\n  WARNING: GPU crashed. "
                        "Retrying automatically on CPU — this will be slower but stable."
                    )
                    _run_subprocess(_build_env(force_cpu=True))
                else:
                    raise

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"DiffuEraser inference failed: {e}")
    
    # Find output video
    output_video = results_dir / "diffueraser_result.mp4"
    if output_video.exists():
        final_output = str(output_dir_abs / "edited.mp4")
        shutil.copy(str(output_video), final_output)
        return final_output
    
    # Try to find any mp4 in results
    output_files = list(results_dir.glob("*.mp4"))
    if output_files:
        latest = max(output_files, key=lambda p: p.stat().st_mtime)
        final_output = str(output_dir_abs / "edited.mp4")
        shutil.copy(str(latest), final_output)
        return final_output
    
    raise RuntimeError(f"DiffuEraser did not produce output in {results_dir}")


class DiffuEraserBackend:
    """
    DiffuEraser video inpainting backend.
    
    Provides temporally consistent object removal using DiffuEraser.
    
    Reference:
    - GitHub: https://github.com/lixiaowen-xw/DiffuEraser
    - HuggingFace: https://huggingface.co/lixiaowen/diffuEraser
    """
    
    def __init__(self, cfg: Optional[DiffuEraserConfig] = None):
        self.cfg = cfg or DiffuEraserConfig.default()
        self._checked = False
    
    def check_available(self) -> bool:
        """Check if DiffuEraser is available."""
        available, msg = check_diffueraser_available(self.cfg)
        if not available:
            print(msg, file=sys.stderr)
        self._checked = True
        return available
    
    def run_remove_inpaint(
        self,
        frames_dir: str,
        masks_dir: str,
        out_dir: str,
    ) -> str:
        """
        Run object removal inpainting.
        
        Args:
            frames_dir: Directory with input frames (or list of frames)
            masks_dir: Directory with masks (or list of masks)
            out_dir: Output directory
            
        Returns:
            Path to edited video or frames directory
        """
        # Load frames and masks if directories provided
        if isinstance(frames_dir, str) and os.path.isdir(frames_dir):
            frames = load_frames_from_dir(frames_dir)
        else:
            frames = frames_dir  # Assume list
        
        if isinstance(masks_dir, str) and os.path.isdir(masks_dir):
            masks = load_frames_from_dir(masks_dir)
            masks = [(m[:, :, 0] if m.ndim == 3 else m) for m in masks]
        else:
            masks = masks_dir  # Assume list
        
        return self.run_remove_inpaint_frames(frames, masks, out_dir)
    
    def run_remove_inpaint_frames(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        out_dir: str,
        fps: float = 25.0,
        inprocess_runner: Optional["InProcessDiffuEraserRunner"] = None,
        scene_reference_frames: Optional[List[np.ndarray]] = None,
        scene_overlap_masks: Optional[List[np.ndarray]] = None,
        ip_adapter_image=None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Run object removal on frame arrays.

        Args:
            frames: List of RGB frames
            masks: List of binary masks
            out_dir: Output directory
            fps: Frame rate for video conversion
            scene_reference_frames: Optional composite reference frames for Zone B guidance
            scene_overlap_masks: Optional binary overlap masks for Zone B regions

        Returns:
            (edited_frames, info): Tuple of edited frames and metadata
        """
        if not self._checked:
            if not self.check_available():
                print("DiffuEraser not available. Falling back to Telea inpainting.")
                return run_telea_fallback(frames, masks, out_dir)
        
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        info = {
            "backend": "diffueraser",
            "config": asdict(self.cfg),
            "num_frames": len(frames),
            "resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}" if frames else None,
            "timestamp": datetime.now().isoformat(),
            "github": "https://github.com/lixiaowen-xw/DiffuEraser",
            "huggingface": "https://huggingface.co/lixiaowen/diffuEraser",
        }
        
        # Preprocess masks.  DiffuEraser internally downscales to max_img_size
        # (512) and erodes by 1px at that resolution, which destroys thin mask
        # features like leashes.  A minimum 4px dilation at full res ensures
        # thin structures survive the downscale + erosion.
        effective_dilate = max(self.cfg.mask_dilate_px, 4)
        processed_masks = preprocess_masks(
            masks,
            erode_px=self.cfg.mask_erode_px,
            dilate_px=effective_dilate,
            feather_sigma=self.cfg.mask_feather_sigma,
        )
        
        # DiffuEraser requires MP4 video input, not frames
        # Convert frames to video
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = os.path.join(tmpdir, "input.mp4")
            mask_video = os.path.join(tmpdir, "mask.mp4")
            
            print("Converting frames to video for DiffuEraser...")
            frames_to_video(frames, input_video, fps)
            masks_to_video(processed_masks, mask_video, fps)
            
            # Run inference
            try:
                if inprocess_runner is not None:
                    output_video = inprocess_runner.run(
                        input_video, mask_video, out_dir,
                        lock_background=self.cfg.lock_background,
                        scene_reference_frames=scene_reference_frames,
                        scene_overlap_masks=scene_overlap_masks,
                        ip_adapter_image=ip_adapter_image,
                    )
                else:
                    output_video = run_diffueraser_inference(
                        input_video, mask_video, out_dir, self.cfg
                    )
                
                # Load output frames
                edited_frames = video_to_frames(output_video)
                
                # Resize back to original resolution if needed
                original_h, original_w = frames[0].shape[:2]
                output_h, output_w = edited_frames[0].shape[:2]
                
                if (output_h, output_w) != (original_h, original_w):
                    print(f"Resizing DiffuEraser output from {output_w}x{output_h} to {original_w}x{original_h}...")
                    edited_frames = [
                        cv2.resize(f, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
                        for f in edited_frames
                    ]
                    info["resized_from"] = f"{output_w}x{output_h}"
                
                info["method"] = "diffueraser" + (" (in-process)" if inprocess_runner else "")
                info["output_video"] = output_video
                
            except Exception as e:
                print(f"DiffuEraser failed: {e}")
                print("Falling back to Telea inpainting...")
                edited_frames, fallback_info = run_telea_fallback(frames, masks, out_dir)
                info["method"] = "telea_fallback"
                info["fallback_reason"] = str(e)
        
        # Save info
        info_path = os.path.join(out_dir, "log.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        return edited_frames, info


def load_frames_from_dir(frames_dir: str) -> List[np.ndarray]:
    """Load frames from a directory."""
    frames = []
    
    # Get sorted list of frame files
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    for fname in frame_files:
        fpath = os.path.join(frames_dir, fname)
        frame = np.array(Image.open(fpath).convert("RGB"))
        frames.append(frame)
    
    return frames


# ============================================================================
# Fallback: Telea + Low-Strength Diffusion (when DiffuEraser unavailable)
# ============================================================================

def telea_inpaint_frame(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    radius: int = 5,
) -> np.ndarray:
    """
    Telea inpainting for single frame.
    
    Args:
        frame_rgb: RGB frame (H, W, 3)
        mask: Binary mask (H, W)
        radius: Inpaint radius
        
    Returns:
        Inpainted frame
    """
    # Ensure mask is uint8 0/255
    if mask.max() <= 1:
        mask_u8 = (mask.astype(np.uint8) * 255)
    else:
        mask_u8 = mask.astype(np.uint8)
    
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(frame_bgr, mask_u8, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def run_telea_fallback(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    out_dir: str,
    radius: int = 5,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Fallback inpainting using Telea algorithm.
    
    This is fast but may produce blurry results.
    
    Args:
        frames: List of RGB frames
        masks: List of binary masks
        out_dir: Output directory
        radius: Inpaint radius
        
    Returns:
        (edited_frames, info)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    edited_frames = []
    for frame, mask in tqdm(zip(frames, masks), total=len(frames), desc="Telea inpaint"):
        edited = telea_inpaint_frame(frame, mask, radius)
        edited_frames.append(edited)
    
    info = {
        "backend": "telea_fallback",
        "radius": radius,
        "num_frames": len(frames),
        "timestamp": datetime.now().isoformat(),
        "note": "Using Telea fallback. For better results, install DiffuEraser.",
    }
    
    # Save info
    info_path = os.path.join(out_dir, "log.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    return edited_frames, info


# ============================================================================
# Swap placeholder (not implemented yet)
# ============================================================================

def run_swap_stub(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    target_concept: str,
    out_dir: str,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Placeholder for object swap using video inpainting + insertion.
    
    This will be implemented in Phase 2 using:
    1. DiffuEraser for removal
    2. VideoPainter/CoCoCo for insertion
    
    Args:
        frames: List of RGB frames
        masks: List of binary masks
        target_concept: What to replace with
        out_dir: Output directory
        
    Returns:
        Not implemented
    """
    raise NotImplementedError(
        "Swap mode is not yet implemented for DiffuEraser backend.\n"
        "Phase 2 will implement this as a two-step process:\n"
        "  1. Remove object with DiffuEraser\n"
        "  2. Insert new object with VideoPainter/CoCoCo\n\n"
        "For now, use the diffusion backend (--edit_backend diffusion) for swap mode."
    )


# ============================================================================
# Convenience function to download all weights
# ============================================================================

def setup_diffueraser(
    install_path: str = DEFAULT_DIFFUERASER_PATH,
    download_weights: bool = True,
) -> None:
    """
    Setup DiffuEraser with instructions.
    
    Args:
        install_path: Where to clone DiffuEraser
        download_weights: Whether to download weights from HuggingFace
    """
    print("=" * 60)
    print("DiffuEraser Setup")
    print("=" * 60)
    
    diffueraser_path = Path(install_path)
    
    if not diffueraser_path.exists():
        print(f"""
1. Clone DiffuEraser:
   git clone https://github.com/lixiaowen-xw/DiffuEraser.git {install_path}

2. Create conda environment:
   conda create -n diffueraser python=3.9.19
   conda activate diffueraser
   cd {install_path}
   pip install -r requirements.txt
""")
    
    weights_path = diffueraser_path / "weights"
    
    print(f"""
3. Download weights to {weights_path}/

   Option A - DiffuEraser weights from HuggingFace:
   https://huggingface.co/lixiaowen/diffuEraser
   
   Option B - DiffuEraser weights from ModelScope:
   https://modelscope.cn/models/lixiaowen/diffuEraser

4. Download additional required models:
   
   stable-diffusion-v1-5 (4GB minimal):
   https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
   Download: feature_extractor, model_index.json, safety_checker, scheduler, text_encoder, tokenizer
   
   PCM_Weights:
   https://huggingface.co/wangfuyun/PCM_Weights
   
   ProPainter:
   https://github.com/sczhou/ProPainter/releases
   
   sd-vae-ft-mse:
   https://huggingface.co/stabilityai/sd-vae-ft-mse

5. Final structure:
   {weights_path}/
   ├── diffuEraser/
   │   ├── brushnet/
   │   └── unet_main/
   ├── stable-diffusion-v1-5/
   ├── PCM_Weights/
   ├── propainter/
   └── sd-vae-ft-mse/
""")
    print("=" * 60)


class InProcessDiffuEraserRunner:
    """
    Keeps DiffuEraser + ProPainter models loaded on GPU across calls.

    Use this for batch workloads to avoid re-loading ~4 GB of weights per video.
    The single-video subprocess path remains the default for one-off runs.
    """

    def __init__(self, cfg: DiffuEraserConfig):
        self.cfg = cfg
        self._diffueraser = None
        self._propainter = None
        self._device = None
        self._loaded = False

    def offload_to_cpu(self) -> None:
        """Move all models to CPU to free GPU for SAM3 segmentation."""
        if not self._loaded:
            return
        import torch, gc
        if hasattr(self._diffueraser, "pipeline"):
            self._diffueraser.pipeline.to("cpu")
        if self._propainter is not None:
            for _name, val in vars(self._propainter).items():
                if isinstance(val, torch.nn.Module):
                    val.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[InProcess] DiffuEraser offloaded to CPU.")

    def reload_to_gpu(self) -> None:
        """Move all models back to GPU for inpainting."""
        if not self._loaded or self._device is None:
            return
        import torch
        if hasattr(self._diffueraser, "pipeline"):
            self._diffueraser.pipeline.to(self._device)
        if self._propainter is not None:
            for _name, val in vars(self._propainter).items():
                if isinstance(val, torch.nn.Module):
                    val.to(self._device)
        print(f"[InProcess] DiffuEraser reloaded to {self._device}.")

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        diffueraser_path = Path(self.cfg.diffueraser_path).resolve()
        weights_path = Path(self.cfg.weights_path).resolve()

        sys.path.insert(0, str(diffueraser_path))
        saved_cwd = os.getcwd()
        os.chdir(str(diffueraser_path))

        try:
            from propainter.inference import Propainter
            from propainter.model.misc import get_device
            from diffueraser.diffueraser import DiffuEraser as _DiffuEraser

            self._device = get_device()
            print(f"[InProcess] Loading DiffuEraser + ProPainter on {self._device} ...")

            self._diffueraser = _DiffuEraser(
                self._device,
                str(weights_path / "stable-diffusion-v1-5"),
                str(weights_path / "sd-vae-ft-mse"),
                str(weights_path / "diffuEraser"),
                ckpt="2-Step",
            )
            self._propainter = Propainter(
                str(weights_path / "propainter"),
                device=self._device,
            )
            self._loaded = True
            print("[InProcess] Models loaded and ready for reuse.")
        finally:
            os.chdir(saved_cwd)

    def _prepare_zone_b(
        self,
        scene_reference_frames: List[np.ndarray],
        scene_overlap_masks: List[np.ndarray],
        max_img_size: int = 512,
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """
        Preprocess scene reference frames and overlap masks for Zone B
        latent locking in DiffuEraser.

        Resizes to DiffuEraser working resolution, VAE-encodes reference
        frames, and downsamples masks to latent space (8x).

        Returns:
            (zone_b_latents, zone_b_mask) or (None, None) if insufficient data.
        """
        import torch
        from PIL import Image
        from diffusers.image_processor import VaeImageProcessor

        n = len(scene_reference_frames)
        if n == 0 or len(scene_overlap_masks) == 0:
            return None, None

        n = min(n, len(scene_overlap_masks))

        # Determine DiffuEraser working resolution: match what forward() computes
        # from max_img_size. Use the reference frame dimensions as source.
        src_h, src_w = scene_reference_frames[0].shape[:2]
        max_dim = max(src_w, src_h)
        if max_dim > max_img_size:
            ratio = max_dim / max_img_size
            tar_w = int(src_w / ratio)
            tar_h = int(src_h / ratio)
        else:
            tar_w, tar_h = src_w, src_h
        # Round to multiples of 8 (required by SD 1.5 VAE)
        tar_w = tar_w - tar_w % 8
        tar_h = tar_h - tar_h % 8

        device = self._device or "cuda"
        vae = self._diffueraser.vae
        image_processor = self._diffueraser.image_processor

        # Encode reference frames through VAE
        ref_tensors = []
        for i in range(n):
            img = Image.fromarray(scene_reference_frames[i]).resize((tar_w, tar_h))
            t = image_processor.preprocess(img, height=tar_h, width=tar_w)
            t = t.to(device=device, dtype=torch.float16)
            ref_tensors.append(t)
        ref_cat = torch.cat(ref_tensors, dim=0)  # (N, 3, H, W)

        with torch.no_grad():
            enc_chunks = []
            for i in range(0, ref_cat.shape[0], 4):
                enc_chunks.append(
                    vae.encode(ref_cat[i : i + 4]).latent_dist.sample()
                )
            zone_b_latents = torch.cat(enc_chunks, dim=0)
        zone_b_latents = zone_b_latents * vae.config.scaling_factor  # (N, 4, H/8, W/8)

        # Downsample overlap masks to latent space (8x spatial reduction)
        lat_h, lat_w = zone_b_latents.shape[2], zone_b_latents.shape[3]
        mask_tensors = []
        for i in range(n):
            m = scene_overlap_masks[i]
            if m.dtype == bool:
                m = m.astype(np.uint8) * 255
            if m.ndim == 3:
                m = m[:, :, 0]
            # Resize to working resolution, then downsample to latent
            m_resized = cv2.resize(m, (tar_w, tar_h), interpolation=cv2.INTER_NEAREST)
            m_lat = cv2.resize(m_resized, (lat_w, lat_h), interpolation=cv2.INTER_NEAREST)
            m_f = (m_lat.astype(np.float32) / 255.0) if m_lat.max() > 1 else m_lat.astype(np.float32)
            mask_tensors.append(
                torch.from_numpy(m_f).unsqueeze(0)  # (1, H_lat, W_lat)
            )
        zone_b_mask = torch.stack(mask_tensors, dim=0)  # (N, 1, H_lat, W_lat)
        zone_b_mask = zone_b_mask.to(device=device, dtype=torch.float16)

        print(f"[Zone B] Prepared: latents {zone_b_latents.shape}, mask {zone_b_mask.shape}")
        return zone_b_latents, zone_b_mask

    def load_ip_adapter(self, scale: float = 0.4):
        """Load IP-Adapter onto the DiffuEraser pipeline. Reversible."""
        self.load()
        self._diffueraser.load_ip_adapter(scale=scale)

    def unload_ip_adapter(self):
        """Remove IP-Adapter and restore original attention processors."""
        if self._diffueraser is not None:
            self._diffueraser.unload_ip_adapter()

    def run(
        self,
        input_video_path: str,
        input_mask_path: str,
        output_dir: str,
        video_length: int = 50,
        max_img_size: int = 512,
        mask_dilation_iter: int = 8,
        lock_background: bool = False,
        scene_reference_frames: Optional[List[np.ndarray]] = None,
        scene_overlap_masks: Optional[List[np.ndarray]] = None,
        ip_adapter_image=None,
    ) -> str:
        """Run DiffuEraser on one video, reusing loaded models."""
        self.load()

        import torch, time

        diffueraser_path = Path(self.cfg.diffueraser_path).resolve()
        saved_cwd = os.getcwd()
        os.chdir(str(diffueraser_path))

        # KV-Lock setup
        kvlock_ctrl = None
        original_processors = None
        if self.cfg.kvlock:
            from inference_conditioned.diffueraser_kvlock import (
                DiffuEraserKVLockConfig,
                DiffuEraserKVLockController,
                install_kvlock_processors,
                uninstall_kvlock_processors,
            )
            kvlock_cfg = DiffuEraserKVLockConfig(
                base_alpha=self.cfg.kvlock_alpha,
                max_alpha=self.cfg.kvlock_max_alpha,
                enable_dynamic_cfg=self.cfg.kvlock_dynamic_cfg,
            )
            kvlock_ctrl = DiffuEraserKVLockController(kvlock_cfg)
            unet = self._diffueraser.pipeline.unet
            original_processors = install_kvlock_processors(unet, kvlock_ctrl, kvlock_cfg)
            n_wrapped = len(kvlock_ctrl._processors)
            print(f"[KV-Lock] Installed on {n_wrapped} attention layers.")

        try:
            results_dir = Path(output_dir).resolve() / "diffueraser_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            priori_path = str(results_dir / "priori.mp4")
            output_path = str(results_dir / "diffueraser_result.mp4")

            start = time.time()

            self._propainter.forward(
                input_video_path, input_mask_path, priori_path,
                video_length=video_length,
                ref_stride=10, neighbor_length=10, subvideo_length=50,
                mask_dilation=mask_dilation_iter,
            )

            # Zone B: preprocess scene reference frames and overlap masks
            # Skip Zone B when IP-Adapter is active — IP-Adapter handles
            # identity conditioning through cross-attention, not latent blending.
            zb_latents = None
            zb_mask = None
            if (scene_reference_frames is not None
                    and scene_overlap_masks is not None
                    and ip_adapter_image is None):
                zb_latents, zb_mask = self._prepare_zone_b(
                    scene_reference_frames, scene_overlap_masks, max_img_size,
                )

            self._diffueraser.forward(
                input_video_path, input_mask_path, priori_path, output_path,
                max_img_size=max_img_size,
                video_length=video_length,
                mask_dilation_iter=mask_dilation_iter,
                guidance_scale=None,
                lock_background=lock_background,
                kvlock_controller=kvlock_ctrl,
                zone_b_latents=zb_latents,
                zone_b_mask=zb_mask,
                ip_adapter_image=ip_adapter_image,
            )

            elapsed = time.time() - start
            print(f"[InProcess] DiffuEraser finished in {elapsed:.1f}s")
            torch.cuda.empty_cache()

            if os.path.exists(output_path):
                return output_path
            raise RuntimeError(f"DiffuEraser did not produce output at {output_path}")
        finally:
            if original_processors is not None:
                from inference_conditioned.diffueraser_kvlock import uninstall_kvlock_processors
                uninstall_kvlock_processors(self._diffueraser.pipeline.unet, original_processors)
            os.chdir(saved_cwd)


if __name__ == "__main__":
    # Run setup instructions when called directly
    setup_diffueraser()
