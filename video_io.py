"""
video_io.py - Video decode/encode utilities

Handles:
- Decoding video → RGB frames (list of numpy arrays or PIL Images)
- Encoding RGB frames → mp4
- Stable frame indices for reproducibility
"""

import os
from typing import List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


def decode_video(
    video_path: str,
    max_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
    uniform_subsample: bool = False,
) -> Tuple[List[np.ndarray], float, int, int]:
    """
    Decode video to list of RGB frames.
    
    Args:
        video_path: Path to input video file
        max_frames: Optional limit on number of frames to extract
        target_fps: Optional target FPS (if None, uses source FPS)
        uniform_subsample: When True and max_frames is set, pick evenly-spaced
            frame indices across the full video duration instead of taking the
            first N frames. Returns source_fps so the output encodes at the
            original temporal rate (resulting in a shorter video).
        
    Returns:
        frames: List of numpy arrays (H, W, 3) in RGB format
        fps: Frame rate of the video
        width: Video width
        height: Video height
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Uniform subsampling path: seek to evenly-spaced indices ---
    if uniform_subsample and max_frames is not None and total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"No frames extracted from video: {video_path}")
        return frames, source_fps, width, height

    # --- Sequential sampling path (original behaviour) ---
    fps = target_fps if target_fps is not None else source_fps
    
    if target_fps is not None and target_fps != source_fps:
        frame_interval = source_fps / target_fps
    else:
        frame_interval = 1.0
    
    frames = []
    frame_idx = 0
    next_sample_idx = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= next_sample_idx:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            next_sample_idx += frame_interval
            
            if max_frames is not None and len(frames) >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    
    return frames, fps, width, height


def decode_video_to_pil(
    video_path: str,
    max_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
    uniform_subsample: bool = False,
) -> Tuple[List[Image.Image], float, int, int]:
    """
    Decode video to list of PIL Images (RGB).
    
    Args:
        video_path: Path to input video file
        max_frames: Optional limit on number of frames
        target_fps: Optional target FPS
        uniform_subsample: When True, evenly space frames across the full
            video rather than taking the first N.
        
    Returns:
        frames: List of PIL Images in RGB format
        fps: Frame rate
        width: Video width
        height: Video height
    """
    frames_np, fps, width, height = decode_video(
        video_path, max_frames, target_fps, uniform_subsample=uniform_subsample,
    )
    frames_pil = [Image.fromarray(f) for f in frames_np]
    return frames_pil, fps, width, height


def encode_video(
    frames: List[Union[np.ndarray, Image.Image]],
    output_path: str,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> str:
    """
    Encode list of frames to mp4 video.
    
    Args:
        frames: List of frames (numpy RGB or PIL Images)
        output_path: Output video path
        fps: Frame rate
        codec: Video codec (default: mp4v)
        
    Returns:
        output_path: Path to saved video
    """
    if len(frames) == 0:
        raise ValueError("No frames to encode")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert first frame to get dimensions
    first_frame = frames[0]
    if isinstance(first_frame, Image.Image):
        first_frame = np.array(first_frame)
    
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
    
    for frame in frames:
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    return output_path


def save_frames(
    frames: List[Union[np.ndarray, Image.Image]],
    output_dir: str,
    prefix: str = "frame",
    format: str = "png",
) -> List[str]:
    """
    Save frames as individual image files.
    
    Args:
        frames: List of frames
        output_dir: Output directory
        prefix: Filename prefix
        format: Image format (png, jpg)
        
    Returns:
        List of saved file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for idx, frame in enumerate(frames):
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        
        filename = f"{prefix}_{idx:06d}.{format}"
        filepath = os.path.join(output_dir, filename)
        frame.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata without decoding all frames.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": frame_count,
        "duration_sec": frame_count / fps if fps > 0 else 0,
    }
    
    cap.release()
    return info
