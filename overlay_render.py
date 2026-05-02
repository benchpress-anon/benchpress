"""
overlay_render.py - Mask Overlay Visualization

Renders colored mask overlays on video frames:
- Deterministic object_id → color mapping
- Alpha blending for visualization
- Produces overlay.mp4 for mask review
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image

from video_io import encode_video
from sam3_segmenter import SegmentationResult


# Deterministic color palette (no RNG)
# Using a fixed set of distinguishable colors
DETERMINISTIC_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Rose
    (128, 255, 0),    # Lime
    (0, 128, 255),    # Sky Blue
]


def get_object_color(object_id: int) -> Tuple[int, int, int]:
    """
    Get deterministic RGB color for an object ID.
    
    Uses a fixed palette indexed by object_id, no randomness.
    
    Args:
        object_id: Object identifier (1-indexed typically)
        
    Returns:
        RGB tuple (0-255)
    """
    # Use modulo to wrap around if more objects than colors
    idx = (object_id - 1) % len(DETERMINISTIC_COLORS)
    return DETERMINISTIC_COLORS[idx]


def blend_mask_on_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend a colored mask overlay onto a frame.
    
    Args:
        frame: RGB image (H, W, 3) uint8
        mask: Binary mask (H, W) bool or uint8
        color: RGB color tuple
        alpha: Blend opacity (0=invisible, 1=opaque)
        
    Returns:
        Blended RGB image (H, W, 3) uint8
    """
    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # Create a copy to avoid modifying original
    output = frame.copy()
    
    # Create colored overlay
    overlay = np.zeros_like(frame)
    overlay[mask] = color
    
    # Alpha blend only in masked region
    output[mask] = (
        (1 - alpha) * frame[mask] + alpha * overlay[mask]
    ).astype(np.uint8)
    
    return output


def render_overlay_frame(
    frame: Union[np.ndarray, Image.Image],
    masks: Dict[int, np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Render all mask overlays on a single frame.
    
    Args:
        frame: RGB frame (numpy or PIL)
        masks: Dict mapping object_id → mask
        alpha: Blend opacity
        
    Returns:
        RGB frame with overlays (H, W, 3) uint8
    """
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
    
    output = frame.copy()
    
    # Apply each mask with its deterministic color
    for object_id, mask in sorted(masks.items()):
        color = get_object_color(object_id)
        output = blend_mask_on_frame(output, mask, color, alpha)
    
    return output


def render_overlay_video(
    frames: List[Union[np.ndarray, Image.Image]],
    segmentation: SegmentationResult,
    alpha: float = 0.5,
) -> List[np.ndarray]:
    """
    Render mask overlays for all video frames.
    
    Args:
        frames: List of RGB frames
        segmentation: SegmentationResult with masks
        alpha: Blend opacity (0.4-0.6 recommended)
        
    Returns:
        List of overlay frames
    """
    overlay_frames = []
    
    for frame_idx, frame in enumerate(frames):
        if frame_idx in segmentation.masks_by_frame:
            masks = segmentation.masks_by_frame[frame_idx]
        else:
            masks = {}
        
        overlay_frame = render_overlay_frame(frame, masks, alpha)
        overlay_frames.append(overlay_frame)
    
    return overlay_frames


def save_overlay_video(
    frames: List[Union[np.ndarray, Image.Image]],
    segmentation: SegmentationResult,
    output_path: str,
    fps: float = 30.0,
    alpha: float = 0.5,
) -> str:
    """
    Render and save overlay video.
    
    Args:
        frames: Original video frames
        segmentation: SegmentationResult with masks
        output_path: Output video path
        fps: Frame rate
        alpha: Blend opacity
        
    Returns:
        Path to saved video
    """
    print(f"Rendering overlay video to {output_path}...")
    
    overlay_frames = render_overlay_video(frames, segmentation, alpha)
    
    encode_video(overlay_frames, output_path, fps)
    
    print(f"Overlay video saved: {output_path}")
    return output_path


def create_mask_comparison_grid(
    frame: Union[np.ndarray, Image.Image],
    mask: np.ndarray,
    edited_frame: Optional[np.ndarray] = None,
    overlay_alpha: float = 0.5,
) -> np.ndarray:
    """
    Create a 2x2 comparison grid showing:
    - Original frame
    - Mask overlay
    - Mask only
    - Edited frame (if provided)
    
    Args:
        frame: Original RGB frame
        mask: Binary mask
        edited_frame: Optional edited result
        overlay_alpha: Blend opacity for overlay
        
    Returns:
        Grid image (2H, 2W, 3)
    """
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
    
    H, W = frame.shape[:2]
    
    # Create overlay
    overlay = render_overlay_frame(
        frame, {1: mask}, alpha=overlay_alpha
    )
    
    # Create mask visualization (white on black)
    mask_vis = np.zeros_like(frame)
    mask_vis[mask] = (255, 255, 255)
    
    # Edited or placeholder
    if edited_frame is not None:
        bottom_right = edited_frame
    else:
        bottom_right = np.zeros_like(frame)
    
    # Assemble grid
    top = np.concatenate([frame, overlay], axis=1)
    bottom = np.concatenate([mask_vis, bottom_right], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    
    return grid


def add_text_annotation(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Add text annotation to a frame.
    
    Args:
        frame: RGB frame
        text: Text to add
        position: (x, y) position
        font_scale: Font size multiplier
        color: Text color (RGB)
        thickness: Line thickness
        
    Returns:
        Annotated frame
    """
    import cv2
    
    output = frame.copy()
    
    # Convert RGB to BGR for OpenCV
    color_bgr = (color[2], color[1], color[0])
    
    # Add black outline for visibility
    cv2.putText(
        output, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    
    # Add colored text
    cv2.putText(
        output, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        color_bgr, thickness, cv2.LINE_AA
    )
    
    return output
