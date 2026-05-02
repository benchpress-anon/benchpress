"""
BenchPress — SAM3 + DiffuEraser Video Object Removal Pipeline

End-to-end pipeline for video object removal using:
- SAM 3 for segmentation/tracking
- DiffuEraser for temporal-consistent inpainting
"""

from .video_io import (
    decode_video,
    decode_video_to_pil,
    encode_video,
    save_frames,
    get_video_info,
)

from .sam3_segmenter import (
    SAM3VideoSegmenter,
    SegmentationResult,
    save_masks,
    load_masks,
)

from .overlay_render import (
    get_object_color,
    render_overlay_frame,
    render_overlay_video,
    save_overlay_video,
)

from .diffueraser_backend import (
    DiffuEraserBackend,
    DiffuEraserConfig,
    run_telea_fallback,
    run_swap_stub,
    check_diffueraser_available,
)

from .pipeline import run_pipeline

__version__ = "1.0.0"
