"""
prompts.py - Reusable prompt templates for video editing backends.

Usage:
    from prompts import VACE_REMOVE, NEGATIVE_PROMPT_ZH
    prompt = VACE_REMOVE
"""

VACE_REMOVE = (
    "A clean background scene. Smooth natural surfaces, consistent lighting."
)

VACE_REMOVE_DETAILED = (
    "A steady video of the background. Natural floor, walls, and furniture. "
    "Consistent lighting and color. High quality, sharp details."
)

# Template for object-specific removal. Format with the object name.
VACE_REMOVE_OBJECT = (
    "The scene with no {object}. Clean background, natural surfaces, "
    "consistent lighting. High quality."
)

ROSE_REMOVE = (
    "Remove the specified object and all related effects, "
    "then restore a clean background."
)

EFFECTERASE_REMOVE = (
    "Remove the specified object and all related effects, "
    "then restore a clean background."
)

NEGATIVE_PROMPT_ZH = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
