"""
sam3_segmenter.py - SAM 3 Video Segmentation

Uses SAM 3 for text-prompted video segmentation:
- Takes video frames and text prompt
- Returns object IDs and per-frame masks
- Saves masks to disk for auditing
"""

# CRITICAL: Set this BEFORE any imports to prevent torchvision::nms registration errors on macOS.
# This must be at the very top of the file, before torch/torchvision are imported anywhere.
import os
os.environ["TORCHVISION_DISABLE_NMS_OP"] = "1"
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


@dataclass
class SegmentationResult:
    """Container for SAM 3 segmentation results."""
    object_ids: List[int]
    selected_object_id: int
    masks_by_frame: Dict[int, Dict[int, np.ndarray]]  # {frame_idx: {obj_id: mask}}
    num_frames: int
    height: int
    width: int
    text_prompt: str
    mean_mask_area_frac: float
    merge_info: Optional[Dict[str, Any]] = None
    
    def get_mask(self, frame_idx: int, object_id: Optional[int] = None) -> np.ndarray:
        """Get mask for a specific frame and object."""
        if object_id is None:
            object_id = self.selected_object_id
        return self.masks_by_frame[frame_idx][object_id]
    
    def get_all_masks_for_object(self, object_id: Optional[int] = None) -> List[np.ndarray]:
        """Get all masks for a specific object across frames."""
        if object_id is None:
            object_id = self.selected_object_id
        return [self.masks_by_frame[i][object_id] for i in range(self.num_frames)]

    def get_union_mask(self, frame_idx: int) -> np.ndarray:
        """Get union of all object masks for a frame."""
        frame_masks = self.masks_by_frame[frame_idx]
        combined = np.zeros((self.height, self.width), dtype=bool)
        for mask in frame_masks.values():
            combined |= mask.astype(bool)
        return combined


def _cluster_masks_by_proximity(
    masks: List[np.ndarray],
    proximity_px: int,
) -> List[List[int]]:
    """
    Group masks into spatially connected clusters.

    Dilates each mask by ``proximity_px`` pixels, then merges any pair whose
    dilated regions overlap (union-find). Returns a list of groups where each
    group is a list of original mask indices.
    """
    import cv2

    n = len(masks)
    if n <= 1:
        return [list(range(n))]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * proximity_px + 1, 2 * proximity_px + 1)
    )
    dilated = [
        cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool)
        for m in masks
    ]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.any(dilated[i] & dilated[j]):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    return list(groups.values())


def _merge_mask_group(masks: List[np.ndarray], indices: List[int]) -> np.ndarray:
    """Union all masks at *indices* into a single binary mask."""
    merged = np.zeros_like(masks[0], dtype=bool)
    for idx in indices:
        merged |= masks[idx].astype(bool)
    return merged


def _detect_gpus() -> List[str]:
    """Return list of available CUDA device strings, e.g. ['cuda:0', 'cuda:1']."""
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


class SAM3VideoSegmenter:
    """
    SAM 3 Video Segmenter using Hugging Face Transformers.
    
    Performs Promptable Concept Segmentation (PCS) on videos using text prompts.
    Multi-GPU aware: spreads SAM3 image model, tracker model, and DINOv3
    across available GPUs to avoid OOM and speed up processing.
    """
    
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize SAM 3 Video Segmenter.
        
        Args:
            model_id: Hugging Face model ID
            device: Device to run on (auto-detected if None)
            dtype: Model dtype (bfloat16 recommended for efficiency)
        """
        self.model_id = model_id
        self.dtype = dtype
        
        self._gpus = _detect_gpus()
        self.num_gpus = len(self._gpus)

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0" if self.num_gpus > 0 else "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                if dtype == torch.bfloat16:
                    self.dtype = torch.float32
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Assign models to different GPUs when available
        self.tracker_device = self._gpus[1] if self.num_gpus >= 2 else self.device
        self.dino_device = self._gpus[min(2, self.num_gpus - 1)] if self.num_gpus >= 2 else self.device

        if self.num_gpus > 1:
            print(f"[SAM3] Multi-GPU detected: {self.num_gpus} GPUs")
            print(f"  Image model:   {self.device}")
            print(f"  Tracker model: {self.tracker_device}")
            print(f"  DINOv3:        {self.dino_device}")
        
        self.model = None
        self.processor = None
        self._loaded = False
        self.batch_size = 4
    
    def load_model(self):
        """Load SAM 3 model and processor."""
        if self._loaded:
            return
        
        print(f"Loading SAM 3 model from {self.model_id}...")
        
        # Prefer image model/processor for frame-by-frame text prompts.
        try:
            # Try root namespace first
            from transformers import Sam3Model, Sam3Processor
        except Exception:
            # Try module paths (some versions don't export to root)
            try:
                from transformers.models.sam3.modeling_sam3 import Sam3Model
                from transformers.models.sam3.processing_sam3 import Sam3Processor
            except Exception as exc:
                raise ImportError(
                    "SAM 3 classes are missing from transformers. "
                    "Install a version that includes SAM3 (often the latest "
                    "main branch), e.g.: "
                    "`python -m pip install -U git+https://github.com/huggingface/transformers`"
                ) from exc
        
        self.model = Sam3Model.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        
        self.processor = Sam3Processor.from_pretrained(self.model_id)
        self._loaded = True
        self._fallback_mode = True

        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        print(f"SAM 3 (image mode) loaded on {self.device}")

    def offload_model_to_cpu(self) -> None:
        """Move SAM3 to CPU to free GPU memory (e.g. before loading ROSE on the same GPU)."""
        if self.model is None or not self._loaded:
            return
        try:
            self.model = self.model.to("cpu")
        except Exception:
            return
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            print("SAM 3 offloaded to CPU (GPU memory freed).")

    def reload_model_to_device(self) -> None:
        """Move SAM3 back to the configured device after :meth:`offload_model_to_cpu`."""
        if self.model is None or not self._loaded:
            return
        try:
            p0 = next(self.model.parameters())
            if p0.device.type == "cpu" and self.device != "cpu":
                self.model = self.model.to(self.device)
                print(f"SAM 3 moved back to {self.device}.")
        except StopIteration:
            pass
    
    def segment_video(
        self,
        frames: List[Image.Image],
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        select_largest: bool = True,
        multi_object: bool = False,
        merge_proximity_px: int = 50,
    ) -> SegmentationResult:
        """
        Segment objects in video using text prompt.

        When ``multi_object=True`` the segmenter keeps **all** detected masks
        on the first frame, clusters them by spatial proximity, and decides:

        * **Close masks** (within ``merge_proximity_px``) → union into one
          group.  Every subsequent frame also unions all detections into that
          single mask.  One DiffuEraser run is enough.
        * **Distant masks** → separate groups.  Each subsequent frame's
          detections are assigned to the nearest group, giving one mask
          sequence per group.  The pipeline runs DiffuEraser once per group.
        """
        self.load_model()

        num_frames = len(frames)
        height, width = np.array(frames[0]).shape[:2]

        print(f"Segmenting {num_frames} frames with prompt: '{text_prompt}'")

        masks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}

        first_frame_masks, first_frame_boxes, first_frame_scores = (
            self._segment_frame(frames[0], text_prompt, threshold, mask_threshold)
        )

        if len(first_frame_masks) == 0:
            raise ValueError(f"No objects found matching prompt: '{text_prompt}'")

        num_raw = len(first_frame_masks)

        # ==================================================================
        # Multi-object path
        # ==================================================================
        if multi_object:
            groups = _cluster_masks_by_proximity(first_frame_masks, merge_proximity_px)
            num_groups = len(groups)
            print(f"  Detected {num_raw} instance(s), "
                  f"clustered into {num_groups} group(s) "
                  f"(merge_proximity={merge_proximity_px}px)")

            group_ids: List[int] = []
            for g_idx, member_indices in enumerate(groups):
                gid = g_idx + 1
                group_ids.append(gid)
                merged = _merge_mask_group(first_frame_masks, member_indices)
                masks_by_frame.setdefault(0, {})[gid] = merged
                member_str = ", ".join(str(i + 1) for i in member_indices)
                print(f"    Group {gid}: masks [{member_str}]")

            object_ids = group_ids
            selected_object_id = group_ids[0]

            merge_info: Dict[str, Any] = {
                "multi_object": True,
                "num_raw_detections": num_raw,
                "num_groups": num_groups,
                "merge_proximity_px": merge_proximity_px,
                "groups": {
                    g_idx + 1: members
                    for g_idx, members in enumerate(groups)
                },
            }

            remaining = list(range(1, num_frames))
            pbar = tqdm(total=len(remaining), desc="Segmenting frames")
            for batch_start in range(0, len(remaining), self.batch_size):
                batch_indices = remaining[batch_start:batch_start + self.batch_size]
                batch_frames = [frames[i] for i in batch_indices]
                batch_results = self._segment_frames_batch(
                    batch_frames, text_prompt, threshold, mask_threshold
                )
                for frame_idx, (frame_masks, _, _) in zip(batch_indices, batch_results):
                    self._assign_multi_object_frame(
                        frame_idx, frame_masks, masks_by_frame,
                        group_ids, num_groups, height, width,
                    )
                pbar.update(len(batch_indices))
            pbar.close()

        # ==================================================================
        # Single-object path (original behaviour)
        # ==================================================================
        else:
            merge_info = None
            num_objects = num_raw
            object_ids = list(range(1, num_objects + 1))

            if select_largest and num_objects > 1:
                areas = [mask.sum() for mask in first_frame_masks]
                selected_idx = int(np.argmax(areas))
                selected_object_id = object_ids[selected_idx]
                print(f"Selected object {selected_object_id} "
                      f"(largest of {num_objects} detected)")
                object_ids = [selected_object_id]
                first_frame_masks = [first_frame_masks[selected_idx]]
            else:
                selected_object_id = object_ids[0]

            masks_by_frame[0] = {
                oid: mask for oid, mask in zip(object_ids, first_frame_masks)
            }

            remaining = list(range(1, num_frames))
            pbar = tqdm(total=len(remaining), desc="Segmenting frames")
            for batch_start in range(0, len(remaining), self.batch_size):
                batch_indices = remaining[batch_start:batch_start + self.batch_size]
                batch_frames = [frames[i] for i in batch_indices]
                batch_results = self._segment_frames_batch(
                    batch_frames, text_prompt, threshold, mask_threshold
                )
                for fi, (frame_masks, _, _) in zip(batch_indices, batch_results):
                    if len(frame_masks) > 0:
                        areas = [mask.sum() for mask in frame_masks]
                        best_idx = int(np.argmax(areas))
                        masks_by_frame[fi] = {selected_object_id: frame_masks[best_idx]}
                    else:
                        masks_by_frame[fi] = {
                            selected_object_id: np.zeros((height, width), dtype=bool)
                        }
                pbar.update(len(batch_indices))
            pbar.close()

        # Statistics
        total_area = 0
        total_pixels = height * width * num_frames
        for fm in masks_by_frame.values():
            for mask in fm.values():
                total_area += mask.sum()
        mean_mask_area_frac = total_area / total_pixels if total_pixels > 0 else 0.0

        return SegmentationResult(
            object_ids=object_ids,
            selected_object_id=selected_object_id,
            masks_by_frame=masks_by_frame,
            num_frames=num_frames,
            height=height,
            width=width,
            text_prompt=text_prompt,
            mean_mask_area_frac=float(mean_mask_area_frac),
            merge_info=merge_info,
        )
    
    @staticmethod
    def _assign_multi_object_frame(
        frame_idx: int,
        frame_masks: List[np.ndarray],
        masks_by_frame: Dict[int, Dict[int, np.ndarray]],
        group_ids: List[int],
        num_groups: int,
        height: int,
        width: int,
    ) -> None:
        """Assign detected masks to object groups for one frame."""
        if len(frame_masks) == 0:
            masks_by_frame[frame_idx] = {
                gid: np.zeros((height, width), dtype=bool)
                for gid in group_ids
            }
            return

        if num_groups == 1:
            union_mask = np.zeros((height, width), dtype=bool)
            for m in frame_masks:
                union_mask |= m.astype(bool)
            masks_by_frame[frame_idx] = {group_ids[0]: union_mask}
        else:
            prev = masks_by_frame[frame_idx - 1]
            assigned: Dict[int, np.ndarray] = {
                gid: np.zeros((height, width), dtype=bool)
                for gid in group_ids
            }
            for det in frame_masks:
                best_gid = group_ids[0]
                best_overlap = 0
                for gid in group_ids:
                    ov = int(np.sum(det.astype(bool) & prev[gid].astype(bool)))
                    if ov > best_overlap:
                        best_overlap = ov
                        best_gid = gid
                assigned[best_gid] |= det.astype(bool)
            masks_by_frame[frame_idx] = assigned

    def _segment_frame(
        self,
        frame: Image.Image,
        text_prompt: str,
        threshold: float,
        mask_threshold: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Segment a single frame using text prompt."""
        results = self._segment_frames_batch(
            [frame], text_prompt, threshold, mask_threshold
        )
        return results[0]

    def _segment_frame_with_box(
        self,
        frame: Image.Image,
        bbox: Tuple[int, int, int, int],
        mask_threshold: float = 0.5,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Segment a frame using a bounding box prompt (no text).

        Args:
            frame:  PIL image.
            bbox:   (x_min, y_min, x_max, y_max) in pixel coordinates.
            mask_threshold: Threshold for converting logits to binary mask.

        Returns:
            (masks, scores) — lists of binary masks and confidence scores.
        """
        self.load_model()

        x_min, y_min, x_max, y_max = bbox
        input_boxes = [[[float(x_min), float(y_min), float(x_max), float(y_max)]]]

        inputs = self.processor(
            images=[frame],
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        use_amp = self.device.startswith("cuda") and self.dtype != torch.float32
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            outputs = self.model(**inputs)

        original_sizes = inputs.get(
            "original_sizes",
            [[frame.height, frame.width]],
        )
        if hasattr(original_sizes, "tolist"):
            original_sizes = original_sizes.tolist()

        all_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            mask_threshold=mask_threshold,
            target_sizes=original_sizes,
        )

        masks: List[np.ndarray] = []
        scores: List[float] = []

        if all_results and "masks" in all_results[0] and len(all_results[0]["masks"]) > 0:
            result = all_results[0]
            for i in range(len(result["masks"])):
                masks.append(result["masks"][i].cpu().numpy().astype(bool))
                if "scores" in result:
                    scores.append(float(result["scores"][i]))

        return masks, scores

    def _segment_frames_batch(
        self,
        frames: List[Image.Image],
        text_prompt: str,
        threshold: float,
        mask_threshold: float,
    ) -> List[Tuple[List[np.ndarray], List[np.ndarray], List[float]]]:
        """
        Segment a batch of frames in one forward pass.

        Returns one (masks, boxes, scores) tuple per frame.
        """
        inputs = self.processor(
            images=frames,
            text=[text_prompt] * len(frames),
            return_tensors="pt",
        ).to(self.device)

        use_amp = self.device.startswith("cuda") and self.dtype != torch.float32
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            outputs = self.model(**inputs)

        original_sizes = inputs.get(
            "original_sizes",
            [[f.height, f.width] for f in frames],
        )
        if hasattr(original_sizes, "tolist"):
            original_sizes = original_sizes.tolist()

        all_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=original_sizes,
        )

        batch_out: List[Tuple[List[np.ndarray], List[np.ndarray], List[float]]] = []
        for result in all_results:
            masks: List[np.ndarray] = []
            boxes: List[np.ndarray] = []
            scores: List[float] = []

            if "masks" in result and len(result["masks"]) > 0:
                for i in range(len(result["masks"])):
                    masks.append(result["masks"][i].cpu().numpy().astype(bool))
                    if "boxes" in result:
                        boxes.append(result["boxes"][i].float().cpu().numpy())
                    if "scores" in result:
                        scores.append(float(result["scores"][i]))

            batch_out.append((masks, boxes, scores))

        return batch_out


    def segment_video_multi_prompt(
        self,
        frames: List[Image.Image],
        prompts: List[str],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        merge_proximity_px: int = 50,
    ) -> SegmentationResult:
        """
        Segment objects using multiple prompts, one per concept.

        Each prompt is segmented independently (largest detection per frame).
        First-frame masks are clustered by spatial proximity:
        - Physically connected prompts → single union mask (one inpainting run).
        - Distant prompts → separate groups (sequential inpainting runs).
        """
        self.load_model()

        num_frames = len(frames)
        height, width = np.array(frames[0]).shape[:2]

        print(f"Segmenting {num_frames} frames with {len(prompts)} prompts: {prompts}")

        per_prompt_masks: List[List[np.ndarray]] = []

        for p_idx, prompt in enumerate(prompts):
            print(f"  Prompt {p_idx + 1}/{len(prompts)}: '{prompt}'")
            prompt_frame_masks: List[np.ndarray] = []
            pbar = tqdm(total=num_frames, desc=f"  '{prompt}'")
            for batch_start in range(0, num_frames, self.batch_size):
                batch_frames = frames[batch_start:batch_start + self.batch_size]
                batch_results = self._segment_frames_batch(
                    batch_frames, prompt, threshold, mask_threshold
                )
                for frame_masks, _, _ in batch_results:
                    if len(frame_masks) > 0:
                        areas = [m.sum() for m in frame_masks]
                        best_idx = int(np.argmax(areas))
                        prompt_frame_masks.append(frame_masks[best_idx])
                    else:
                        prompt_frame_masks.append(
                            np.zeros((height, width), dtype=bool)
                        )
                pbar.update(len(batch_frames))
            pbar.close()
            per_prompt_masks.append(prompt_frame_masks)

        first_frame_masks = [pm[0] for pm in per_prompt_masks]
        groups = _cluster_masks_by_proximity(first_frame_masks, merge_proximity_px)
        num_groups = len(groups)

        print(f"  {len(prompts)} prompt masks clustered into {num_groups} group(s) "
              f"(proximity={merge_proximity_px}px)")
        for g_idx, members in enumerate(groups):
            member_prompts = [prompts[i] for i in members]
            print(f"    Group {g_idx + 1}: {member_prompts}")

        group_ids = list(range(1, num_groups + 1))
        masks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}

        for frame_idx in range(num_frames):
            masks_by_frame[frame_idx] = {}
            for g_idx, member_indices in enumerate(groups):
                gid = g_idx + 1
                union_mask = np.zeros((height, width), dtype=bool)
                for p_idx in member_indices:
                    union_mask |= per_prompt_masks[p_idx][frame_idx].astype(bool)
                masks_by_frame[frame_idx][gid] = union_mask

        total_area = 0
        total_pixels = height * width * num_frames
        for fm in masks_by_frame.values():
            for mask in fm.values():
                total_area += mask.sum()
        mean_mask_area_frac = total_area / total_pixels if total_pixels > 0 else 0.0

        merge_info: Dict[str, Any] = {
            "multi_prompt": True,
            "prompts": prompts,
            "num_groups": num_groups,
            "merge_proximity_px": merge_proximity_px,
            "groups": {
                g_idx + 1: [prompts[i] for i in members]
                for g_idx, members in enumerate(groups)
            },
        }

        return SegmentationResult(
            object_ids=group_ids,
            selected_object_id=group_ids[0],
            masks_by_frame=masks_by_frame,
            num_frames=num_frames,
            height=height,
            width=width,
            text_prompt=", ".join(prompts),
            mean_mask_area_frac=float(mean_mask_area_frac),
            merge_info=merge_info,
        )


    def segment_video_dino_tracked(
        self,
        frames: List[Image.Image],
        text_prompt: str,
        dino_tracker: "DINOv3Tracker" = None,
        anchor_stride: int = 5,
        anchor_confidence: float = 0.90,
        anchor_min_confidence: float = 0.30,
        similarity_threshold: float = 0.35,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_area_ratio: float = 3.0,
    ) -> SegmentationResult:
        """Segment a video using SAM3 Tracker Video propagation with
        optional DINOv3 verification.

        Architecture:
          Phase 1 — Anchor hunting (SAM3 text-search with stride)
          Phase 2 — SAM3 Tracker Video propagation (locks onto the object
                    from the anchor mask and tracks through all frames
                    using memory attention at pixel level)
          Phase 3 — DINOv3 verification (optional quality gate: flag frames
                    where the tracked mask doesn't overlap with the DINOv3
                    target signature; skipped when dino_tracker is None)

        Args:
            frames:  List of PIL images.
            text_prompt: Text prompt for SAM3 anchor search.
            dino_tracker: Pre-loaded DINOv3Tracker instance, or None to
                skip DINOv3 verification (Phase 3).
            anchor_stride: Frame stride for anchor hunting.
            anchor_confidence: Early-exit confidence threshold.
            anchor_min_confidence: Abort if no frame exceeds this.
            similarity_threshold: DINOv3 cosine similarity floor for verification.
            threshold: SAM3 detection threshold.
            mask_threshold: SAM3 mask binarization threshold.

        Returns:
            SegmentationResult compatible with the rest of the pipeline.
        """
        self.load_model()

        num_frames = len(frames)
        height, width = np.array(frames[0]).shape[:2]
        masks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        object_id = 1

        print(f"[SAM3 video tracker] {num_frames} frames, prompt='{text_prompt}'")

        # ====== Phase 1: Anchor Hunting ======
        print("[Phase 1] Hunting for anchor frame ...")
        best_anchor_idx = 0
        best_anchor_score = 0.0
        best_anchor_mask = None

        search_indices = list(range(0, num_frames, anchor_stride))
        if search_indices[-1] != num_frames - 1:
            search_indices.append(num_frames - 1)

        for idx in search_indices:
            frame_masks, _, frame_scores = self._segment_frame(
                frames[idx], text_prompt, threshold, mask_threshold,
            )
            if len(frame_masks) == 0 or len(frame_scores) == 0:
                continue

            top_score = max(frame_scores)
            if top_score > best_anchor_score:
                best_anchor_score = top_score
                best_anchor_idx = idx
                best_mask_i = int(np.argmax([s for s in frame_scores]))
                best_anchor_mask = frame_masks[best_mask_i]

            if top_score >= anchor_confidence:
                print(f"  Early exit: frame {idx} score={top_score:.3f}")
                break
        else:
            print(f"  Best anchor: frame {best_anchor_idx} "
                  f"score={best_anchor_score:.3f}")

        if best_anchor_mask is None or best_anchor_score < anchor_min_confidence:
            raise ValueError(
                f"No anchor found for prompt '{text_prompt}': "
                f"best score={best_anchor_score:.3f} < min={anchor_min_confidence}"
            )

        anchor_idx = best_anchor_idx
        anchor_mask = best_anchor_mask
        anchor_area = float(anchor_mask.sum())
        print(f"  Anchor: frame {anchor_idx} (score={best_anchor_score:.3f}, "
              f"area={anchor_area:.0f}px / "
              f"{anchor_area / (height * width) * 100:.1f}%)")

        # ====== Phase 2: SAM3 Tracker Video Propagation ======
        print("[Phase 2] SAM3 Tracker Video — propagating mask ...")

        import torch
        from transformers import (
            Sam3TrackerVideoModel,
            Sam3TrackerVideoProcessor,
        )

        # When tracker runs on a different GPU, keep the image model loaded.
        # When same GPU, free image model first to make room for tracker + video tensor.
        tracker_dev = self.tracker_device
        same_gpu = (tracker_dev == self.device) or (
            tracker_dev.startswith("cuda") and self.device.startswith("cuda")
            and tracker_dev.split(":")[-1] == self.device.split(":")[-1]
        )
        if same_gpu and self.model is not None:
            del self.model
            self.model = None
            self._loaded = False
            torch.cuda.empty_cache()

        # Subsample long videos to fit in GPU memory.
        # SAM3 tracker loads all frames as a GPU tensor; ~12 bytes/pixel/frame.
        # With multi-GPU, tracker gets a dedicated GPU so we can afford more frames.
        MAX_TRACKER_FRAMES = 800 if not same_gpu else 500

        # Use the same dtype as the image model — Turing GPUs (SM 7.5) lack
        # native bf16 SDPA support, so self.dtype may be float16.
        tracker_dtype = self.dtype

        def _run_tracker_pass(pass_frames, pass_anchor_idx, pass_label):
            """Run SAM3 tracker on a sequence of frames, return {frame_idx: bool_mask}."""
            n_pass = len(pass_frames)
            if n_pass > MAX_TRACKER_FRAMES:
                s_idx = np.linspace(0, n_pass - 1, MAX_TRACKER_FRAMES, dtype=np.int64)
                s_idx = np.unique(s_idx)
                if pass_anchor_idx not in s_idx:
                    s_idx = np.sort(np.append(s_idx, pass_anchor_idx))
                s_frames = [pass_frames[i] for i in s_idx]
                s_anchor = int(np.searchsorted(s_idx, pass_anchor_idx))
                print(f"  [{pass_label}] Subsampling {n_pass} -> {len(s_idx)} frames")
            else:
                s_idx = np.arange(n_pass)
                s_frames = pass_frames
                s_anchor = pass_anchor_idx

            print(f"  [{pass_label}] Loading tracker on {tracker_dev} (dtype={tracker_dtype})")
            t_model = Sam3TrackerVideoModel.from_pretrained(
                "facebook/sam3", torch_dtype=tracker_dtype,
            ).to(tracker_dev).eval()
            t_proc = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

            sess = t_proc.init_video_session(
                video=s_frames, inference_device=tracker_dev, dtype=tracker_dtype,
            )
            mt = torch.from_numpy(anchor_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            t_proc.add_inputs_to_inference_session(
                inference_session=sess, frame_idx=s_anchor,
                obj_ids=object_id, input_masks=mt,
            )
            _ = t_model(inference_session=sess, frame_idx=s_anchor)

            n_sub = len(s_idx)
            print(f"  [{pass_label}] Propagating through {n_sub} frames ...")
            result_sub = {}
            for output in tqdm(
                t_model.propagate_in_video_iterator(sess),
                total=n_sub, desc=f"  {pass_label}",
            ):
                fi = output.frame_idx
                res_masks = t_proc.post_process_masks(
                    [output.pred_masks], original_sizes=[[height, width]], binarize=True,
                )[0]
                result_sub[fi] = res_masks[0, 0].cpu().numpy().astype(bool)

            del t_model, t_proc, sess
            torch.cuda.empty_cache()

            # Map subsampled indices back
            result = {}
            for sub_fi, orig_fi in enumerate(s_idx):
                if sub_fi in result_sub:
                    result[int(orig_fi)] = result_sub[sub_fi]

            if n_pass > MAX_TRACKER_FRAMES:
                sorted_tracked = sorted(result.keys())
                for fi in range(n_pass):
                    if fi in result:
                        continue
                    pos = np.searchsorted(sorted_tracked, fi)
                    if pos == 0:
                        nearest = sorted_tracked[0]
                    elif pos >= len(sorted_tracked):
                        nearest = sorted_tracked[-1]
                    else:
                        left, right = sorted_tracked[pos - 1], sorted_tracked[pos]
                        nearest = left if (fi - left) <= (right - fi) else right
                    result[fi] = result[nearest]

            return result

        # Bidirectional propagation: forward pass (anchor → end) always runs.
        # Backward pass (anchor → start) runs only when anchor is not at frame 0,
        # by reversing the pre-anchor frames so the tracker propagates "forward"
        # through the reversed sequence.
        need_backward = anchor_idx > 0
        propagated = {}

        if need_backward:
            # Forward pass: anchor frame to end
            fwd_frames = frames[anchor_idx:]
            fwd_result = _run_tracker_pass(fwd_frames, 0, "Forward")
            for fi, mask in fwd_result.items():
                propagated[fi + anchor_idx] = mask

            # Backward pass: reverse frames [0..anchor] so tracker goes "forward"
            bwd_frames = frames[:anchor_idx + 1][::-1]  # reversed
            bwd_anchor = 0  # anchor is now at position 0 in reversed sequence
            bwd_result = _run_tracker_pass(bwd_frames, bwd_anchor, "Backward")
            # Map reversed indices back to original
            for rev_fi, mask in bwd_result.items():
                orig_fi = anchor_idx - rev_fi
                # Only use backward result for frames that are empty in forward pass
                if orig_fi not in propagated or not propagated[orig_fi].any():
                    propagated[orig_fi] = mask

            print(f"  Bidirectional propagation: forward {len(fwd_result)} + backward {len(bwd_result)} frames")
        else:
            # Anchor at frame 0 — single forward pass is sufficient
            propagated = _run_tracker_pass(frames, anchor_idx, "Forward")

        tracked_count = 0
        for fi in range(num_frames):
            if fi in propagated:
                masks_by_frame[fi] = {object_id: propagated[fi]}
                if propagated[fi].any():
                    tracked_count += 1
            else:
                masks_by_frame[fi] = {
                    object_id: np.zeros((height, width), dtype=bool)
                }

        print(f"  SAM3 Tracker: {tracked_count}/{num_frames} frames have masks")

        # ====== Phase 3: DINOv3 Verification (optional) ======
        verified = 0
        flagged = 0
        if dino_tracker is not None:
            print("[Phase 3] DINOv3 verification ...")
            signature = dino_tracker.extract_signature(
                frames[anchor_idx], anchor_mask,
            )
            area_ceiling = anchor_area * max_area_ratio

            for fi in tqdm(range(num_frames), desc="  Verifying"):
                mask = masks_by_frame[fi][object_id]
                if not mask.any():
                    continue

                mask_area = float(mask.sum())
                if mask_area > area_ceiling:
                    masks_by_frame[fi][object_id] = np.zeros(
                        (height, width), dtype=bool
                    )
                    flagged += 1
                    continue

                heatmap = dino_tracker.compute_heatmap(frames[fi], signature)
                spatial = dino_tracker.heatmap_to_spatial_mask(
                    heatmap, frame_h=height, frame_w=width,
                    threshold=similarity_threshold,
                )
                overlap = float((mask & spatial).sum()) / max(mask_area, 1.0)
                if overlap < 0.10:
                    masks_by_frame[fi][object_id] = np.zeros(
                        (height, width), dtype=bool
                    )
                    flagged += 1
                else:
                    verified += 1

            print(f"  Verified: {verified}, flagged/zeroed: {flagged}")
        else:
            print("[Phase 3] DINOv3 verification skipped (no tracker provided)")

        # Statistics
        total_area = 0
        total_pixels = height * width * num_frames
        for fm in masks_by_frame.values():
            for mask in fm.values():
                total_area += mask.sum()
        mean_mask_area_frac = total_area / total_pixels if total_pixels > 0 else 0.0

        non_zero = sum(
            1 for fi in range(num_frames)
            if masks_by_frame.get(fi, {}).get(object_id) is not None
            and masks_by_frame[fi][object_id].any()
        )
        print(f"[DINO-tracked] Done: {non_zero}/{num_frames} frames have masks, "
              f"mean area={mean_mask_area_frac:.4f}")

        return SegmentationResult(
            object_ids=[object_id],
            selected_object_id=object_id,
            masks_by_frame=masks_by_frame,
            num_frames=num_frames,
            height=height,
            width=width,
            text_prompt=text_prompt,
            mean_mask_area_frac=float(mean_mask_area_frac),
            merge_info={
                "tracker": "dinov3",
                "anchor_frame": anchor_idx,
                "anchor_score": float(best_anchor_score),
                "similarity_threshold": similarity_threshold,
                "sam3_tracker_propagated": tracked_count,
                "dino_verified": verified,
                "dino_flagged": flagged,
            },
        )


    def track_with_anchor_mask(
        self,
        frames: List,
        anchor_mask: np.ndarray,
        anchor_idx: int = 0,
    ) -> Dict[int, np.ndarray]:
        """
        Run SAM3 video tracker with a pre-computed anchor mask (bidirectional).
        Returns {frame_idx: bool_mask} for all frames.
        Useful for tracking a known object without text-prompted anchor hunting.
        """
        import torch
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        num_frames = len(frames)
        height, width = np.array(frames[0]).shape[:2]
        object_id = 1

        tracker_dev = self.tracker_device
        same_gpu = (tracker_dev == self.device) or (
            tracker_dev.startswith("cuda") and self.device.startswith("cuda")
            and tracker_dev.split(":")[-1] == self.device.split(":")[-1]
        )
        MAX_TRACKER_FRAMES = 800 if not same_gpu else 500
        tracker_dtype = self.dtype

        def _run_pass(pass_frames, pass_anchor_idx, label):
            n = len(pass_frames)
            if n > MAX_TRACKER_FRAMES:
                s_idx = np.unique(np.linspace(0, n - 1, MAX_TRACKER_FRAMES, dtype=np.int64))
                if pass_anchor_idx not in s_idx:
                    s_idx = np.sort(np.append(s_idx, pass_anchor_idx))
                s_frames = [pass_frames[i] for i in s_idx]
                s_anchor = int(np.searchsorted(s_idx, pass_anchor_idx))
            else:
                s_idx = np.arange(n)
                s_frames = pass_frames
                s_anchor = pass_anchor_idx

            t_model = Sam3TrackerVideoModel.from_pretrained(
                "facebook/sam3", torch_dtype=tracker_dtype,
            ).to(tracker_dev).eval()
            t_proc = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
            sess = t_proc.init_video_session(
                video=s_frames, inference_device=tracker_dev, dtype=tracker_dtype,
            )
            mt = torch.from_numpy(anchor_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            t_proc.add_inputs_to_inference_session(
                inference_session=sess, frame_idx=s_anchor,
                obj_ids=object_id, input_masks=mt,
            )
            _ = t_model(inference_session=sess, frame_idx=s_anchor)

            result_sub = {}
            for output in t_model.propagate_in_video_iterator(sess):
                fi = output.frame_idx
                res = t_proc.post_process_masks(
                    [output.pred_masks], original_sizes=[[height, width]], binarize=True,
                )[0]
                result_sub[fi] = res[0, 0].cpu().numpy().astype(bool)

            del t_model, t_proc, sess
            torch.cuda.empty_cache()

            result = {}
            for sub_fi, orig_fi in enumerate(s_idx):
                if sub_fi in result_sub:
                    result[int(orig_fi)] = result_sub[sub_fi]
            return result

        propagated = {}
        need_backward = anchor_idx > 0

        if need_backward:
            fwd = _run_pass(frames[anchor_idx:], 0, "Fwd")
            for fi, m in fwd.items():
                propagated[fi + anchor_idx] = m
            bwd = _run_pass(frames[:anchor_idx + 1][::-1], 0, "Bwd")
            for rev_fi, m in bwd.items():
                orig_fi = anchor_idx - rev_fi
                if orig_fi not in propagated or not propagated[orig_fi].any():
                    propagated[orig_fi] = m
        else:
            propagated = _run_pass(frames, anchor_idx, "Fwd")

        # Fill missing frames with empty masks
        for fi in range(num_frames):
            if fi not in propagated:
                propagated[fi] = np.zeros((height, width), dtype=bool)

        return propagated


def save_masks(
    result: SegmentationResult,
    output_dir: str,
    save_png: bool = True,
    save_npz: bool = True,
) -> Dict[str, str]:
    """
    Save segmentation masks to disk.
    
    Args:
        result: SegmentationResult from segmentation
        output_dir: Directory to save masks
        save_png: Save individual PNG files
        save_npz: Save compressed numpy archive
        
    Returns:
        Dictionary of saved file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    
    if save_png:
        png_dir = os.path.join(output_dir, "png")
        Path(png_dir).mkdir(parents=True, exist_ok=True)
        
        for frame_idx, frame_masks in result.masks_by_frame.items():
            for obj_id, mask in frame_masks.items():
                # Convert bool mask to uint8 (0 or 255)
                mask_img = (mask.astype(np.uint8) * 255)
                img = Image.fromarray(mask_img, mode="L")
                
                filename = f"mask_f{frame_idx:06d}_o{obj_id}.png"
                filepath = os.path.join(png_dir, filename)
                img.save(filepath)
        
        saved_paths["png_dir"] = png_dir
    
    if save_npz:
        npz_path = os.path.join(output_dir, "masks.npz")
        is_multi = len(result.object_ids) > 1

        if is_multi:
            npz_data: Dict[str, Any] = {
                "text_prompt": np.array(result.text_prompt),
                "multi_object": np.array(True),
            }
            for obj_id in result.object_ids:
                masks_array = np.stack([
                    result.masks_by_frame[i][obj_id]
                    for i in range(result.num_frames)
                ])
                npz_data[f"masks_obj{obj_id}"] = masks_array
            np.savez_compressed(npz_path, **npz_data)
        else:
            obj_id = result.selected_object_id
            masks_array = np.stack([
                result.masks_by_frame[i][obj_id]
                for i in range(result.num_frames)
            ])
            np.savez_compressed(
                npz_path,
                masks=masks_array,
                object_id=obj_id,
                text_prompt=result.text_prompt,
            )
        saved_paths["npz"] = npz_path
    
    # Save metadata
    metadata: Dict[str, Any] = {
        "object_ids": result.object_ids,
        "selected_object_id": result.selected_object_id,
        "num_frames": result.num_frames,
        "height": result.height,
        "width": result.width,
        "text_prompt": result.text_prompt,
        "mean_mask_area_frac": result.mean_mask_area_frac,
    }
    if result.merge_info is not None:
        metadata["merge_info"] = result.merge_info
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    saved_paths["metadata"] = metadata_path
    
    return saved_paths


def load_masks(masks_dir: str) -> SegmentationResult:
    """
    Load previously saved masks from disk.
    
    Args:
        masks_dir: Directory containing saved masks
        
    Returns:
        SegmentationResult reconstructed from saved data
    """
    metadata_path = os.path.join(masks_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    npz_path = os.path.join(masks_dir, "masks.npz")
    data = np.load(npz_path, allow_pickle=False)

    is_multi = "multi_object" in data.files and bool(data["multi_object"])
    obj_ids = metadata["object_ids"]

    if is_multi:
        masks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        for i in range(metadata["num_frames"]):
            masks_by_frame[i] = {}
            for obj_id in obj_ids:
                masks_by_frame[i][obj_id] = data[f"masks_obj{obj_id}"][i]
    else:
        obj_id = metadata["selected_object_id"]
        masks_array = data["masks"]
        masks_by_frame = {
            i: {obj_id: masks_array[i]}
            for i in range(metadata["num_frames"])
        }
    
    return SegmentationResult(
        object_ids=obj_ids,
        selected_object_id=metadata["selected_object_id"],
        masks_by_frame=masks_by_frame,
        num_frames=metadata["num_frames"],
        height=metadata["height"],
        width=metadata["width"],
        text_prompt=metadata["text_prompt"],
        mean_mask_area_frac=metadata["mean_mask_area_frac"],
        merge_info=metadata.get("merge_info"),
    )
