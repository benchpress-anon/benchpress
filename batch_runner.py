"""
batch_runner.py - Multi-GPU batch runner for SAM3 + DiffuEraser video object removal.

Reads per-video prompts from data.json and processes videos in parallel across
available GPUs. Pre-filters to only unprocessed videos with valid prompts, then
splits work evenly across GPUs. Failures are recorded but never crash the batch.
"""

import argparse
import atexit
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from multiprocessing import Event, Process, Queue
from pathlib import Path
from typing import Dict, List, Optional


def build_video_index(dataset_root: str) -> Dict[str, str]:
    """Walk dataset_root and map video_id (stem) -> absolute path."""
    index: Dict[str, str] = {}
    for dirpath, _dirs, filenames in os.walk(dataset_root):
        for fname in filenames:
            if fname.endswith(".mp4"):
                vid_id = Path(fname).stem
                index[vid_id] = os.path.join(dirpath, fname)
    return index


def load_data_json(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def write_status(fh, record: dict):
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def filter_jobs(
    data: List[dict],
    video_index: Dict[str, str],
    output_root: str,
    probe_frame_count: bool = False,
) -> tuple:
    """Pre-filter data.json: return (actionable_jobs, skipped_records).

    Actionable jobs have a valid prompt, a resolvable video path, and no
    existing output. Everything else is recorded as skipped/failed upfront
    so workers only receive real work.
    """
    actionable = []
    skipped = []

    for entry in data:
        video_id = str(entry["id"])
        prompt = entry.get("prompt_word")

        if not prompt or (isinstance(prompt, str) and not prompt.strip()):
            skipped.append({
                "video_id": video_id,
                "prompt_word": prompt,
                "status": "skipped",
                "error_message": "null or empty prompt",
            })
            continue

        video_path = video_index.get(video_id)
        if video_path is None:
            skipped.append({
                "video_id": video_id,
                "prompt_word": prompt,
                "status": "failed",
                "error_message": "video file not found",
            })
            continue

        final_output = os.path.join(output_root, f"{video_id}.mp4")
        if os.path.exists(final_output):
            skipped.append({
                "video_id": video_id,
                "prompt_word": prompt,
                "input_video_path": video_path,
                "output_video_path": final_output,
                "status": "skipped",
                "error_message": "output already exists",
            })
            continue

        frame_count = 0
        if probe_frame_count:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception:
                pass
        actionable.append({
            "id": video_id,
            "prompt_word": prompt,
            "video_path": video_path,
            "frame_count": frame_count,
        })

    return actionable, skipped


def _match_resolution(src: str, dst: str, width: int, height: int):
    """Scale + crop video to exactly (width x height) via ffmpeg.

    DiffuEraser may output at a different resolution than the source (it
    downscales to max 512px internally).  This rescales to cover the target
    dimensions then center-crops to the exact size, preventing systematic
    resolution mismatches that VLMs could learn as an "edited video" signal.
    """
    # scale2ref would be ideal, but a simple scale+crop chain works:
    # 1. Scale so the smaller dimension matches the target (may overshoot the other)
    # 2. Center-crop to exact target size
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # ffmpeg failed — fall back to plain copy rather than crashing
        shutil.copy2(src, dst)


def _cuda_cleanup(gpu_id: int):
    """Release all CUDA resources so the NVIDIA driver stays clean on exit."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset forces the driver to fully release this device context
            torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()
        print(f"[GPU {gpu_id}] CUDA resources released cleanly.")
    except Exception as exc:
        print(f"[GPU {gpu_id}] CUDA cleanup warning: {exc}")


def worker_fn(
    gpu_id: int,
    job_queue: Queue,
    total_jobs: int,
    output_root: str,
    max_duration: float,
    result_file: str,
    dry_run: bool,
    diffueraser_path: str,
    threads_per_worker: int = 8,
    stop_event: Optional[Event] = None,
    use_dino_tracker: bool = False,
    dino_verify: bool = False,
    edit_backend: str = "diffueraser",
):
    """Pull videos from a shared queue and process on a single GPU.

    Dynamic scheduling: each worker grabs the next available video from the
    queue when it finishes the current one, ensuring no GPU sits idle while
    others still have work.
    """

    # CUDA_VISIBLE_DEVICES must be set BEFORE any torch import so the CUDA
    # runtime only sees one physical GPU (mapped to logical cuda:0).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    try:
        _worker_fn_inner(
            gpu_id, job_queue, total_jobs, output_root, max_duration,
            result_file, dry_run, diffueraser_path, threads_per_worker,
            stop_event, use_dino_tracker, dino_verify, edit_backend,
        )
    except Exception as exc:
        print(f"[GPU {gpu_id}] FATAL: worker crashed: {exc}")
        import traceback as _tb
        _tb.print_exc()


def _worker_fn_inner(
    gpu_id, job_queue, total_jobs, output_root, max_duration,
    result_file, dry_run, diffueraser_path, threads_per_worker,
    stop_event, use_dino_tracker, dino_verify, edit_backend,
):
    """Inner worker function — exceptions propagate to worker_fn for logging."""

    # Ensure CUDA cleanup runs no matter how this process exits.
    if not dry_run:
        atexit.register(_cuda_cleanup, gpu_id)

    def _handle_signal(signum, frame):
        name = signal.Signals(signum).name
        print(f"[GPU {gpu_id}] Received {name} — finishing current video then exiting.")
        if stop_event is not None:
            stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    sys.path.insert(0, os.path.dirname(__file__))
    from video_io import get_video_info

    segmenter = None
    diffueraser_runner = None
    dino_tracker = None

    if not dry_run:
        from pipeline import run_pipeline, BACKEND_DIFFUERASER
        import torch
        torch.set_num_threads(threads_per_worker)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        from sam3_segmenter import SAM3VideoSegmenter

        # CUDA_VISIBLE_DEVICES remaps physical GPUs so only device 0 is visible
        # to this worker.  Force SAM3 onto cuda:0 with fp16 (Turing lacks bf16
        # SDPA) and disable its multi-GPU spreading which would reference
        # non-existent cuda:1/cuda:2.
        print(f"[GPU {gpu_id}] Loading SAM3 model ...")
        segmenter = SAM3VideoSegmenter(device="cuda:0", dtype=torch.float16)
        segmenter.load_model()

        if edit_backend == BACKEND_DIFFUERASER:
            from diffueraser_backend import DiffuEraserConfig, InProcessDiffuEraserRunner

            weights_path = os.path.join(diffueraser_path, "weights")
            cfg = DiffuEraserConfig(
                diffueraser_path=diffueraser_path,
                weights_path=weights_path,
                device="auto",
            )
            print(f"[GPU {gpu_id}] Loading DiffuEraser ...")
            diffueraser_runner = InProcessDiffuEraserRunner(cfg)
            try:
                diffueraser_runner.load()
            except Exception as exc:
                print(f"[GPU {gpu_id}] WARNING: DiffuEraser load failed ({exc}); "
                      "falling back to subprocess per video.")
                diffueraser_runner = None
        else:
            print(f"[GPU {gpu_id}] Skipping DiffuEraser preload (backend={edit_backend}).")

        dino_tracker = None
        if dino_verify:
            from dino_tracker import DINOv3Tracker
            print(f"[GPU {gpu_id}] Loading DINOv3 tracker for verification ...")
            dino_tracker = DINOv3Tracker()
            dino_tracker.load_model()

        print(f"[GPU {gpu_id}] Ready. ~{total_jobs} videos in shared queue.")

    os.makedirs(output_root, exist_ok=True)

    with open(result_file, "w") as fh:
        idx = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                print(f"[GPU {gpu_id}] Graceful stop after {idx} videos.")
                break

            # Pull next job from shared queue
            try:
                job = job_queue.get(timeout=5)
            except Exception:
                # Queue empty — all work consumed
                break

            idx += 1

            video_id = job["id"]
            prompt = job["prompt_word"]
            video_path = job["video_path"]
            final_output = os.path.join(output_root, f"{video_id}.mp4")
            tag = f"[GPU {gpu_id} | #{idx}] {video_id}"

            if os.path.exists(final_output):
                print(f"{tag} SKIP (output appeared)")
                write_status(fh, {
                    "video_id": video_id,
                    "prompt_word": prompt,
                    "input_video_path": video_path,
                    "output_video_path": final_output,
                    "status": "skipped",
                    "error_message": "output already exists",
                })
                continue

            try:
                vinfo = get_video_info(video_path)
            except Exception as exc:
                print(f"{tag} FAIL (video info: {exc})")
                write_status(fh, {
                    "video_id": video_id,
                    "prompt_word": prompt,
                    "input_video_path": video_path,
                    "status": "failed",
                    "error_message": f"get_video_info failed: {exc}",
                })
                continue

            duration = vinfo.get("duration_sec", 0)
            fps = vinfo.get("fps", 30)
            width = vinfo.get("width", 0)
            height = vinfo.get("height", 0)
            frame_reduction = duration > max_duration

            max_frames = None
            uniform_subsample = False
            if frame_reduction:
                max_frames = int(max_duration * fps)
                uniform_subsample = True

            if dry_run:
                print(f"{tag} DRY-RUN  prompt={prompt!r}  "
                      f"dur={duration:.1f}s  reduction={frame_reduction}")
                write_status(fh, {
                    "video_id": video_id,
                    "prompt_word": prompt,
                    "input_video_path": video_path,
                    "output_video_path": final_output,
                    "duration_seconds": round(duration, 2),
                    "frame_reduction_applied": frame_reduction,
                    "status": "dry_run",
                    "error_message": None,
                })
                continue

            # Use a work dir under output_root instead of system /tmp so
            # orphaned dirs (from OOM kills) stay visible and cleanable.
            batch_tmp = os.path.join(output_root, "_work")
            os.makedirs(batch_tmp, exist_ok=True)
            work_dir = tempfile.mkdtemp(prefix=f"batch_{video_id}_", dir=batch_tmp)
            t0 = time.time()
            try:
                print(f"{tag} START  prompt={prompt!r}  dur={duration:.1f}s  "
                      f"reduction={frame_reduction}")
                result = run_pipeline(
                    input_video_path=video_path,
                    text_prompt=prompt,
                    edit_type="remove",
                    output_dir=work_dir,
                    max_frames=max_frames,
                    uniform_subsample=uniform_subsample,
                    skip_postprocess=True,
                    edit_backend=edit_backend,
                    diffueraser_path=diffueraser_path,
                    use_dino_tracker=use_dino_tracker,
                    dino_verify=dino_verify,
                    _preloaded_segmenter=segmenter,
                    _preloaded_diffueraser_runner=diffueraser_runner,
                    _preloaded_dino_tracker=dino_tracker if dino_verify else None,
                )

                edited_video = result.get("edited_video", "")
                if not edited_video or not os.path.exists(edited_video):
                    raise RuntimeError(
                        f"Pipeline finished but edited video not found at {edited_video}"
                    )

                os.makedirs(os.path.dirname(final_output), exist_ok=True)
                # Crop back to the original source resolution to eliminate any
                # macro-block padding added by imageio/ffmpeg. Systematic edge
                # artifacts at identical positions across all outputs can become
                # spurious learned cues for VLMs ("this was edited").
                _match_resolution(edited_video, final_output, width, height)

                elapsed = time.time() - t0
                print(f"{tag} SUCCESS  ({elapsed:.1f}s)")
                write_status(fh, {
                    "video_id": video_id,
                    "prompt_word": prompt,
                    "input_video_path": video_path,
                    "output_video_path": final_output,
                    "duration_seconds": round(duration, 2),
                    "frame_reduction_applied": frame_reduction,
                    "status": "success",
                    "error_message": None,
                })

            except Exception as exc:
                elapsed = time.time() - t0
                tb = traceback.format_exc()
                print(f"{tag} FAIL  ({elapsed:.1f}s) {exc}")
                print(tb)
                write_status(fh, {
                    "video_id": video_id,
                    "prompt_word": prompt,
                    "input_video_path": video_path,
                    "output_video_path": final_output,
                    "duration_seconds": round(duration, 2),
                    "frame_reduction_applied": frame_reduction,
                    "status": "failed",
                    "error_message": str(exc),
                })
                # Release GPU memory leaked by the failed run so the next
                # video starts with a clean slate.
                try:
                    import gc
                    gc.collect()
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass

            finally:
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description="Batch video inpainting with SAM3 + DiffuEraser",
    )
    parser.add_argument(
        "--data-json",
        default="data.json",
        help="Path to data.json with video_id -> prompt_word mapping",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory containing video files",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory for final inpainted videos",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs (0 = auto-detect)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Max video duration in seconds; longer videos get uniformly subsampled",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without running the pipeline",
    )
    parser.add_argument(
        "--diffueraser-path",
        default="third_party/DiffuEraser",
        help="Path to DiffuEraser source tree",
    )
    parser.add_argument(
        "--edit-backend",
        default="diffueraser",
        choices=["diffueraser", "diffusion", "telea"],
        help="Inpainting backend (default: diffueraser).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N videos (0 = all)",
    )
    parser.add_argument(
        "--sam3-video-tracker",
        "--use-dino-tracker",
        dest="use_dino_tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "SAM3 Tracker Video bidirectional propagation (temporally consistent "
            "masks on all frames). Default: ON. Use --no-sam3-video-tracker to "
            "disable and fall back to per-frame text segmentation."
        ),
    )
    parser.add_argument(
        "--dino-verify",
        action="store_true",
        default=False,
        help="DINOv3 verification after SAM3 tracker (requires --sam3-video-tracker)",
    )
    args = parser.parse_args()

    # --- Resolve diffueraser path to absolute (avoids breakage when workers chdir) ---
    args.diffueraser_path = str(Path(args.diffueraser_path).resolve())

    # --- Load data ---
    print(f"Loading data from {args.data_json} ...")
    data = load_data_json(args.data_json)
    if args.limit > 0:
        data = data[: args.limit]
    print(f"  {len(data)} total entries")

    # --- Build video index ---
    print(f"Indexing videos under {args.dataset_root} ...")
    video_index = build_video_index(args.dataset_root)
    print(f"  {len(video_index)} video files found")

    # --- Pre-filter: only keep jobs that actually need work ---
    actionable, skipped_records = filter_jobs(data, video_index, args.output_root,
                                              probe_frame_count=True)
    print(f"  Pre-filter: {len(actionable)} to process, {len(skipped_records)} skipped")

    if not actionable:
        print("Nothing to do -- all videos are already processed or skipped.")
        return

    # --- Pre-download SAM3 weights before spawning workers ---
    # Downloads model files to the HuggingFace cache without loading them
    # into memory or initializing CUDA (which would break forked workers).
    if not args.dry_run:
        print("Pre-caching SAM3 model weights ...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download("facebook/sam3")
            print("  SAM3 weights cached.")
        except Exception as exc:
            print(f"  WARNING: SAM3 pre-cache failed ({exc}). Workers will download individually.")

    # --- Determine GPU count ---
    num_gpus = args.num_gpus
    if num_gpus <= 0:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except Exception:
            num_gpus = 1
    if num_gpus <= 0:
        num_gpus = 1
    num_gpus = min(num_gpus, len(actionable))
    print(f"  Using {num_gpus} GPU(s)")

    # --- Dynamic scheduling via shared queue ---
    # Sort longest-first so large jobs start early and short ones fill gaps.
    actionable.sort(key=lambda j: -int(j.get("frame_count", 0)))

    job_queue: Queue = Queue()
    for job in actionable:
        job_queue.put(job)

    total_jobs = len(actionable)
    print(f"  Shared queue: {total_jobs} jobs (longest-first)")

    cpu_count = os.cpu_count() or 8
    threads_per_worker = max(2, cpu_count // num_gpus)
    print(f"  CPU cores: {cpu_count}, threads per worker: {threads_per_worker}")

    # --- Write pre-filter skips to results upfront ---
    os.makedirs(args.output_root, exist_ok=True)
    result_files = [
        os.path.join(args.output_root, f"_results_gpu{g}.jsonl")
        for g in range(num_gpus)
    ]

    # --- Shared stop event for graceful shutdown ---
    stop_event = Event()

    def _shutdown_handler(signum, frame):
        name = signal.Signals(signum).name
        print(f"\n[main] Received {name} — telling workers to stop after current video ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # --- Launch workers ---
    processes: List[Process] = []
    for g in range(num_gpus):
        p = Process(
            target=worker_fn,
            args=(
                g,
                job_queue,
                total_jobs,
                args.output_root,
                args.max_duration,
                result_files[g],
                args.dry_run,
                args.diffueraser_path,
                threads_per_worker,
                stop_event,
                args.use_dino_tracker,
                args.dino_verify,
                args.edit_backend,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # --- Merge results ---
    merged_path = os.path.join(args.output_root, "results.jsonl")
    total = success = skipped = failed = 0
    with open(merged_path, "w") as out:
        for rec in skipped_records:
            out.write(json.dumps(rec) + "\n")
            total += 1
            skipped += 1
        for rf in result_files:
            if not os.path.exists(rf):
                continue
            with open(rf, "r") as inp:
                for line in inp:
                    out.write(line)
                    rec = json.loads(line)
                    total += 1
                    s = rec.get("status", "")
                    if s == "success":
                        success += 1
                    elif s in ("skipped", "dry_run"):
                        skipped += 1
                    else:
                        failed += 1
            os.remove(rf)

    print("\n" + "=" * 60)
    print("Batch complete")
    print(f"  Total:   {total}")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Results: {merged_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
