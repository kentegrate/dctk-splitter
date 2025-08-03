import cv2, json, argparse, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict
import multiprocessing as mp
from models import MetaData, Clip, parse_timestamp
import re
import signal
import sys


@dataclass
class ClipBuffer:
    """Holds state for a single clip being written to disk."""
    clip: Clip
    output_path: str
    writer: cv2.VideoWriter = None
    is_finished: bool = False


def group(meta: MetaData, user_filter=None, session_filter=None):
    """Return {clean_file_name : [Clip, …]}"""
    by_file = defaultdict(list)
    for clip in meta.get_valid_clips():
        if user_filter and clip.summary.userId != user_filter:
            continue
        if session_filter and clip.full.data.sessionId != session_filter:
            continue
        fn = clip.summary.filename
        by_file[fn].append(clip)

    for fn in by_file:
        clips = by_file[fn]
        for clip in clips:
            if clip.summary.start_s is None:
                start_time = (
                    parse_timestamp(clip.full.data.startButtonDownTimestamp) or
                    parse_timestamp(clip.full.data.startButtonUpTimestamp)
                )
                video_start = parse_timestamp(clip.full.data.videoStart)
                clip.summary.start_s = (start_time - video_start).total_seconds()
                clip.summary.end_s = -1  # -1 means "until end of video"

        # Sort by start time
        by_file[fn].sort(key=lambda s: s.summary.start_s)

    return by_file


def create_output_path(dest_dir: str, clip: Clip) -> str:
    """Create sanitized output path for clip"""
    original_filename = clip.summary.filename
    prompt_text = clip.summary.promptText or "untitled"

    # Sanitize prompt text for filename
    sanitized_prompt = re.sub(r'[^\w\-_\. ]', '_', prompt_text)
    sanitized_prompt = sanitized_prompt.strip()[:50]  # Limit length

    name, ext = os.path.splitext(original_filename)
    new_filename = f"{name}-{sanitized_prompt}{ext}"
    new_filename = f"{name}_clip_{clip.summary.start_s}_{clip.summary.end_s}{ext}"
    return os.path.join(dest_dir, new_filename)


def check_clips_exist(dest_dir: str, clips: List[Clip]) -> bool:
    """
    Check if any clips from this file have already been processed.
    Returns True if clips exist (should skip), False if should process.
    Same logic as clip_video.py: if any clip exists, skip the entire file.
    """
    for clip in clips:
        output_path = create_output_path(dest_dir, clip)
        if os.path.exists(output_path):
            return True
    return False


def process_one_file(args_tuple):
    """
    Process a single video file by writing frames directly to clip files,
    avoiding buffering frames in memory.
    """
    src_dir, dest_dir, file_name, clips, skip_existing = args_tuple

    # Check if clips already exist (same logic as clip_video.py)
    if skip_existing and check_clips_exist(dest_dir, clips):
        print(f"Skipping existing clips for: {file_name}. Skipping processing the whole mp4 file.")
        return True

    userId = clips[0].summary.userId
    path = os.path.join(src_dir, userId, "upload", file_name)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Cannot open {path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    print(f"Processing {file_name}: {len(clips)} clips, FPS: {fps}")

    # Create clip buffers with pre-determined output paths
    clip_buffers = [
        ClipBuffer(clip=c, output_path=create_output_path(dest_dir, c))
        for c in clips
    ]
    
    frame_count = 0
    total_clips = len(clip_buffers)
    finished_clips = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or finished_clips == total_clips:
                break

            frame_count += 1
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            for buffer in clip_buffers:
                if buffer.is_finished:
                    continue

                clip = buffer.clip
                start_s = clip.summary.start_s
                end_s = clip.summary.end_s

                # Check if the frame is within the clip's time range
                is_active = start_s <= current_time and (end_s == -1 or current_time <= end_s)

                if is_active:
                    # If this is the first frame, create the video writer
                    if buffer.writer is None:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        buffer.writer = cv2.VideoWriter(buffer.output_path, fourcc, fps, (width, height))
                        if not buffer.writer.isOpened():
                            print(f"Error: Could not open video writer for {buffer.output_path}")
                            buffer.is_finished = True # Mark as finished to avoid retrying
                            finished_clips += 1
                            continue
                    
                    # Write the frame directly to the file
                    buffer.writer.write(frame)

                # If the clip has passed its end time and has an open writer, close it
                elif current_time > end_s and end_s != -1 and buffer.writer is not None:
                    buffer.writer.release()
                    print(f"Video saved: {buffer.output_path}")
                    buffer.is_finished = True
                    finished_clips += 1
    
    except KeyboardInterrupt:
        print(f"Process interrupted during processing of {file_name}")
        raise

    finally:
        # Ensure all writers and the capture object are released
        for buffer in clip_buffers:
            if buffer.writer is not None and not buffer.is_finished:
                buffer.writer.release()
                print(f"Video saved: {buffer.output_path}")
        cap.release()

    print(f"Completed {file_name}: processed {frame_count} frames")
    return True


def process_files_parallel(src_dir: str, dest_dir: str, by_file: Dict, max_workers: int = None, skip_existing: bool = True):
    """Process multiple files in parallel"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(by_file))

    # Filter out files that should be skipped
    files_to_process = {}
    skipped_files = []
    
    for file_name, clips in by_file.items():
        if skip_existing and check_clips_exist(dest_dir, clips):
            skipped_files.append(file_name)
            print(f"Skipping existing clips for: {file_name}")
        else:
            files_to_process[file_name] = clips
    
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files with existing clips")
    
    if not files_to_process:
        print("No files to process - all clips already exist")
        return

    # Prepare arguments for each file to process
    file_args = [
        (src_dir, dest_dir, file_name, clips, skip_existing)
        for file_name, clips in files_to_process.items()
    ]

    print(f"Processing {len(file_args)} files with {max_workers} workers")

    # Process files in parallel
    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        results = list(executor.map(process_one_file, file_args))
        successful = sum(1 for r in results if r)
        print(f"Successfully processed {successful}/{len(file_args)} files")
    except KeyboardInterrupt:
        print("Keyboard interrupt received, cancelling tasks...")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not executor._shutdown:
            executor.shutdown(wait=True)


def main(args):
    try:
        os.makedirs(args.output, exist_ok=True)
        with open(args.metadata) as fh:
            meta = MetaData.from_json(fh.read())

        by_file = group(meta, user_filter=args.user, session_filter=args.session)
        if not by_file:
            print("No clips found to process")
            return

        print(f"Found {len(by_file)} files with clips")
        for file_name, clips in by_file.items():
            print(f"  {file_name}: {len(clips)} clips")

        if args.parallel and len(by_file) > 1:
            process_files_parallel(args.input, args.output, by_file, args.workers, not args.no_skip)
        else:
            for file_name, clips in by_file.items():
                print(f"\nProcessing {file_name}...")
                try:
                    success = process_one_file((args.input, args.output, file_name, clips, not args.no_skip))
                    if not success:
                        print(f"Failed to process {file_name}")
                except KeyboardInterrupt:
                    print(f"Interrupted while processing {file_name}")
                    break

        print("Processing completed successfully")

    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down...")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print('\nReceived interrupt signal, shutting down...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    p = argparse.ArgumentParser(description="Extract video clips from source videos")
    p.add_argument("--metadata", type=str, required=True, help="Path to metadata JSON file")
    p.add_argument("--input", type=str, default="/in", help="Input directory")
    p.add_argument("--output", type=str, default="/out", help="Output directory")
    p.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    p.add_argument("--workers", type=int, default=2, help="Number of worker processes")
    p.add_argument("--user", type=str, default=None, help="Filter by user id")
    p.add_argument("--session", type=str, default=None, help="Filter by session id")
    p.add_argument("--no-skip", action="store_true", help="Don't skip existing clips (process all files)")
    
    args = p.parse_args()
    main(args)