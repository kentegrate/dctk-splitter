import cv2, json, argparse, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict
import multiprocessing as mp
from models import MetaData, Clip, parse_timestamp
import re
import signal
import sys


@dataclass
class ClipBuffer:
    clip: Clip
    frames: List
    active: bool = True


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
    return os.path.join(dest_dir, new_filename)


def write_clip_video(dest_dir: str, clip: Clip, frames: List, fps: float):
    """Write frames to video file"""
    if not frames:
        print(f"Warning: No frames for clip {clip.summary.promptText}")
        return
    
    output_path = create_output_path(dest_dir, clip)
    height, width = frames[0].shape[:2]
    
    # Use better codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    print(f"Video saved: {output_path} ({len(frames)} frames)")


def process_one_file(args_tuple):
    """Process a single video file - designed for multiprocessing"""
    src_dir, dest_dir, file_name, clips, write_workers = args_tuple
    
    # Handle filename with colons
    safe_filename = file_name.replace(":", "_")
    path = os.path.join(src_dir, safe_filename)
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Cannot open {path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    
    print(f"Processing {file_name}: {len(clips)} clips, FPS: {fps}")
    
    # Create clip buffers
    clip_buffers = [ClipBuffer(clip=clip, frames=[]) for clip in clips]
    active_clips = clip_buffers.copy()
    
    # Create thread pool for writing videos
    write_executor = ThreadPoolExecutor(max_workers=write_workers)
    futures = []
    
    frame_count = 0
    
    try:
        while active_clips:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Check which clips should include this frame
            clips_to_remove = []
            
            for clip_buffer in active_clips:
                clip = clip_buffer.clip
                start_s = clip.summary.start_s
                end_s = clip.summary.end_s
                
                # Check if we've reached the start of this clip
                if current_time >= start_s:
                    # Check if we're still within the clip duration
                    if end_s == -1 or current_time <= end_s:
                        # Add frame to this clip
                        clip_buffer.frames.append(frame.copy())
                    else:
                        # We've passed the end of this clip
                        clips_to_remove.append(clip_buffer)
                
            # Remove completed clips and submit their videos for writing
            for clip_buffer in clips_to_remove:
                active_clips.remove(clip_buffer)
                # Take ownership of the frames list
                frames_to_write = clip_buffer.frames.copy()
                clip_buffer.frames.clear()
                # Submit the write job to thread pool
                future = write_executor.submit(
                    write_clip_video, 
                    dest_dir, 
                    clip_buffer.clip, 
                    frames_to_write, 
                    fps
                )
                futures.append(future)
        
        # Handle any remaining active clips (reached end of video)
        for clip_buffer in active_clips:
            if clip_buffer.frames:
                frames_to_write = clip_buffer.frames.copy()
                future = write_executor.submit(
                    write_clip_video,
                    dest_dir,
                    clip_buffer.clip,
                    frames_to_write,
                    fps
                )
                futures.append(future)
        
        # Wait for all video writes to complete
        for future in futures:
            try:
                future.result(timeout=30)  # 30 second timeout per write
            except Exception as e:
                print(f"Writer error: {e}")
    
    except KeyboardInterrupt:
        print(f"Process interrupted during processing of {file_name}")
        # Cancel any pending futures
        for future in futures:
            future.cancel()
        write_executor.shutdown(wait=False)
        raise
    
    finally:
        # Clean up resources
        cap.release()
        if not write_executor._shutdown:
            write_executor.shutdown(wait=True)
    
    print(f"Completed {file_name}: processed {frame_count} frames")
    return True


def process_files_parallel(src_dir: str, dest_dir: str, by_file: Dict, max_workers: int = None, write_workers: int = 2):
    """Process multiple files in parallel"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(by_file))
    
    # Prepare arguments for each file
    file_args = [
        (src_dir, dest_dir, file_name, clips, write_workers)
        for file_name, clips in by_file.items()
    ]
    
    print(f"Processing {len(file_args)} files with {max_workers} workers, {write_workers} write workers per file")
    
    # Process files in parallel
    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        results = list(executor.map(process_one_file, file_args))
        successful = sum(results)
        print(f"Successfully processed {successful}/{len(file_args)} files")
    except KeyboardInterrupt:
        print("Keyboard interrupt received, cancelling tasks...")
        # Shutdown executor immediately without waiting
        executor.shutdown(wait=False)
        raise
    finally:
        # Ensure executor is properly closed
        if not executor._shutdown:
            executor.shutdown(wait=True)


def main(args):
    try:
        # Ensure output directory exists
        os.makedirs(args.output, exist_ok=True)
        
        # Load metadata
        with open(args.metadata) as fh:
            meta = MetaData.from_json(fh.read())
        

        by_file = group(meta, user_filter=args.user, session_filter=args.session)
        
        if not by_file:
            print("No clips found to process")
            return
        
        print(f"Found {len(by_file)} files with clips")
        for file_name, clips in by_file.items():
            print(f"  {file_name}: {len(clips)} clips")
        
        # Process files
        if args.parallel and len(by_file) > 1:
            process_files_parallel(args.input, args.output, by_file, args.workers, args.write_workers)
        else:
            # Sequential processing
            for file_name, clips in by_file.items():
                print(f"\nProcessing {file_name}...")
                try:
                    success = process_one_file((args.input, args.output, file_name, clips, args.write_workers))
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
    # Set up signal handling for better Ctrl+C behavior
    def signal_handler(sig, frame):
        print('\nReceived interrupt signal, shutting down...')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    p = argparse.ArgumentParser(description="Extract video clips from source videos")
    p.add_argument("--metadata", type=str, required=True, help="Path to metadata JSON file")
    p.add_argument("--input", type=str, default="/in", help="Input directory")
    p.add_argument("--output", type=str, default="/out", help="Output directory")
    p.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    p.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    p.add_argument("--user", type=str, default=None, help="Filter by user id")
    p.add_argument("--session", type=str, default=None, help="Filter by session id")
    p.add_argument("--write-workers", type=int, default=2, help="Number of write workers per video process")
    
    args = p.parse_args()
    main(args)
