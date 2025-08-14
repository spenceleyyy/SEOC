# One-file pipeline: stitch -> track people (stable IDs) -> grid overlay -> export
print("[DEBUG] person_tracking_stitch_grid.py loaded successfully")
#
# Usage (Python):
#     python person_tracking_stitch_grid.py \
#         --inputs /path/to/clip1.mp4 /path/to/clip2.mp4 \
#         --out merged_tracked.mp4 \
#         --grid-rows 3 --grid-cols 3 \
#         --model yolov8n.pt \
#         --conf 0.25 \
#         --save-csv tracks.csv
#
# In Google Colab (example cell):
#     !pip install -q ultralytics opencv-python pandas
#     from google.colab import drive; drive.mount('/content/drive')
#     !python person_tracking_stitch_grid.py \
#         --inputs "/content/drive/MyDrive/vids/a.mp4" "/content/drive/MyDrive/vids/b.mp4" \
#         --out "/content/drive/MyDrive/vids/out_tracked.mp4" \
#         --grid-rows 3 --grid-cols 4 --model yolov8n.pt --conf 0.3 --save-csv "/content/drive/MyDrive/vids/tracks.csv"
#
# Notes:
#   - IDs are provided by ByteTrack via Ultralytics with persist=True so they remain stable across the single stitched stream.
#   - If your inputs have different sizes/fps, they are normalized to the first clip.
#   - You can swap models (e.g., yolov8s.pt, yolov11n.pt) as long as Ultralytics supports them.

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import time


# Lazy import ultralytics so help/usage works even if not installed yet
def _import_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception as e:
        print("ERROR: ultralytics not installed. Install with `pip install ultralytics`.", file=sys.stderr)
        raise

# Lazy import deep_sort_realtime only if user selects DeepSORT
def _import_deepsort():
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        return DeepSort
    except Exception as e:
        raise ImportError(
            "DeepSORT not available. Install with: pip install deep-sort-realtime\n"
            "Then run with --tracker deepsort"
        )

def draw_grid(frame, rows: int, cols: int, thickness: int = 1):
    """Draw gridlines over the frame."""
    if rows <= 0 and cols <= 0:
        return frame
    h, w = frame.shape[:2]
    if cols > 0:
        step_x = w / cols
        for c in range(1, cols):
            x = int(round(c * step_x))
            cv2.line(frame, (x, 0), (x, h), (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    if rows > 0:
        step_y = h / rows
        for r in range(1, rows):
            y = int(round(r * step_y))
            cv2.line(frame, (0, y), (w, y), (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return frame

def concat_videos(input_paths, out_path, resize_to=None, fps=None):
    """
    Concatenate videos by re-encoding frames. Normalizes size/fps to the first clip by default.
    Returns: (width, height, fps_used, total_frames, path_out)
    """
    if len(input_paths) == 1:
        # If only one input, still re-encode to ensure consistent codec and to avoid ByteTrack reset between clips
        pass

    caps = []
    for p in input_paths:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {p}")
        caps.append(cap)

    print(f"[concat] Starting concat. Inputs: {len(input_paths)} -> {out_path}")
    for idx, p in enumerate(input_paths):
        c = caps[idx]
        w_i = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_i = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_i = c.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"[concat] Input {idx+1}: {p} size=({w_i}x{h_i}) fps={fps_i}")

    # Determine target size and FPS
    if resize_to is None or fps is None:
        w0 = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps0 = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
        if resize_to is None:
            resize_to = (w0, h0)
        if fps is None:
            fps = fps0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, resize_to)
    if not writer.isOpened():
        print(f"[concat] mp4v failed to open {out_path}. Retrying with avc1 (H.264)...")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, resize_to)
    if not writer.isOpened():
        raise RuntimeError(
            f"[concat] Failed to open VideoWriter for {out_path}. Tried codecs: mp4v, avc1."
        )

    total_frames = 0
    for p, cap in zip(input_paths, caps):
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != resize_to[0] or frame.shape[0] != resize_to[1]:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            total_frames += 1
            if total_frames % 30 == 0:
                print(f"[concat] Frames appended: {total_frames}")
        print(f"[concat] Finished segment: {p}, frames so far: {total_frames}")
        cap.release()

    writer.release()

    if total_frames == 0:
        raise RuntimeError("[concat] No frames were concatenated. Verify input videos are readable.")
    try:
        import os
        size_mb = os.path.getsize(str(out_path)) / 1_000_000
        print(f"[concat] Wrote stitched video: {out_path} ({size_mb:.2f} MB)")
    except Exception:
        pass

    return resize_to[0], resize_to[1], fps, total_frames, out_path

def track_people(stitched_path, out_path, model_name="yolov8n.pt", conf=0.25, iou=0.45, grid_rows=0, grid_cols=0, save_csv=None, imgsz=640, max_det=300, agnostic_nms=False, tracker_name="deepsort", max_frames=0):
    """
    Run YOLO + ByteTrack on stitched video, draw boxes & ID labels & gridlines, write annotated video.
    Optionally writes a CSV with per-frame detections & IDs.
    """
    YOLO = _import_ultralytics()
    model = YOLO(model_name)

    # Normalize imgsz to a (w,h) tuple; some code paths call len(imgsz)
    if isinstance(imgsz, int):
        _imgsz = (imgsz, imgsz)
    elif isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            _imgsz = (int(imgsz[0]), int(imgsz[0]))
        else:
            _imgsz = (int(imgsz[0]), int(imgsz[1]))
    else:
        try:
            _imgsz = (int(imgsz), int(imgsz))
        except Exception:
            _imgsz = (640, 640)

    # Prepare writer (lazy init) and FPS from source
    writer = None
    writer_size = None
    writer_fps = 30.0
    try:
        src_fps_probe = cv2.VideoCapture(str(stitched_path)).get(cv2.CAP_PROP_FPS)
        if src_fps_probe and src_fps_probe > 0:
            writer_fps = src_fps_probe
    except Exception:
        pass

    # If user requested DeepSORT, run manual detection + DeepSORT tracking
    use_deepsort = str(tracker_name).lower() in {"deepsort", "deep_sort"}
    if use_deepsort:
        print("[track] Using DeepSORT for ID tracking.")
        DeepSort = _import_deepsort()
        # Reasonable defaults; tweak as needed
        ds_tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7, nms_max_overlap=1.0,
                              embedder='mobilenet', half=True, bgr=True)
        cap = cv2.VideoCapture(str(stitched_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open stitched video: {stitched_path}")
        frame_iter = iter(int, 1)  # dummy placeholder, we'll use while loop below
        deepsort_mode = True
    else:
        print("[track] Initializing Ultralytics tracking stream ...")
        results_stream = model.track(
            source=str(stitched_path),
            stream=True,
            conf=conf,
            iou=iou,
            imgsz=_imgsz,
            max_det=max_det,
            agnostic_nms=agnostic_nms,
            classes=[0],
            persist=True,
            tracker=tracker_name
        )
        print("[track] YOLO stream created. Beginning frame iteration ...")
        deepsort_mode = False

    csv_rows = []
    frame_idx = -1
    frames_written = 0
    print(f"[track] Input stitched video: {stitched_path}")
    print(f"[track] Intended output path: {out_path}")
    t0 = time.time()
    if deepsort_mode:
        # Manual read + detect + DeepSORT update
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                approx_fps = frames_written / elapsed if frames_written else frame_idx / elapsed
                print(f"[track] Processing frame {frame_idx} | elapsed {elapsed:.1f}s | ~{approx_fps:.2f} fps")
            if max_frames and frame_idx >= max_frames:
                print(f"[track] Reached max_frames={max_frames}; stopping early for debug.")
                break

            # Run detection for persons only
            det_res = model.predict(frame, conf=conf, iou=iou, imgsz=_imgsz, classes=[0], verbose=False)
            res0 = det_res[0]
            boxes_np = res0.boxes.xyxy.cpu().numpy().astype(int) if res0.boxes is not None else np.empty((0,4), int)
            confs_np = res0.boxes.conf.cpu().numpy() if getattr(res0.boxes, 'conf', None) is not None else np.zeros((len(boxes_np),), dtype=float)

            # Build detections for DeepSORT (expects: ([x1,y1,x2,y2], conf, class))
            dets = []
            for (x1, y1, x2, y2), c in zip(boxes_np, confs_np):
                dets.append(([int(x1), int(y1), int(x2), int(y2)], float(c), 'person'))

            tracks = ds_tracker.update_tracks(dets, frame=frame)

            # Lazy writer init
            if writer is None:
                hh, ww = frame.shape[:2]
                writer_size = (ww, hh)
                print(f"[track] Opening VideoWriter -> path={out_path} size=({ww}x{hh}) fps={writer_fps}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, writer_size)
                if not writer.isOpened():
                    print("[track] mp4v failed to open. Retrying with avc1 (H.264) ...")
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, writer_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

            # Draw grid
            if grid_rows > 0 or grid_cols > 0:
                draw_grid(frame, grid_rows, grid_cols, thickness=1)

            # Draw DeepSORT tracks and collect CSV
            for trk in tracks:
                if not trk.is_confirmed() or trk.time_since_update > 0:
                    continue
                x1, y1, x2, y2 = map(int, trk.to_tlbr())
                track_id = int(trk.track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                if save_csv:
                    csv_rows.append({
                        "frame": frame_idx,
                        "id": track_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(1.0)  # DeepSORT tracks don't carry a single conf; keep placeholder
                    })

            writer.write(frame)
            frames_written += 1
    else:
        # Ultralytics trackers branch (existing behavior)
        for r in results_stream:
            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                approx_fps = frames_written / elapsed if frames_written else frame_idx / elapsed
                print(f"[track] Processing frame {frame_idx} | elapsed {elapsed:.1f}s | ~{approx_fps:.2f} fps")
            if max_frames and frame_idx >= max_frames:
                print(f"[track] Reached max_frames={max_frames}; stopping early for debug.")
                break
            frame = r.orig_img.copy()
            # Lazy writer init
            if writer is None:
                import os
                hh, ww = frame.shape[:2]
                writer_size = (ww, hh)
                print(f"[DEBUG] Attempting to write video to: {out_path}")
                print(f"[DEBUG] VideoWriter size: ({ww}, {hh}), FPS: {writer_fps}")
                print(f"[track] Opening VideoWriter -> path={out_path} size=({ww}x{hh}) fps={writer_fps}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, writer_size)
                if not writer.isOpened():
                    print("[track] mp4v failed to open. Retrying with avc1 (H.264) ...")
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, writer_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter for {out_path}")
            else:
                if (frame.shape[1], frame.shape[0]) != writer_size:
                    frame = cv2.resize(frame, writer_size, interpolation=cv2.INTER_LINEAR)

            # Draw grid
            if grid_rows > 0 or grid_cols > 0:
                draw_grid(frame, grid_rows, grid_cols, thickness=1)

            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.array([-1]*len(boxes))
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.zeros(len(boxes))
                for (x1, y1, x2, y2), track_id, c in zip(boxes, ids, confs):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID {track_id if track_id>=0 else '?'}  {c:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    if save_csv:
                        csv_rows.append({
                            "frame": frame_idx,
                            "id": int(track_id),
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                            "conf": float(c)
                        })

            print(f"[DEBUG] Writing frame {frame_idx} to {out_path}")
            writer.write(frame)
            frames_written += 1

    import os
    print(f"[DEBUG] Finished writing video: {out_path}, exists: {os.path.exists(out_path)}")
    print(f"[track] Total frames_written so far: {frames_written}")
    if writer is not None:
        writer.release()
        print(f"[DEBUG] Video writer released. Output should be at: {out_path}")
        print(f"[DEBUG] Checking if file exists: {Path(out_path).exists()}")
    try:
        if use_deepsort:
            cap.release()
    except Exception:
        pass

    print(f"[track] Frames written: {frames_written}")
    if frames_written == 0:
        raise RuntimeError(
            "No frames were written to the output video. Possible causes: "
            "(a) the tracker produced no frames, (b) input video couldn't be read, "
            "or (c) codec issues. Try a different model, verify input videos, or change codec."
        )

    if save_csv and len(csv_rows) > 0:
        df = pd.DataFrame(csv_rows)
        df.to_csv(save_csv, index=False)

    return out_path

def main():
    ap = argparse.ArgumentParser(description="Stitch multiple videos, track people with stable IDs, draw gridlines, and export annotated video.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input video files (MP4/AVI/etc).")
    ap.add_argument("--out", required=True, help="Output annotated video path (MP4).")
    ap.add_argument("--temp-stitched", default="_stitched_tmp.mp4", help="Temporary stitched video path (will be overwritten).")
    ap.add_argument("--grid-rows", type=int, default=0, help="Number of horizontal grid divisions (rows). 0 to disable.")
    ap.add_argument("--grid-cols", type=int, default=0, help="Number of vertical grid divisions (cols). 0 to disable.")
    ap.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model name or path.")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    ap.add_argument("--save-csv", default=None, help="Optional path to save per-frame tracks as CSV.")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size (pixels). Try 960 or 1280 to detect smaller persons.")
    ap.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    ap.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS (can help when only 'person' class is used).")
    ap.add_argument("--tracker", default="deepsort", help="Tracker to use for IDs (deepsort, bytetrack.yaml, botsort.yaml).")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, stop after this many frames (debug/perf sanity). 0 = no limit.")
    args = ap.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            ap.error(f"Input not found: {p}")


    temp_stitched = Path(args.temp_stitched)
    # Stitch inputs
    print(f"[main] Staging stitched video at: {temp_stitched}")
    w, h, fps, total_frames, stitched_path = concat_videos(input_paths, temp_stitched)

    # Track on stitched video and overlay gridlines
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Drive-safe: if writing to Google Drive, write locally first and then copy
    write_local_first = str(out_path).startswith("/content/drive/")
    local_tmp = Path("/tmp/_out_tmp.mp4")
    effective_out = local_tmp if write_local_first else out_path

    print(f"[main] Writing annotated output to: {effective_out}")
    track_people(
        stitched_path=str(stitched_path),
        out_path=str(effective_out),
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        save_csv=args.save_csv,
        imgsz=args.imgsz,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        tracker_name=args.tracker,
        max_frames=args.max_frames
    )

    # If we wrote locally first, copy to Drive now
    if write_local_first:
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if effective_out.exists():
            import shutil
            shutil.copyfile(str(effective_out), str(out_path))
            print(f"[main] Copied local output to Drive: {out_path}")
        else:
            print(f"[main] WARNING: Local temp output missing: {effective_out}")

    # Cleanup temp if out != temp
    try:
        if out_path.resolve() != temp_stitched.resolve():
            temp_stitched.unlink(missing_ok=True)
    except Exception:
        pass

    if not out_path.exists():
        print(f"[main] WARNING: Expected output does not exist yet: {out_path}")

    print(f"Done. Wrote annotated video to: {out_path}")
    if args.save_csv:
        print(f"Wrote tracks CSV to: {args.save_csv}")

if __name__ == "__main__":
    main()
