"""
SportsAlgo Hackathon — Layer 1: Player Detection & Within-Match Tracking
Uses YOLOv8 for detection + ByteTrack for tracking
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import defaultdict

# pip install ultralytics supervision
from ultralytics import YOLO
import supervision as sv


@dataclass
class Detection:
    frame_id: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    confidence: float
    cx: float  # center x
    cy: float  # center y


class PlayerDetectorTracker:
    """
    Detects and tracks players within a single match video.
    Output: per-frame bounding boxes with consistent Track IDs
    """

    def __init__(self, model_path: str = "yolov8n.pt", use_pretrained_detections: bool = False):
        """
        Args:
            model_path: Path to YOLOv8 weights (fine-tuned preferred)
            use_pretrained_detections: If True, load pre-computed detections from file
        """
        self.model = YOLO(model_path)
        self.use_pretrained = use_pretrained_detections

        # ByteTrack via supervision
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,       # Keep lost tracks for 60 frames (~2.4s at 25fps)
            minimum_matching_threshold=0.8,
            frame_rate=25
        )

        self.all_detections: List[Detection] = []

    def load_pretrained_detections(self, detection_file: str) -> Dict:
        """Load pre-computed YOLO detections provided by SportsAlgo"""
        with open(detection_file, 'r') as f:
            return json.load(f)

    def detect_frame(self, frame: np.ndarray, frame_id: int) -> sv.Detections:
        """Run YOLOv8 on a single frame, return player detections only"""
        results = self.model(frame, verbose=False)[0]

        # Filter: class 0 = person in COCO, confidence > 0.3
        detections = sv.Detections.from_ultralytics(results)
        mask = (detections.class_id == 0) & (detections.confidence > 0.3)
        return detections[mask]

    def track_frame(self, detections: sv.Detections) -> sv.Detections:
        """Apply ByteTrack to assign/maintain Track IDs"""
        return self.tracker.update_with_detections(detections)

    def crop_player(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        """Crop a player bounding box from frame for re-ID"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return frame[y1:y2, x1:x2]

    def process_video(self, video_path: str, output_dir: str,
                      pretrained_detections_path: str = None,
                      save_crops: bool = True) -> List[Detection]:
        """
        Process full match video — detect + track all players.

        Args:
            video_path: Path to match MP4
            output_dir: Where to save output files + crops
            pretrained_detections_path: Optional pre-computed detections JSON
            save_crops: Whether to save player crop images for re-ID

        Returns:
            List of Detection objects for all frames
        """
        os.makedirs(output_dir, exist_ok=True)
        crops_dir = os.path.join(output_dir, "crops")
        if save_crops:
            os.makedirs(crops_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames} (~{total_frames/fps/60:.1f} min)")

        # Load pre-computed detections if provided
        pretrained = None
        if pretrained_detections_path and os.path.exists(pretrained_detections_path):
            pretrained = self.load_pretrained_detections(pretrained_detections_path)
            print("Using pre-computed detections")

        frame_id = 0
        track_crops: Dict[int, List[np.ndarray]] = defaultdict(list)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % 250 == 0:
                print(f"  Frame {frame_id}/{total_frames} ({100*frame_id/total_frames:.1f}%)")

            # Get detections (from YOLO or pre-computed)
            if pretrained and str(frame_id) in pretrained:
                # Convert pre-computed format to supervision Detections
                pd = pretrained[str(frame_id)]
                if len(pd) == 0:
                    frame_id += 1
                    continue
                xyxy = np.array([[d['x'], d['y'], d['x']+d['w'], d['y']+d['h']] for d in pd])
                conf = np.array([d['confidence'] for d in pd])
                detections = sv.Detections(xyxy=xyxy, confidence=conf,
                                           class_id=np.zeros(len(pd), dtype=int))
            else:
                detections = self.detect_frame(frame, frame_id)

            # Apply ByteTrack
            tracked = self.track_frame(detections)

            # Store results + save crops
            for i, (bbox, track_id, conf) in enumerate(
                    zip(tracked.xyxy, tracked.tracker_id, tracked.confidence)):
                x1, y1, x2, y2 = bbox
                cx, cy = (x1+x2)/2, (y1+y2)/2
                w, h = x2-x1, y2-y1

                det = Detection(
                    frame_id=frame_id,
                    track_id=int(track_id),
                    x=float(x1), y=float(y1),
                    w=float(w), h=float(h),
                    confidence=float(conf),
                    cx=float(cx), cy=float(cy)
                )
                self.all_detections.append(det)

                # Save crop every 25 frames (1 per second) for re-ID
                if save_crops and frame_id % 25 == 0:
                    crop = self.crop_player(frame, bbox)
                    if crop is not None:
                        track_crops[int(track_id)].append(crop)

            frame_id += 1

        cap.release()

        # Save crops to disk
        if save_crops:
            for track_id, crops in track_crops.items():
                track_dir = os.path.join(crops_dir, f"track_{track_id:04d}")
                os.makedirs(track_dir, exist_ok=True)
                for idx, crop in enumerate(crops):
                    cv2.imwrite(os.path.join(track_dir, f"crop_{idx:04d}.jpg"), crop)

        print(f"Done. {len(self.all_detections)} detections across {frame_id} frames.")
        return self.all_detections

    def save_annotations(self, output_path: str):
        """Save per-frame tracking annotations to JSON"""
        data = [asdict(d) for d in self.all_detections]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Annotations saved: {output_path}")

    def get_track_summary(self) -> Dict:
        """Get per-track summary (start frame, end frame, total frames)"""
        tracks = defaultdict(list)
        for d in self.all_detections:
            tracks[d.track_id].append(d)

        summary = {}
        for tid, dets in tracks.items():
            frames = [d.frame_id for d in dets]
            summary[tid] = {
                "track_id": tid,
                "start_frame": min(frames),
                "end_frame": max(frames),
                "total_frames": len(frames),
                "avg_confidence": np.mean([d.confidence for d in dets])
            }
        return summary
