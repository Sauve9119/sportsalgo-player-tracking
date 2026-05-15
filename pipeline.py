"""
SportsAlgo Hackathon — Main Pipeline
Runs the complete two-layer system:
  Layer 1: Detection + Within-Match Tracking
  Layer 2: Cross-Match Identity Mapping
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

from detector_tracker import PlayerDetectorTracker
from reid_identity import (
    AppearanceExtractor, GaitExtractor,
    PhysicalEstimator, PlayerIdentityDB
)


def process_match(
    video_path: str,
    match_id: str,
    output_dir: str,
    player_db: PlayerIdentityDB,
    pretrained_detections: str = None,
    frame_height: int = 1080
) -> Dict:
    """
    Process one match video end-to-end.
    Returns mapping: {track_id → player_id}
    """
    print(f"\n{'='*50}")
    print(f"Processing Match: {match_id}")
    print(f"{'='*50}")

    match_out = os.path.join(output_dir, match_id)
    os.makedirs(match_out, exist_ok=True)

    # ── Layer 1: Detection + Tracking ──────────────────
    print("\n[Layer 1] Detection + Tracking...")
    tracker = PlayerDetectorTracker(model_path="yolov8n.pt")
    detections = tracker.process_video(
        video_path=video_path,
        output_dir=match_out,
        pretrained_detections_path=pretrained_detections,
        save_crops=True
    )
    tracker.save_annotations(os.path.join(match_out, "tracking_annotations.json"))

    track_summary = tracker.get_track_summary()
    print(f"  Unique tracks found: {len(track_summary)}")

    # ── Layer 2: Cross-Match Re-ID ──────────────────────
    print("\n[Layer 2] Cross-Match Identity Mapping...")

    app_extractor = AppearanceExtractor(model_name="osnet_x1_0")
    gait_extractor = GaitExtractor()
    crops_dir = os.path.join(match_out, "crops")

    # Group detections by track
    from collections import defaultdict
    track_dets = defaultdict(list)
    for d in detections:
        track_dets[d.track_id].append({
            'frame_id': d.frame_id,
            'x': d.x, 'y': d.y,
            'w': d.w, 'h': d.h,
            'cx': d.cx, 'cy': d.cy
        })

    track_to_player: Dict[int, str] = {}

    for track_id, dets in track_dets.items():
        # Skip tracks that are too short (likely false positives)
        if len(dets) < 10:
            continue

        # Extract appearance embedding from crops gallery
        appearance_emb = app_extractor.extract_gallery(crops_dir, track_id)
        if appearance_emb is None:
            appearance_emb = np.zeros(512, dtype=np.float32)

        # Extract gait signature from crop sequence
        import cv2
        track_crop_dir = os.path.join(crops_dir, f"track_{track_id:04d}")
        crop_sequence = []
        if os.path.exists(track_crop_dir):
            for fname in sorted(os.listdir(track_crop_dir))[:30]:
                crop = cv2.imread(os.path.join(track_crop_dir, fname))
                if crop is not None:
                    crop_sequence.append(crop)

        gait_emb = gait_extractor.build_gait_signature(crop_sequence)
        if gait_emb is None:
            gait_emb = np.zeros(198, dtype=np.float32)

        # Estimate physical parameters
        height_ratio, build_ratio = PhysicalEstimator.estimate(dets, frame_height)

        # Search database for existing match
        matched_player_id = player_db.find_match(
            appearance_emb=appearance_emb,
            gait_emb=gait_emb,
            height_ratio=height_ratio,
            build_ratio=build_ratio,
            threshold=0.45
        )

        if matched_player_id:
            # Found existing player → update their embedding
            print(f"  Track {track_id} → Matched: {matched_player_id}")
            player_db.update_player(matched_player_id, appearance_emb,
                                     gait_emb, match_id, track_id)
            track_to_player[track_id] = matched_player_id
        else:
            # New player → enroll them
            new_player_id = player_db.enroll_player(
                appearance_emb=appearance_emb,
                gait_emb=gait_emb,
                height_ratio=height_ratio,
                build_ratio=build_ratio,
                match_id=match_id,
                track_id=track_id
            )
            print(f"  Track {track_id} → New player: {new_player_id}")
            track_to_player[track_id] = new_player_id

    # Save match-level mapping
    mapping_path = os.path.join(match_out, "track_to_player_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump({str(k): v for k, v in track_to_player.items()}, f, indent=2)

    # Create per-frame output with Player IDs
    annotated = []
    for d in detections:
        player_id = track_to_player.get(d.track_id, "UNKNOWN")
        annotated.append({
            "frame_id": d.frame_id,
            "track_id": d.track_id,
            "player_id": player_id,
            "bbox": {"x": d.x, "y": d.y, "w": d.w, "h": d.h},
            "center": {"cx": d.cx, "cy": d.cy},
            "confidence": d.confidence
        })

    annotated_path = os.path.join(match_out, "per_frame_annotations_with_player_ids.json")
    with open(annotated_path, 'w') as f:
        json.dump(annotated, f, indent=2)

    print(f"\n  Match output saved to: {match_out}")
    print(f"  Players in this match: {len(set(track_to_player.values()))}")

    return track_to_player


def run_full_pipeline(
    match1_video: str,
    match2_video: str,
    output_dir: str,
    db_path: str = "player_db.pkl",
    match1_detections: str = None,
    match2_detections: str = None
):
    """
    Run complete pipeline on both match videos.
    Match 1 builds the identity database.
    Match 2 tests cross-match recognition.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize player DB (loads existing if available)
    player_db = PlayerIdentityDB(db_path=db_path)

    # Process Match 1 (builds the DB)
    print("\n>>> MATCH 1: Building identity database...")
    m1_mapping = process_match(
        video_path=match1_video,
        match_id="match_1",
        output_dir=output_dir,
        player_db=player_db,
        pretrained_detections=match1_detections
    )

    # Save DB after Match 1
    player_db.save()

    # Process Match 2 (tests cross-match recognition)
    print("\n>>> MATCH 2: Testing cross-match recognition...")
    m2_mapping = process_match(
        video_path=match2_video,
        match_id="match_2",
        output_dir=output_dir,
        player_db=player_db,
        pretrained_detections=match2_detections
    )

    # Save updated DB
    player_db.save()

    # Export final outputs
    print("\n>>> Generating final outputs...")

    # 1. Player identity database (JSON)
    player_db.export_json(os.path.join(output_dir, "player_identity_database.json"))

    # 2. Cross-match mapping
    cross_match = player_db.get_cross_match_mapping()
    cross_match_path = os.path.join(output_dir, "cross_match_identity_mapping.json")
    with open(cross_match_path, 'w') as f:
        json.dump(cross_match, f, indent=2)

    # 3. Summary report
    summary = {
        "total_unique_players": len(player_db.embeddings),
        "players_seen_in_both_matches": len(cross_match),
        "match_1_tracks": len(m1_mapping),
        "match_2_tracks": len(m2_mapping),
        "cross_match_mapping": cross_match
    }
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print("PIPELINE COMPLETE")
    print(f"{'='*50}")
    print(f"Total unique players identified: {summary['total_unique_players']}")
    print(f"Players matched across both matches: {summary['players_seen_in_both_matches']}")
    print(f"All outputs saved to: {output_dir}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SportsAlgo Player Tracking & Re-ID Pipeline")
    parser.add_argument("--match1", required=True, help="Path to Match 1 video (MP4)")
    parser.add_argument("--match2", required=True, help="Path to Match 2 video (MP4)")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--db", default="./player_db.pkl", help="Player DB path")
    parser.add_argument("--det1", default=None, help="Pre-computed detections for Match 1 (JSON)")
    parser.add_argument("--det2", default=None, help="Pre-computed detections for Match 2 (JSON)")

    args = parser.parse_args()

    run_full_pipeline(
        match1_video=args.match1,
        match2_video=args.match2,
        output_dir=args.output,
        db_path=args.db,
        match1_detections=args.det1,
        match2_detections=args.det2
    )
