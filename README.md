# SportsAlgo × ClayGrounds — Player Tracking & Identity Mapping

Hackathon submission for the Player Tracking & Automatic Identity Mapping problem statement.

## What This Does

A two-layer computer vision pipeline that:

1. **Layer 1 — Within-Match Tracking**: Detects all players in every frame using YOLOv8 and assigns consistent Track IDs throughout the match using ByteTrack (handles occlusions, re-entries, player clusters).

2. **Layer 2 — Cross-Match Identity Mapping**: Automatically recognizes the same physical player across different match videos using multi-modal embeddings (appearance + gait + physical parameters), without any manual player registration.

## Approach

```
Video → YOLOv8 Detection → ByteTrack → Player Crops
                                            ↓
                               OSNet Appearance Embedding (512-dim)
                             + MediaPipe Gait Signature (198-dim)
                             + Physical Params (height/build ratio)
                                            ↓
                               Weighted Fusion → FAISS Search
                                            ↓
                           Persistent Player ID (P0001, P0002, ...)
```

**Key design decisions:**
- Appearance weight = 0.65, Gait weight = 0.35 (appearance more discriminative overhead)
- Gallery averaging: mean of top-20 crops per track (robust to pose variation)
- EMA gallery update: 80% old + 20% new embedding (stable across matches)
- Physical filter: reject candidates with >15% height difference before embedding match
- FAISS L2 threshold = 0.45 (tuned on Match 1 validation set)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install torchreid (OSNet)
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -e .
cd ..
```

## Running the Pipeline

```bash
# Full pipeline — both matches
python src/pipeline.py \
  --match1 data/match1.mp4 \
  --match2 data/match2.mp4 \
  --output output/ \
  --db player_db.pkl

# With pre-computed detections (faster)
python src/pipeline.py \
  --match1 data/match1.mp4 \
  --match2 data/match2.mp4 \
  --det1 data/match1_detections.json \
  --det2 data/match2_detections.json \
  --output output/

# Evaluate results
python src/evaluate.py \
  --output output/ \
  --gt data/ground_truth/
```

## Output Files

```
output/
├── match_1/
│   ├── tracking_annotations.json          # Per-frame: bbox + Track ID + confidence
│   ├── per_frame_annotations_with_player_ids.json  # Same + Player IDs
│   ├── track_to_player_mapping.json       # Track ID → Player ID
│   └── crops/                             # Player crop images (for re-ID)
│       ├── track_0001/
│       └── track_0002/
├── match_2/
│   └── (same structure)
├── player_identity_database.json          # All player embeddings + metadata
├── cross_match_identity_mapping.json      # Players matched across matches
├── pipeline_summary.json                  # Summary stats
└── evaluation_results.json               # Precision/Recall/MOTA/Re-ID accuracy
```

## Annotation Format

**tracking_annotations.json** (per row):
```json
{
  "frame_id": 1250,
  "track_id": 7,
  "x": 423.5, "y": 310.2, "w": 45.1, "h": 78.3,
  "cx": 446.05, "cy": 349.35,
  "confidence": 0.87
}
```

**player_identity_database.json** (per player):
```json
{
  "P0003": {
    "player_id": "P0003",
    "height_ratio": 0.072,
    "build_ratio": 0.41,
    "seen_in_matches": ["match_1", "match_2"],
    "track_ids": {"match_1": 7, "match_2": 12},
    "appearance_emb": [...],
    "gait_emb": [...]
  }
}
```

## Dependencies

| Library | Purpose |
|---------|---------|
| YOLOv8 (ultralytics) | Player detection |
| ByteTrack (supervision) | Within-match tracking |
| OSNet (torchreid) | Appearance embeddings |
| MediaPipe | Pose / gait estimation |
| FAISS | Vector similarity search |
| OpenCV | Video I/O, image processing |
