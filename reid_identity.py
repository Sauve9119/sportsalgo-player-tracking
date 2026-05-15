"""
SportsAlgo Hackathon — Layer 2: Cross-Match Player Re-Identification
Multi-modal approach: OSNet appearance + gait pose signatures + physical params
"""

import cv2
import numpy as np
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

# pip install torchreid faiss-cpu mediapipe insightface
import torch
import faiss


@dataclass
class PlayerEmbedding:
    player_id: str          # Persistent cross-match ID (e.g., "P001")
    track_id: int           # Within-match track ID
    match_id: str           # Which match this came from
    appearance_emb: list    # OSNet embedding vector
    gait_emb: list          # Pose/gait signature
    height_ratio: float     # Estimated relative height (bbox_h / frame_h)
    build_ratio: float      # Estimated build (bbox_w / bbox_h)


class AppearanceExtractor:
    """
    Extracts appearance embeddings using OSNet (person re-ID model).
    Works well even with overhead camera — captures texture, color patterns.
    """

    def __init__(self, model_name: str = "osnet_x1_0"):
        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                pretrained=True
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.use_torchreid = True
            print(f"OSNet loaded: {model_name}")
        except ImportError:
            print("torchreid not available — using OpenCV HOG as fallback")
            self.use_torchreid = False

        # Image preprocessing
        self.img_size = (128, 256)  # OSNet standard input

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Resize + normalize crop for OSNet"""
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
        return img

    def _hog_fallback(self, img: np.ndarray) -> np.ndarray:
        """HOG descriptor as fallback when OSNet not available"""
        img = cv2.resize(img, self.img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        return hog.compute(gray).flatten()

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract 512-dim appearance embedding from a player crop"""
        if not self.use_torchreid:
            return self._hog_fallback(crop)

        with torch.no_grad():
            x = self._preprocess(crop)
            if torch.cuda.is_available():
                x = x.cuda()
            emb = self.model(x)
            emb = emb.cpu().numpy().flatten()
            # L2 normalize
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def extract_gallery(self, crops_dir: str, track_id: int) -> Optional[np.ndarray]:
        """
        Extract and average embeddings from multiple crops of same track.
        Averaging improves robustness to pose variation.
        """
        track_dir = os.path.join(crops_dir, f"track_{track_id:04d}")
        if not os.path.exists(track_dir):
            return None

        embeddings = []
        for fname in sorted(os.listdir(track_dir))[:20]:  # Max 20 crops per track
            crop = cv2.imread(os.path.join(track_dir, fname))
            if crop is not None:
                emb = self.extract(crop)
                embeddings.append(emb)

        if not embeddings:
            return None

        # Gallery embedding = mean of all crops (more robust)
        gallery_emb = np.mean(embeddings, axis=0)
        gallery_emb = gallery_emb / (np.linalg.norm(gallery_emb) + 1e-8)
        return gallery_emb


class GaitExtractor:
    """
    Extracts gait/pose signatures using MediaPipe.
    Appearance-independent — works even when jersey colors match.
    Captures how a player moves: stride length, posture, running style.
    """

    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5
            )
            self.available = True
            print("MediaPipe Pose loaded")
        except ImportError:
            print("mediapipe not available — gait features disabled")
            self.available = False

    def extract_keypoints(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract 33 pose keypoints from player crop"""
        if not self.available:
            return None

        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if not results.pose_landmarks:
            return None

        # Extract x,y,visibility for each landmark → 99-dim vector
        kpts = []
        for lm in results.pose_landmarks.landmark:
            kpts.extend([lm.x, lm.y, lm.visibility])
        return np.array(kpts, dtype=np.float32)

    def build_gait_signature(self, crops: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Build gait signature from sequence of crops.
        Uses mean + std of keypoints across frames → captures movement style.
        """
        if not self.available or not crops:
            return np.zeros(198, dtype=np.float32)  # Zero vector if unavailable

        sequences = []
        for crop in crops[:30]:  # Use up to 30 frames
            kpts = self.extract_keypoints(crop)
            if kpts is not None:
                sequences.append(kpts)

        if len(sequences) < 3:
            return np.zeros(198, dtype=np.float32)

        seq_arr = np.array(sequences)
        # Gait signature = [mean_pose, std_pose] → captures both posture + variation
        mean_pose = np.mean(seq_arr, axis=0)
        std_pose = np.std(seq_arr, axis=0)
        signature = np.concatenate([mean_pose, std_pose])
        signature = signature / (np.linalg.norm(signature) + 1e-8)
        return signature


class PhysicalEstimator:
    """
    Estimates physical parameters from bounding boxes.
    Used as a coarse filter before fine-grained embedding matching.
    """

    @staticmethod
    def estimate(detections_for_track: list, frame_height: int) -> Tuple[float, float]:
        """
        Estimate height ratio and build ratio from track detections.
        Returns: (height_ratio, build_ratio)
        """
        heights = [d['h'] for d in detections_for_track]
        widths = [d['w'] for d in detections_for_track]

        median_h = np.median(heights)
        median_w = np.median(widths)

        height_ratio = median_h / frame_height  # Relative height
        build_ratio = median_w / median_h if median_h > 0 else 0.4  # Width/height ratio

        return float(height_ratio), float(build_ratio)


class PlayerIdentityDB:
    """
    FAISS-based vector database for player identity storage and matching.
    Stores multi-modal embeddings and supports cross-match identity lookup.
    """

    APPEARANCE_DIM = 512
    GAIT_DIM = 198
    TOTAL_DIM = APPEARANCE_DIM + GAIT_DIM  # 710-dim fused embedding

    def __init__(self, db_path: str = "player_db.pkl"):
        self.db_path = db_path
        self.embeddings: Dict[str, np.ndarray] = {}   # player_id → fused embedding
        self.metadata: Dict[str, dict] = {}            # player_id → metadata
        self.player_counter = 0

        # FAISS index for fast nearest-neighbour search
        self.index = faiss.IndexFlatL2(self.TOTAL_DIM)
        self.id_map: List[str] = []  # Maps FAISS index position → player_id

        # Load existing DB if it exists
        if os.path.exists(db_path):
            self.load()

    def _fuse_embeddings(self, appearance: np.ndarray, gait: np.ndarray,
                          w_app: float = 0.65, w_gait: float = 0.35) -> np.ndarray:
        """
        Fuse appearance + gait embeddings with learned weights.
        Appearance gets higher weight (more discriminative for same players).
        Gait weight increases when appearance is unreliable (same jersey color).
        """
        # Pad/truncate to exact dimensions
        app_vec = np.zeros(self.APPEARANCE_DIM, dtype=np.float32)
        gait_vec = np.zeros(self.GAIT_DIM, dtype=np.float32)

        app_dim = min(len(appearance), self.APPEARANCE_DIM)
        gait_dim = min(len(gait), self.GAIT_DIM)
        app_vec[:app_dim] = appearance[:app_dim] * w_app
        gait_vec[:gait_dim] = gait[:gait_dim] * w_gait

        fused = np.concatenate([app_vec, gait_vec])
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        return fused.astype(np.float32)

    def enroll_player(self, appearance_emb: np.ndarray, gait_emb: np.ndarray,
                       height_ratio: float, build_ratio: float,
                       match_id: str, track_id: int) -> str:
        """
        Add a new player to the database.
        Returns the assigned persistent player_id.
        """
        self.player_counter += 1
        player_id = f"P{self.player_counter:04d}"

        fused = self._fuse_embeddings(appearance_emb, gait_emb)
        self.embeddings[player_id] = fused
        self.metadata[player_id] = {
            "player_id": player_id,
            "height_ratio": height_ratio,
            "build_ratio": build_ratio,
            "seen_in_matches": [match_id],
            "track_ids": {match_id: track_id},
            "appearance_emb": appearance_emb.tolist(),
            "gait_emb": gait_emb.tolist()
        }

        # Add to FAISS index
        self.index.add(fused.reshape(1, -1))
        self.id_map.append(player_id)

        return player_id

    def find_match(self, appearance_emb: np.ndarray, gait_emb: np.ndarray,
                   height_ratio: float, build_ratio: float,
                   top_k: int = 5, threshold: float = 0.45) -> Optional[str]:
        """
        Search for a matching player in the database.

        Strategy:
        1. Physical filter: reject candidates with very different height/build
        2. FAISS nearest-neighbour on fused embedding
        3. Accept match if distance < threshold

        Returns player_id if match found, None if new player.
        """
        if self.index.ntotal == 0:
            return None

        fused = self._fuse_embeddings(appearance_emb, gait_emb)

        # FAISS search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(fused.reshape(1, -1), k)

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue

            candidate_id = self.id_map[idx]
            candidate_meta = self.metadata[candidate_id]

            # Physical plausibility filter
            h_diff = abs(candidate_meta['height_ratio'] - height_ratio)
            b_diff = abs(candidate_meta['build_ratio'] - build_ratio)

            # L2 distance in FAISS — lower = more similar
            # Typical range: 0 (identical) to 2.0 (completely different)
            if dist < threshold and h_diff < 0.15 and b_diff < 0.2:
                return candidate_id

        return None  # New player

    def update_player(self, player_id: str, appearance_emb: np.ndarray,
                       gait_emb: np.ndarray, match_id: str, track_id: int):
        """
        Update existing player's embedding (online gallery update).
        Exponential moving average keeps embedding fresh across matches.
        """
        if player_id not in self.embeddings:
            return

        new_fused = self._fuse_embeddings(appearance_emb, gait_emb)
        old_fused = self.embeddings[player_id]

        # EMA update: 80% old + 20% new
        updated = 0.8 * old_fused + 0.2 * new_fused
        updated = updated / (np.linalg.norm(updated) + 1e-8)
        self.embeddings[player_id] = updated.astype(np.float32)

        # Update metadata
        if match_id not in self.metadata[player_id]['seen_in_matches']:
            self.metadata[player_id]['seen_in_matches'].append(match_id)
        self.metadata[player_id]['track_ids'][match_id] = track_id

        # Rebuild FAISS index (necessary after update)
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild FAISS index from stored embeddings"""
        self.index = faiss.IndexFlatL2(self.TOTAL_DIM)
        self.id_map = []
        for pid, emb in self.embeddings.items():
            self.index.add(emb.reshape(1, -1))
            self.id_map.append(pid)

    def save(self):
        """Save database to disk"""
        data = {
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
            "metadata": self.metadata,
            "player_counter": self.player_counter,
            "id_map": self.id_map
        }
        with open(self.db_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Player DB saved: {self.db_path} ({len(self.embeddings)} players)")

    def load(self):
        """Load database from disk"""
        with open(self.db_path, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = {k: np.array(v) for k, v in data['embeddings'].items()}
        self.metadata = data['metadata']
        self.player_counter = data['player_counter']
        self.id_map = data['id_map']
        self._rebuild_index()
        print(f"Player DB loaded: {len(self.embeddings)} players")

    def export_json(self, output_path: str):
        """Export DB as JSON for submission"""
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Player DB exported: {output_path}")

    def get_cross_match_mapping(self) -> Dict:
        """Generate cross-match identity mapping for submission"""
        mapping = {}
        for pid, meta in self.metadata.items():
            if len(meta['seen_in_matches']) > 1:
                mapping[pid] = {
                    "player_id": pid,
                    "matched_across": meta['seen_in_matches'],
                    "track_ids_per_match": meta['track_ids'],
                    "height_ratio": meta['height_ratio'],
                    "build_ratio": meta['build_ratio']
                }
        return mapping
