"""
SportsAlgo Hackathon — Evaluation Script
Computes:
  - Detection: Precision, Recall, F1
  - Tracking: MOTA (Multi-Object Tracking Accuracy)
  - Re-ID: Cross-match identity mapping accuracy
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def compute_iou(box1: dict, box2: dict) -> float:
    """Compute IoU between two bounding boxes (x,y,w,h format)"""
    x1_min, y1_min = box1['x'], box1['y']
    x1_max, y1_max = x1_min + box1['w'], y1_min + box1['h']

    x2_min, y2_min = box2['x'], box2['y']
    x2_max, y2_max = x2_min + box2['w'], y2_min + box2['h']

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = box1['w'] * box1['h']
    box2_area = box2['w'] * box2['h']
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-8)


def evaluate_detection(pred_file: str, gt_file: str, iou_threshold: float = 0.5) -> Dict:
    """
    Evaluate detection precision and recall against ground truth.

    Args:
        pred_file: Path to predicted annotations JSON
        gt_file: Path to ground truth JSON (same format)
        iou_threshold: IoU threshold for true positive (default 0.5)
    """
    with open(pred_file) as f:
        predictions = json.load(f)
    with open(gt_file) as f:
        ground_truth = json.load(f)

    # Group by frame
    pred_by_frame = defaultdict(list)
    gt_by_frame = defaultdict(list)

    for d in predictions:
        pred_by_frame[d['frame_id']].append(d)
    for d in ground_truth:
        gt_by_frame[d['frame_id']].append(d)

    all_frames = set(list(pred_by_frame.keys()) + list(gt_by_frame.keys()))

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for frame_id in all_frames:
        preds = pred_by_frame[frame_id]
        gts = gt_by_frame[frame_id]

        matched_gts = set()
        for pred in preds:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gts:
                    continue
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_total += 1
                matched_gts.add(best_gt_idx)
            else:
                fp_total += 1

        fn_total += len(gts) - len(matched_gts)

    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall = tp_total / (tp_total + fn_total + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total
    }


def evaluate_tracking_mota(pred_file: str, gt_file: str, iou_threshold: float = 0.5) -> Dict:
    """
    Compute MOTA (Multi-Object Tracking Accuracy).
    MOTA = 1 - (FP + FN + ID_switches) / GT_total

    Higher MOTA = better tracking.
    MOTA > 0.6 is considered good for sports tracking.
    """
    with open(pred_file) as f:
        predictions = json.load(f)
    with open(gt_file) as f:
        ground_truth = json.load(f)

    pred_by_frame = defaultdict(list)
    gt_by_frame = defaultdict(list)
    for d in predictions:
        pred_by_frame[d['frame_id']].append(d)
    for d in ground_truth:
        gt_by_frame[d['frame_id']].append(d)

    all_frames = sorted(set(list(pred_by_frame.keys()) + list(gt_by_frame.keys())))

    fp_total = fn_total = id_switches = gt_total = 0
    # Track which pred track_id is assigned to each GT track_id
    gt_to_pred_assignment = {}

    for frame_id in all_frames:
        preds = pred_by_frame[frame_id]
        gts = gt_by_frame[frame_id]
        gt_total += len(gts)

        matched_preds = set()
        matched_gts = set()

        for gt_idx, gt in enumerate(gts):
            gt_id = gt.get('track_id', gt_idx)
            best_iou = 0
            best_pred_idx = -1

            for pred_idx, pred in enumerate(preds):
                if pred_idx in matched_preds:
                    continue
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            if best_iou >= iou_threshold and best_pred_idx >= 0:
                pred_id = preds[best_pred_idx].get('track_id')

                # Check for ID switch
                if gt_id in gt_to_pred_assignment:
                    if gt_to_pred_assignment[gt_id] != pred_id:
                        id_switches += 1
                gt_to_pred_assignment[gt_id] = pred_id

                matched_preds.add(best_pred_idx)
                matched_gts.add(gt_idx)
            else:
                fn_total += 1

        fp_total += len(preds) - len(matched_preds)

    mota = 1 - (fp_total + fn_total + id_switches) / (gt_total + 1e-8)

    return {
        "mota": round(mota, 4),
        "fp": fp_total,
        "fn": fn_total,
        "id_switches": id_switches,
        "gt_total": gt_total
    }


def evaluate_reid_accuracy(
    match1_gt_file: str,       # Ground truth: {gt_player_id → track_id_in_match1}
    match2_gt_file: str,       # Ground truth: {gt_player_id → track_id_in_match2}
    predicted_mapping_file: str  # Our output: cross_match_identity_mapping.json
) -> Dict:
    """
    Evaluate cross-match identity mapping accuracy.

    Metric: What fraction of players seen in both matches
    were correctly assigned the same Player ID?
    """
    with open(match1_gt_file) as f:
        m1_gt = json.load(f)  # {gt_player_id: track_id}
    with open(match2_gt_file) as f:
        m2_gt = json.load(f)  # {gt_player_id: track_id}
    with open(predicted_mapping_file) as f:
        pred_mapping = json.load(f)

    # Find players who appear in both matches (ground truth)
    players_in_both_gt = set(m1_gt.keys()) & set(m2_gt.keys())
    total_to_match = len(players_in_both_gt)

    if total_to_match == 0:
        return {"re_id_accuracy": 0, "message": "No overlapping players in GT"}

    # For each GT player in both matches, check if our system assigned same Player ID
    correct = 0
    details = []

    for gt_pid in players_in_both_gt:
        m1_track = m1_gt[gt_pid]
        m2_track = m2_gt[gt_pid]

        # Find our Player ID for this track in Match 1
        our_pid_m1 = None
        our_pid_m2 = None

        for our_pid, info in pred_mapping.items():
            if 'match_1' in info.get('matched_across', []):
                if info.get('track_ids', {}).get('match_1') == m1_track:
                    our_pid_m1 = our_pid
            if 'match_2' in info.get('matched_across', []):
                if info.get('track_ids', {}).get('match_2') == m2_track:
                    our_pid_m2 = our_pid

        is_correct = (our_pid_m1 is not None and
                      our_pid_m1 == our_pid_m2)

        if is_correct:
            correct += 1

        details.append({
            "gt_player_id": gt_pid,
            "m1_track": m1_track,
            "m2_track": m2_track,
            "our_id_m1": our_pid_m1,
            "our_id_m2": our_pid_m2,
            "correct": is_correct
        })

    accuracy = correct / total_to_match

    return {
        "re_id_accuracy": round(accuracy, 4),
        "correct_matches": correct,
        "total_to_match": total_to_match,
        "details": details
    }


def run_full_evaluation(output_dir: str, gt_dir: str):
    """Run all evaluations and print summary report"""
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)

    results = {}

    # Detection evaluation (Match 1)
    try:
        det_results = evaluate_detection(
            pred_file=f"{output_dir}/match_1/tracking_annotations.json",
            gt_file=f"{gt_dir}/match1_gt_annotations.json"
        )
        results['detection'] = det_results
        print(f"\n[Detection] P={det_results['precision']:.3f}  "
              f"R={det_results['recall']:.3f}  F1={det_results['f1']:.3f}")
    except FileNotFoundError:
        print("[Detection] GT file not found — skipping")

    # Tracking MOTA
    try:
        mota_results = evaluate_tracking_mota(
            pred_file=f"{output_dir}/match_1/tracking_annotations.json",
            gt_file=f"{gt_dir}/match1_gt_annotations.json"
        )
        results['tracking'] = mota_results
        print(f"[Tracking]  MOTA={mota_results['mota']:.3f}  "
              f"ID_switches={mota_results['id_switches']}")
    except FileNotFoundError:
        print("[Tracking] GT file not found — skipping")

    # Re-ID accuracy
    try:
        reid_results = evaluate_reid_accuracy(
            match1_gt_file=f"{gt_dir}/match1_player_map.json",
            match2_gt_file=f"{gt_dir}/match2_player_map.json",
            predicted_mapping_file=f"{output_dir}/cross_match_identity_mapping.json"
        )
        results['reid'] = reid_results
        print(f"[Re-ID]     Accuracy={reid_results['re_id_accuracy']:.3f}  "
              f"({reid_results['correct_matches']}/{reid_results['total_to_match']})")
    except FileNotFoundError:
        print("[Re-ID] GT files not found — skipping")

    # Save results
    with open(f"{output_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved: {output_dir}/evaluation_results.json")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Pipeline output directory")
    parser.add_argument("--gt", required=True, help="Ground truth directory")
    args = parser.parse_args()
    run_full_evaluation(args.output, args.gt)
