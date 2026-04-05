"""
Edge AI Object Detection System
Visualization Utilities
=======================
Drawing bounding boxes, FPS overlay, and saving annotated frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# Color palette (BGR) for up to 20 classes
CLASS_COLORS = [
    (56, 56, 255),    (151, 157, 255),  (31, 112, 255),   (29, 178, 255),
    (49, 210, 207),   (10, 249, 72),    (23, 204, 146),   (134, 219, 61),
    (182, 210, 57),   (218, 194, 24),   (254, 172, 0),    (253, 138, 0),
    (255, 95, 0),     (255, 37, 34),    (241, 0, 73),     (224, 0, 130),
    (188, 24, 196),   (130, 37, 233),   (75, 55, 240),    (48, 80, 245),
]


def draw_detections(
    frame: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    confidences: List[float],
    class_ids: List[int],
    class_names: List[str],
    fps: Optional[float] = None,
    latency_ms: Optional[float] = None,
) -> np.ndarray:
    """
    Draw bounding boxes with labels on a frame.
    
    Args:
        frame: BGR image (H, W, 3)
        boxes: List of (x1, y1, x2, y2) bounding boxes
        confidences: List of confidence scores
        class_ids: List of class indices
        class_names: List of class name strings
        fps: Optional FPS to overlay
        latency_ms: Optional latency to overlay
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        
        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        label = f"{name} {conf:.0%}"
        
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Label background
        cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # FPS / Latency overlay
    if fps is not None or latency_ms is not None:
        overlay_parts = []
        if fps is not None:
            overlay_parts.append(f"FPS: {fps:.1f}")
        if latency_ms is not None:
            overlay_parts.append(f"Latency: {latency_ms:.1f}ms")
        overlay_parts.append(f"Objects: {len(boxes)}")
        
        overlay_text = " | ".join(overlay_parts)
        
        cv2.rectangle(annotated, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(annotated, overlay_text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1)
    
    return annotated


def save_annotated_frame(
    frame: np.ndarray,
    output_path: str,
    frame_id: int = 0,
) -> str:
    """Save an annotated frame to disk."""
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, frame)
    return output_path


def create_comparison_grid(
    frames: List[np.ndarray],
    labels: List[str],
    cols: int = 2,
) -> np.ndarray:
    """Create a grid of frames for side-by-side comparison."""
    n = len(frames)
    rows = (n + cols - 1) // cols
    
    # Resize all frames to same size
    target_h, target_w = frames[0].shape[:2]
    resized = []
    for f in frames:
        r = cv2.resize(f, (target_w, target_h))
        resized.append(r)
    
    # Add labels
    for i, (f, label) in enumerate(zip(resized, labels)):
        cv2.putText(f, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Pad to fill grid
    while len(resized) < rows * cols:
        resized.append(np.zeros_like(resized[0]))
    
    # Build grid
    grid_rows = []
    for r in range(rows):
        row_frames = resized[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_frames))
    
    return np.vstack(grid_rows)
