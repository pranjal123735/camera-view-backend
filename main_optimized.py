"""
ULTRA LOW LATENCY Car Vision Backend
Optimized for instant response and minimal processing time
"""

from __future__ import annotations

import os
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
except Exception:
    np = None
    Image = None
    YOLO = None

# ULTRA-FAST CONFIGURATION - Optimized for speed
YOLO_MODEL_NAME = os.environ.get("CAR_VISION_YOLO_MODEL", "yolov8n.pt")  # Nano model for speed
YOLO_CONF = float(os.environ.get("CAR_VISION_YOLO_CONF", "0.4"))  # Higher confidence for fewer false positives
YOLO_IOU = float(os.environ.get("CAR_VISION_YOLO_IOU", "0.6"))   # Higher IoU for better separation
YOLO_MAX_DET = int(os.environ.get("CAR_VISION_YOLO_MAX_DET", "20"))  # Limit detections for speed
YOLO_IMGSZ = int(os.environ.get("CAR_VISION_YOLO_IMGSZ", "416"))  # Smaller image size for speed

# Simplified tracking for performance
VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
MOVING_SPEED_THRESHOLD_KMH = 2.0

# Pre-computed object heights for instant distance calculation
OBJECT_HEIGHTS = {
    "person": 1.7,
    "bicycle": 1.1,
    "motorcycle": 1.2,
    "car": 1.5,
    "truck": 3.0,
    "bus": 3.2,
}
DEFAULT_FOCAL = 800.0  # Simplified focal length
DISTANCE_MIN_M = 0.5
DISTANCE_MAX_M = 100.0

app = FastAPI(title="Car Vision Backend - Ultra Fast", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for ultra-fast serialization
class DetectionOut(BaseModel):
    track_id: int
    label: str
    confidence: float
    bbox_xyxy: List[float]
    distance_m: float
    speed_kmh: float
    is_moving: bool
    ttc_s: float
    risk_percent: float

class AnalyzeResponse(BaseModel):
    frame_time_s: float
    detections: List[DetectionOut]
    processing_time_ms: float

@dataclass
class TrackState:
    cx: float
    cy: float
    t_s: float
    speed_mps: float

# Global state - optimized for speed
MODEL = None
MODEL_LOAD_ERROR: Optional[str] = None
TRACKS: Dict[int, TrackState] = {}
NEXT_ID = 1

# Load model once at startup
if YOLO is not None:
    try:
        MODEL = YOLO(YOLO_MODEL_NAME)
        # Warm up the model with a dummy prediction for faster first inference
        if MODEL:
            dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
            MODEL.predict(dummy_img, verbose=False, conf=YOLO_CONF, imgsz=YOLO_IMGSZ)
    except Exception as e:
        MODEL = None
        MODEL_LOAD_ERROR = str(e)
else:
    MODEL_LOAD_ERROR = "ultralytics import failed"

def fast_distance_estimate(label: str, bbox_height: float, frame_height: float) -> float:
    """Ultra-fast distance estimation with pre-computed values"""
    obj_height = OBJECT_HEIGHTS.get(label, 1.5)
    if bbox_height <= 0 or frame_height <= 0:
        return 10.0
    
    # Simplified pinhole model
    distance = (obj_height * DEFAULT_FOCAL) / bbox_height
    return max(DISTANCE_MIN_M, min(distance, DISTANCE_MAX_M))

def fast_risk_calculation(distance_m: float, speed_kmh: float, is_moving: bool) -> Tuple[float, float]:
    """Ultra-fast risk and TTC calculation"""
    if not is_moving or speed_kmh <= 0:
        # Static object risk based on proximity only
        if distance_m < 2:
            return 999.0, 80.0
        elif distance_m < 5:
            return 999.0, 40.0
        else:
            return 999.0, 10.0
    
    # Moving object - calculate TTC
    speed_mps = speed_kmh / 3.6
    ttc = distance_m / max(speed_mps, 0.1)
    
    # Fast risk lookup table
    if ttc < 1.0:
        risk = 95.0
    elif ttc < 2.0:
        risk = 80.0
    elif ttc < 3.0:
        risk = 60.0
    elif ttc < 5.0:
        risk = 35.0
    else:
        risk = 15.0
    
    return ttc, risk

def fast_track_update(cx: float, cy: float, now_s: float) -> Tuple[int, float]:
    """Ultra-fast tracking with minimal computation"""
    global NEXT_ID
    
    # Simple nearest neighbor tracking with distance threshold
    best_id = None
    best_dist = 100.0  # pixels
    
    for tid, track in TRACKS.items():
        if now_s - track.t_s > 2.0:  # Expire old tracks
            continue
        
        dist = ((cx - track.cx) ** 2 + (cy - track.cy) ** 2) ** 0.5
        if dist < best_dist and dist < 80:  # 80 pixel threshold
            best_id = tid
            best_dist = dist
    
    if best_id is None:
        best_id = NEXT_ID
        NEXT_ID += 1
        speed_mps = 0.0
    else:
        # Calculate speed
        prev_track = TRACKS[best_id]
        dt = max(now_s - prev_track.t_s, 0.1)
        pixel_dist = ((cx - prev_track.cx) ** 2 + (cy - prev_track.cy) ** 2) ** 0.5
        speed_mps = (pixel_dist * 0.05) / dt  # Assume 0.05 m/pixel
    
    # Update track
    TRACKS[best_id] = TrackState(cx=cx, cy=cy, t_s=now_s, speed_mps=speed_mps)
    
    return best_id, speed_mps

def filter_detections_fast(detections: List[Tuple[str, float, List[float]]], 
                          frame_w: float, frame_h: float) -> List[Tuple[str, float, List[float]]]:
    """Ultra-fast detection filtering - only essential filters"""
    filtered = []
    
    for label, conf, bbox in detections:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Skip tiny detections (likely noise)
        if width < 10 or height < 10:
            continue
            
        # Skip very low confidence
        if conf < 0.25:
            continue
            
        # Skip objects that are too large (likely misclassifications)
        area_fraction = (width * height) / (frame_w * frame_h)
        if area_fraction > 0.4:  # More than 40% of frame
            continue
            
        filtered.append((label, conf, bbox))
    
    return filtered

@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "mode": "ultra_fast",
        "model": YOLO_MODEL_NAME if MODEL else None,
        "model_loaded": MODEL is not None,
        "model_load_error": MODEL_LOAD_ERROR,
        "optimizations": [
            "nano_model",
            "reduced_image_size", 
            "limited_detections",
            "simplified_tracking",
            "fast_risk_calculation",
            "minimal_filtering"
        ]
    }

@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image_fast(file: UploadFile = File(...)) -> AnalyzeResponse:
    """Ultra-fast image analysis optimized for minimal latency"""
    start_time = time.time()
    now_s = start_time
    
    if np is None or Image is None or MODEL is None:
        return AnalyzeResponse(
            frame_time_s=now_s,
            detections=[],
            processing_time_ms=0.0
        )
    
    try:
        # Fast image loading
        raw = await file.read()
        image = Image.open(BytesIO(raw)).convert("RGB")
        
        # Resize for speed if image is too large
        if image.width > 640 or image.height > 640:
            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        
        frame = np.array(image)
        fh, fw = float(frame.shape[0]), float(frame.shape[1])
        
        # Ultra-fast YOLO inference with minimal parameters
        results = MODEL.predict(
            frame,
            verbose=False,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            max_det=YOLO_MAX_DET,
            imgsz=YOLO_IMGSZ,
            augment=False,  # Disable augmentation for speed
            half=True,      # Use half precision for speed
            device='cpu'    # Ensure CPU usage for consistency
        )
        
        # Fast detection extraction
        raw_detections = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_idx = int(b.cls[0].item())
                label = MODEL.names.get(cls_idx, str(cls_idx))
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                raw_detections.append((label, conf, [x1, y1, x2, y2]))
        
        # Fast filtering
        filtered_detections = filter_detections_fast(raw_detections, fw, fh)
        
        # Convert to output format with minimal processing
        out_detections = []
        for label, conf, bbox in filtered_detections:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Fast tracking
            track_id, speed_mps = fast_track_update(cx, cy, now_s)
            
            # Fast distance estimation
            bbox_height = y2 - y1
            distance_m = fast_distance_estimate(label, bbox_height, fh)
            
            # Fast speed and movement detection
            speed_kmh = speed_mps * 3.6
            is_moving = speed_kmh >= MOVING_SPEED_THRESHOLD_KMH
            
            # Fast risk calculation
            ttc_s, risk_percent = fast_risk_calculation(distance_m, speed_kmh, is_moving)
            
            detection = DetectionOut(
                track_id=track_id,
                label=label,
                confidence=round(conf, 3),
                bbox_xyxy=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                distance_m=round(distance_m, 1),
                speed_kmh=round(speed_kmh, 1),
                is_moving=is_moving,
                ttc_s=round(ttc_s, 1) if ttc_s < 900 else 999.0,
                risk_percent=round(risk_percent, 1)
            )
            out_detections.append(detection)
        
        # Sort by confidence for priority rendering
        out_detections.sort(key=lambda d: d.confidence, reverse=True)
        
        # Limit to top detections for ultra-fast response
        out_detections = out_detections[:15]
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return AnalyzeResponse(
            frame_time_s=now_s,
            detections=out_detections,
            processing_time_ms=round(processing_time_ms, 1)
        )
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        return AnalyzeResponse(
            frame_time_s=now_s,
            detections=[],
            processing_time_ms=round(processing_time_ms, 1)
        )

@app.get("/performance-stats")
def get_performance_stats() -> dict:
    """Get performance statistics"""
    return {
        "active_tracks": len(TRACKS),
        "model_loaded": MODEL is not None,
        "model_name": YOLO_MODEL_NAME,
        "config": {
            "confidence_threshold": YOLO_CONF,
            "iou_threshold": YOLO_IOU,
            "max_detections": YOLO_MAX_DET,
            "image_size": YOLO_IMGSZ
        },
        "optimizations_enabled": [
            "nano_yolo_model",
            "reduced_image_size",
            "limited_max_detections", 
            "simplified_tracking",
            "fast_distance_estimation",
            "minimal_post_processing",
            "half_precision_inference",
            "disabled_augmentation"
        ]
    }

@app.post("/clear-tracks")
def clear_tracks() -> dict:
    """Clear all tracking data for fresh start"""
    global TRACKS, NEXT_ID
    TRACKS.clear()
    NEXT_ID = 1
    return {"ok": True, "message": "All tracks cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)