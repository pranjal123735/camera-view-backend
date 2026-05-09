from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

# Import enhanced services
try:
    from analytics_service import AnalyticsService
    from safety_service import SafetyService
    from learning_service import LearningService
    from motorcycle_360_vision import Motorcycle360Vision
    from surround_vision_renderer import SurroundVisionRenderer, DetectedObject
except ImportError:
    AnalyticsService = None
    SafetyService = None
    LearningService = None
    Motorcycle360Vision = None
    SurroundVisionRenderer = None
    DetectedObject = None

from rag_enhanced_detection import rag_enhancer
from ensemble_detection import ensemble_system
from knowledge_graph import knowledge_graph
from analytics_service import analytics_service, PerformanceMetrics, SafetyMetrics, AIEnhancementStats
from safety_service import safety_service, CollisionPrediction, SafetyAlert, EmergencyEvent
from learning_service import learning_service, UserProfile, CustomObject
from performance_controller import router as performance_router, should_skip_heavy_processing, should_use_rag, should_use_knowledge_graph, should_use_ensemble, should_use_learning, get_yolo_params
import asyncio


VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}
MOVING_SPEED_THRESHOLD_KMH = 1.8

# YOLO inference (tune for fewer false positives; raise conf if still noisy)
# Default small model; override with CAR_VISION_YOLO_MODEL if needed.
YOLO_MODEL_NAME = os.environ.get("CAR_VISION_YOLO_MODEL", "yolov8s.pt")
# Higher = fewer false boxes (better label accuracy). Lower via env only if you need more recall on tiny/distant objects.
YOLO_CONF = float(os.environ.get("CAR_VISION_YOLO_CONF", "0.3"))
YOLO_IOU = float(os.environ.get("CAR_VISION_YOLO_IOU", "0.5"))
YOLO_MAX_DET = int(os.environ.get("CAR_VISION_YOLO_MAX_DET", "60"))
YOLO_IMGSZ = int(os.environ.get("CAR_VISION_YOLO_IMGSZ", "640"))
# Slower (~2× infer) but can slightly improve robustness; use when you care more about detection than FPS.
YOLO_TTA = os.environ.get("CAR_VISION_YOLO_TTA", "").lower() in ("1", "true", "yes")

# Set CAR_VISION_DISABLE_PERSON_FP_FILTER=1 only if you want maximum recall (more wrong "person" boxes possible).
DISABLE_PERSON_FP_FILTER = os.environ.get("CAR_VISION_DISABLE_PERSON_FP_FILTER", "").lower() in (
    "1",
    "true",
    "yes",
)

# Monocular height: when bbox is tall vs frame, visible extent is often << nominal object height (e.g. face vs 1.7m person).
DISTANCE_CLOSE_FRAC = float(os.environ.get("CAR_VISION_DISTANCE_CLOSE_FRAC", "0.14"))
DISTANCE_MIN_M = float(os.environ.get("CAR_VISION_DISTANCE_MIN_M", "0.08"))
DISTANCE_MAX_M = float(os.environ.get("CAR_VISION_DISTANCE_MAX_M", "120.0"))

# Temporal smoothing: stable class over last N frames (reduces phone↔person flicker)
CLASS_HISTORY_LEN = int(os.environ.get("CAR_VISION_CLASS_HISTORY", "7"))
# Minimum frames with same winning class before switching label (hysteresis)
CLASS_SWITCH_AFTER = int(os.environ.get("CAR_VISION_CLASS_SWITCH_FRAMES", "3"))

# Person false-positive suppression (handheld phone confused as person)
PERSON_MIN_CONF_SMALL = float(os.environ.get("CAR_VISION_PERSON_MIN_CONF_SMALL", "0.35"))
PERSON_MAX_AREA_FRAC_WEAK = float(os.environ.get("CAR_VISION_PERSON_MAX_AREA_FRAC", "0.001"))
PERSON_TALL_NARROW_AR = float(os.environ.get("CAR_VISION_PERSON_TALL_AR", "4.5"))
PERSON_TALL_NARROW_MAX_CONF = float(os.environ.get("CAR_VISION_PERSON_TALL_MAX_CONF", "0.55"))

# IoU: if person overlaps cell phone and phone scores reasonably, drop person
PHONE_PERSON_IOU = float(os.environ.get("CAR_VISION_PHONE_PERSON_IOU", "0.25"))
PHONE_PERSON_CONF_RATIO = float(os.environ.get("CAR_VISION_PHONE_CONF_RATIO", "0.85"))

# Dash / instrument cluster: COCO "clock" often matches analog speedometers; "tv" can match digital clusters.
CAR_VISION_SUPPRESS_DASHBOARD_INTERIOR = os.environ.get(
    "CAR_VISION_SUPPRESS_DASHBOARD_INTERIOR", "1"
).lower() in ("1", "true", "yes")
CAR_VISION_DASHBOARD_MIN_CY_NORM = float(os.environ.get("CAR_VISION_DASHBOARD_MIN_CY_NORM", "0.4"))
CAR_VISION_DASHBOARD_MIN_BOTTOM_Y_NORM = float(
    os.environ.get("CAR_VISION_DASHBOARD_MIN_BOTTOM_Y_NORM", "0.56")
)

API_VERSION = "0.4.0"

app = FastAPI(title="Car Vision Backend", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include performance router
app.include_router(performance_router)


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
    track_age_s: float = 0.0


class FrameDiagnostics(BaseModel):
    """Cheap frame stats so the client can warn when ranging/detection is less reliable."""

    brightness_01: float
    texture_variance: float
    low_light: bool
    glare_risk: bool
    low_contrast: bool
    quality_hint: str


class TripSnapshot(BaseModel):
    """Phase 4: per-session counters (in-memory until POST /trip/reset)."""

    frames: int
    danger_frames: int
    caution_frames: int
    safe_frames: int
    near_miss_count: int
    trip_elapsed_s: float


class SurroundVisionRequest(BaseModel):
    """Request for surround vision scene generation"""
    road_type: str = "urban"
    speed: float = 0.0
    turn_direction: str = "straight"
    detected_objects: List[Dict[str, Any]] = []


class TripEventOut(BaseModel):
    ts_s: float
    severity: str
    label: str
    track_id: int
    distance_m: float
    risk_percent: float
    ttc_s: Optional[float] = None


class TripStatsOut(BaseModel):
    trip_started_s: float
    frames: int
    danger_frames: int
    caution_frames: int
    safe_frames: int
    near_miss_count: int
    trip_elapsed_s: float
    events: List[TripEventOut]


class AnalyzeResponse(BaseModel):
    frame_time_s: float
    detections: List[DetectionOut]
    trip: TripSnapshot
    api_version: str = API_VERSION
    frame_diagnostics: Optional[FrameDiagnostics] = None


class CalibrationConfig(BaseModel):
    focal_like: float
    meters_per_px: float
    default_object_height_m: float
    object_heights_m: Dict[str, float]


class LearningFeedback(BaseModel):
    detection_result: Dict[str, Any]
    ground_truth: Optional[Dict[str, Any]] = None
    feedback_type: str  # 'false_positive', 'false_negative', 'correct', 'improve'
    user_correction: Optional[str] = None

class UserBehaviorData(BaseModel):
    user_id: str
    session_data: Dict[str, Any]

class CustomObjectTraining(BaseModel):
    user_id: str
    object_name: str
    training_images: List[str]

class PersonalizedSettingsRequest(BaseModel):
    user_id: str
    context: Optional[Dict[str, Any]] = None

class EmergencyContactRequest(BaseModel):
    name: str
    phone: str
    email: Optional[str] = None
    relationship: str = "emergency"
    priority: int = 1

class SafetyZoneRequest(BaseModel):
    name: str
    center_lat: float
    center_lon: float
    radius_m: float
    zone_type: str = "general"
    speed_limit: Optional[float] = None


@dataclass
class TrackState:
    cx: float
    cy: float
    t_s: float
    speed_mps: float


MODEL = None
MODEL_LOAD_ERROR: Optional[str] = None
if YOLO is not None:
    try:
        MODEL = YOLO(YOLO_MODEL_NAME)
    except Exception as e:
        MODEL = None
        MODEL_LOAD_ERROR = str(e)
else:
    MODEL_LOAD_ERROR = "ultralytics import failed"

TRACKS: Dict[int, TrackState] = {}
TRACK_FIRST_SEEN_S: Dict[int, float] = {}
NEXT_ID = 1
# Per track: deque of (raw_model_label, confidence) from YOLO — output label is smoothed, never random
TRACK_CLASS_HIST: Dict[int, Deque[Tuple[str, float]]] = defaultdict(
    lambda: deque(maxlen=CLASS_HISTORY_LEN)
)
# Last emitted stable label per track (hysteresis)
TRACK_STABLE_LABEL: Dict[int, str] = {}
# Phase 4: trip / near-miss session (process-local; reset via POST /trip/reset)
NEAR_MISS_DEBOUNCE_S = float(os.environ.get("CAR_VISION_NEAR_MISS_DEBOUNCE_S", "4.0"))
TRIP_EVENTS_MAX = int(os.environ.get("CAR_VISION_TRIP_EVENTS_MAX", "80"))

# Initialize enhanced services
analytics_service = AnalyticsService() if AnalyticsService else None
safety_service = SafetyService() if SafetyService else None
learning_service = LearningService() if LearningService else None
motorcycle_vision = Motorcycle360Vision() if Motorcycle360Vision else None
surround_renderer = SurroundVisionRenderer() if SurroundVisionRenderer else None

TRIP_STARTED_S: float = time.time()
TRIP_FRAMES: int = 0
TRIP_DANGER_FRAMES: int = 0
TRIP_CAUTION_FRAMES: int = 0
TRIP_SAFE_FRAMES: int = 0
TRIP_NEAR_MISS_COUNT: int = 0
TRIP_EVENTS: Deque[dict] = deque(maxlen=TRIP_EVENTS_MAX)
_LAST_NEAR_MISS_LOG: Dict[str, float] = {}

CALIBRATION = CalibrationConfig(
    focal_like=float(os.environ.get("CAR_VISION_DEFAULT_FOCAL_LIKE", "900.0")),
    meters_per_px=0.05,
    default_object_height_m=1.5,
    object_heights_m={
        "person": 1.7,
        "bicycle": 1.1,
        "motorcycle": 1.2,
        "car": 1.5,
        "truck": 3.0,
        "bus": 3.2,
    },
)


def estimate_distance_m(
    label: str,
    bbox_xyxy: Tuple[float, float, float, float],
    frame_w: float,
    frame_h: float,
) -> float:
    """
    Pinhole-ish ranging from bbox vertical span. When the box is large vs frame height,
    the visible real-world extent is usually much smaller than nominal class height
    (e.g. close-up face vs full 1.7m person), so we shrink effective height to avoid
    overstating distance.
    """
    _x1, y1, _x2, y2 = bbox_xyxy
    h = max(y2 - y1, 1.0)
    fh = max(frame_h, 1.0)
    obj_h_m = CALIBRATION.object_heights_m.get(label, CALIBRATION.default_object_height_m)
    # 1.0 when object is small on screen; <1 when it fills many rows (close / partial subject).
    shrink = min(1.0, (DISTANCE_CLOSE_FRAC * fh) / h)
    effective_h_m = obj_h_m * shrink
    distance = (effective_h_m * CALIBRATION.focal_like) / h
    return float(max(DISTANCE_MIN_M, min(distance, DISTANCE_MAX_M)))


def ttc_and_risk(distance_m: float, speed_kmh: float) -> Tuple[float, float]:
    if speed_kmh <= 0.0:
        return 999.0, 0.0
    speed_mps = max(0.0, speed_kmh / 3.6)
    rel_speed = max(0.1, speed_mps)
    ttc = distance_m / rel_speed
    if ttc < 1.5:
        risk = 96.0
    elif ttc < 3.0:
        risk = 72.0
    elif ttc < 5.0:
        risk = 42.0
    else:
        risk = 15.0
    return float(ttc), float(risk)


def threat_score_value(det: DetectionOut) -> float:
    """Same weighting as the web client `threatScore()` for consistent ranking."""
    distance_m = float(det.distance_m)
    speed_kmh = float(det.speed_kmh)
    risk = float(det.risk_percent)
    moving = bool(det.is_moving)
    df = max(0.0, min(1.0, (45.0 - distance_m) / 45.0))
    sf = max(0.0, min(1.0, speed_kmh / 70.0)) if moving else 0.0
    rf = max(0.0, min(1.0, risk / 100.0))
    return df * 0.45 + sf * 0.25 + rf * 0.3


def band_for_detection(det: DetectionOut) -> str:
    risk = float(det.risk_percent)
    ttc = float(det.ttc_s)
    if risk >= 75.0 or ttc < 1.8:
        return "DANGER"
    if risk >= 40.0 or ttc < 3.5:
        return "CAUTION"
    return "SAFE"


def pick_top_detection(detections: List[DetectionOut]) -> Optional[DetectionOut]:
    if not detections:
        return None
    pool = [d for d in detections if d.label in VEHICLE_LABELS and d.is_moving]
    if not pool:
        pool = list(detections)
    return max(pool, key=threat_score_value)


def trip_snapshot(now_s: float) -> TripSnapshot:
    elapsed = max(0.0, now_s - TRIP_STARTED_S)
    return TripSnapshot(
        frames=TRIP_FRAMES,
        danger_frames=TRIP_DANGER_FRAMES,
        caution_frames=TRIP_CAUTION_FRAMES,
        safe_frames=TRIP_SAFE_FRAMES,
        near_miss_count=TRIP_NEAR_MISS_COUNT,
        trip_elapsed_s=round(elapsed, 1),
    )


def update_trip_stats(detections: List[DetectionOut], now_s: float) -> None:
    global TRIP_FRAMES, TRIP_DANGER_FRAMES, TRIP_CAUTION_FRAMES, TRIP_SAFE_FRAMES, TRIP_NEAR_MISS_COUNT

    TRIP_FRAMES += 1
    top = pick_top_detection(detections)
    if top is None:
        TRIP_SAFE_FRAMES += 1
        return
    band = band_for_detection(top)
    if band == "DANGER":
        TRIP_DANGER_FRAMES += 1
    elif band == "CAUTION":
        TRIP_CAUTION_FRAMES += 1
    else:
        TRIP_SAFE_FRAMES += 1

    if top.label in VEHICLE_LABELS and top.is_moving and band in ("CAUTION", "DANGER"):
        key = f"{top.track_id}:{band}"
        last_t = _LAST_NEAR_MISS_LOG.get(key, 0.0)
        if now_s - last_t >= NEAR_MISS_DEBOUNCE_S:
            _LAST_NEAR_MISS_LOG[key] = now_s
            TRIP_NEAR_MISS_COUNT += 1
            ttc_val = float(top.ttc_s)
            TRIP_EVENTS.appendleft(
                {
                    "ts_s": now_s,
                    "severity": band,
                    "label": top.label,
                    "track_id": int(top.track_id),
                    "distance_m": round(float(top.distance_m), 2),
                    "risk_percent": round(float(top.risk_percent), 1),
                    "ttc_s": None if ttc_val >= 900.0 else round(ttc_val, 2),
                }
            )


def reset_trip_state() -> None:
    global TRIP_STARTED_S, TRIP_FRAMES, TRIP_DANGER_FRAMES, TRIP_CAUTION_FRAMES, TRIP_SAFE_FRAMES
    global TRIP_NEAR_MISS_COUNT, _LAST_NEAR_MISS_LOG

    TRIP_STARTED_S = time.time()
    TRIP_FRAMES = 0
    TRIP_DANGER_FRAMES = 0
    TRIP_CAUTION_FRAMES = 0
    TRIP_SAFE_FRAMES = 0
    TRIP_NEAR_MISS_COUNT = 0
    TRIP_EVENTS.clear()
    _LAST_NEAR_MISS_LOG.clear()


def centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def compute_frame_diagnostics(frame: "np.ndarray") -> FrameDiagnostics:
    """Lightweight scene cues from the RGB frame (no extra model)."""
    gray = np.mean(frame.astype(np.float32), axis=2)
    step = max(1, int(min(gray.shape[0], gray.shape[1]) / 90))
    small = gray[::step, ::step]
    brightness_01 = float(np.clip(small.mean() / 255.0, 0.0, 1.0))
    texture_variance = float(np.var(small))
    bright_frac = float((small >= 235.0).mean())
    low_light = brightness_01 < 0.18  # triggers at dusk/dim roads, not just pitch black
    # Hot pixels + smeared highlights (tail lamps / street lights) often sit on a flattened texture.
    glare_risk = bright_frac > 0.075 or (bright_frac > 0.038 and texture_variance < 11.0)
    low_contrast = texture_variance < 12.0
    hints: List[str] = []
    if low_light:
        hints.append("low_light")
    if glare_risk:
        hints.append("glare")
    if low_contrast:
        hints.append("low_contrast")
    if len(hints) > 1:
        quality_hint = "mixed"
    elif hints:
        quality_hint = hints[0]
    else:
        quality_hint = "ok"
    return FrameDiagnostics(
        brightness_01=brightness_01,
        texture_variance=texture_variance,
        low_light=low_light,
        glare_risk=glare_risk,
        low_contrast=low_contrast,
        quality_hint=quality_hint,
    )


def enhance_night_frame(frame: "np.ndarray", diagnostics: "FrameDiagnostics") -> "np.ndarray":
    """
    Night / low-light frame enhancement so YOLO can see vehicle bodies behind headlights.

    Strategy:
      1. Always run when low_light OR low_contrast (dark road scenes).
      2. CLAHE per channel in LAB colour space — lifts dark regions without blowing out
         already-bright areas (headlights stay bright, vehicle body becomes visible).
      3. Highlight suppression — headlight halos are clipped so the surrounding pixels
         (bumper, bonnet, bike frame) get more relative contrast.
      4. Mild unsharp-mask sharpening — recovers edge detail lost to sensor noise.
      5. When glare is also present (oncoming headlights), we reduce the CLAHE clip
         limit so we don't amplify the bloom further.

    Returns the enhanced frame (same shape/dtype as input).
    """
    import cv2  # OpenCV is bundled with ultralytics

    if not (diagnostics.low_light or diagnostics.low_contrast):
        return frame  # Daytime / well-lit — skip entirely

    # --- 1. Convert to LAB so we only touch luminance ---
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # --- 2. CLAHE on L channel ---
    # Lower clip when glare is present to avoid amplifying headlight bloom.
    clip = 1.5 if diagnostics.glare_risk else 3.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)

    # --- 3. Highlight suppression ---
    # Pixels already very bright (headlights, street lamps) are pulled back toward
    # a softer ceiling so the vehicle body around them gains relative visibility.
    if diagnostics.glare_risk:
        # Blend: keep 60% of CLAHE result, pull 40% toward a soft ceiling of 220
        ceiling = np.full_like(l_eq, 220, dtype=np.uint8)
        l_eq = cv2.addWeighted(l_eq, 0.6, ceiling, 0.0, 0).clip(0, 255).astype(np.uint8)
        # Soft-clip anything above 230 → compress to 230-245 range
        bright_mask = l_eq > 230
        l_eq[bright_mask] = (230 + (l_eq[bright_mask].astype(np.int32) - 230) // 3).clip(0, 245).astype(np.uint8)

    # --- 4. Merge back and convert to RGB ---
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # --- 5. Unsharp mask — sharpens edges (bike frames, car outlines) ---
    # Only when very dark; skip when glare is the main issue (sharpening halos = worse).
    if diagnostics.low_light and not diagnostics.glare_risk:
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
        enhanced = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)

    return enhanced.astype(np.uint8)


def bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return float(inter / union) if union > 0 else 0.0


def match_track(
    cx: float,
    cy: float,
    now_s: float,
    frame_w: float,
    frame_h: float,
) -> Optional[int]:
    gate_px = max(48.0, min(frame_w, frame_h) * 0.07)
    best_id = None
    best_d2 = 1e18
    for tid, st in TRACKS.items():
        if now_s - st.t_s > 1.5:
            continue
        dx, dy = cx - st.cx, cy - st.cy
        d2 = dx * dx + dy * dy
        if d2 < gate_px * gate_px and d2 < best_d2:
            best_id = tid
            best_d2 = d2
    return best_id


def update_track(tid: int, cx: float, cy: float, now_s: float) -> float:
    prev = TRACKS.get(tid)
    if not prev:
        TRACKS[tid] = TrackState(cx=cx, cy=cy, t_s=now_s, speed_mps=0.0)
        return 0.0
    dt = max(now_s - prev.t_s, 1e-3)
    px_dist = ((cx - prev.cx) ** 2 + (cy - prev.cy) ** 2) ** 0.5
    speed_mps = (px_dist * CALIBRATION.meters_per_px) / dt
    TRACKS[tid] = TrackState(cx=cx, cy=cy, t_s=now_s, speed_mps=speed_mps)
    return speed_mps


def smooth_label_from_history(tid: int, raw_label: str, raw_conf: float) -> Tuple[str, float]:
    """Weighted vote + hysteresis on YOLO class names only (reduces single-frame mislabels)."""
    hist = TRACK_CLASS_HIST[tid]
    hist.append((raw_label, raw_conf))
    scores: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for lbl, c in hist:
        scores[lbl] += c
        counts[lbl] += 1
    winner = max(scores.keys(), key=lambda k: scores[k])
    stable = TRACK_STABLE_LABEL.get(tid)
    if stable is None:
        out_label = winner
        TRACK_STABLE_LABEL[tid] = winner
    elif winner == stable:
        out_label = stable
    else:
        recent = list(hist)[-CLASS_SWITCH_AFTER:]
        consecutive_new = len(recent) >= CLASS_SWITCH_AFTER and all(l == winner for l, _ in recent)
        margin = scores[winner] > scores.get(stable, 0.0) * 1.12
        if consecutive_new or margin:
            out_label = winner
            TRACK_STABLE_LABEL[tid] = winner
        else:
            out_label = stable
    avg_conf = scores[out_label] / max(counts[out_label], 1)
    return out_label, float(min(0.99, max(raw_conf, avg_conf)))


def keep_detection_for_ride_mount(
    label: str,
    bbox: Tuple[float, float, float, float],
    frame_h: float,
) -> bool:
    """Drop interior false positives when phone/cam is mounted over the cluster (portrait)."""
    if not CAR_VISION_SUPPRESS_DASHBOARD_INTERIOR or frame_h <= 1.0:
        return True
    ln = (label or "").strip().lower().replace("_", " ")
    if ln not in ("clock", "tv"):
        return True
    _x1, y1, _x2, y2 = bbox
    cy = (y1 + y2) / 2.0
    cy_norm = cy / frame_h
    bot_norm = y2 / frame_h
    if cy_norm >= CAR_VISION_DASHBOARD_MIN_CY_NORM or bot_norm >= CAR_VISION_DASHBOARD_MIN_BOTTOM_Y_NORM:
        return False
    return True


def person_false_positive_filter(
    label: str,
    conf: float,
    bbox: Tuple[float, float, float, float],
    frame_w: float,
    frame_h: float,
) -> bool:
    """Return True to keep detection, False to drop weak person-like boxes."""
    if label != "person":
        return True
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    area = w * h
    area_frac = area / max(frame_w * frame_h, 1.0)
    ar = h / w
    if area_frac < PERSON_MAX_AREA_FRAC_WEAK and conf < PERSON_MIN_CONF_SMALL:
        return False
    if ar >= PERSON_TALL_NARROW_AR and w < 0.035 * frame_w and conf < PERSON_TALL_NARROW_MAX_CONF:
        return False
    return True


def resolve_person_cell_phone_conflicts(
    rows: List[Tuple[str, float, Tuple[float, float, float, float]]],
) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    """
    If a 'person' box strongly overlaps a 'cell phone' box and phone conf is competitive, remove person
    (common misread: phone in hand).
    """
    people: List[int] = []
    phones: List[int] = []
    for i, (lab, conf, bb) in enumerate(rows):
        if lab == "person":
            people.append(i)
        elif lab in ("cell phone", "cell_phone"):
            phones.append(i)
    drop: set[int] = set()
    for pi in people:
        plab, pconf, pbb = rows[pi]
        for hi in phones:
            _, hconf, hbb = rows[hi]
            iou = bbox_iou_xyxy(pbb, hbb)
            if iou >= PHONE_PERSON_IOU and hconf >= pconf * PHONE_PERSON_CONF_RATIO:
                drop.add(pi)
                break
    return [rows[i] for i in range(len(rows)) if i not in drop]


def get_dynamic_yolo_params(diagnostics: Optional[FrameDiagnostics]) -> dict:
    """
    Dynamically adjust YOLO parameters based on scene conditions for better accuracy.
    """
    params = {
        'conf': YOLO_CONF,
        'iou': YOLO_IOU,
        'max_det': YOLO_MAX_DET,
        'imgsz': YOLO_IMGSZ,
        'augment': YOLO_TTA,
    }
    
    if diagnostics:
        # Adjust confidence threshold based on scene quality
        if diagnostics.low_light or diagnostics.low_contrast:
            # Lower confidence in poor conditions to catch more objects
            params['conf'] = max(0.15, YOLO_CONF * 0.85)
        elif diagnostics.glare_risk:
            # Higher confidence in glare to reduce false positives
            params['conf'] = min(0.5, YOLO_CONF * 1.15)
        
        # Adjust IoU threshold for better separation in challenging conditions
        if diagnostics.low_contrast:
            # Lower IoU to separate overlapping objects better
            params['iou'] = max(0.3, YOLO_IOU * 0.9)
        
        # Enable TTA (Test Time Augmentation) for challenging scenes
        if diagnostics.low_light or diagnostics.glare_risk:
            params['augment'] = True
    
    return params


def fix_common_misclassifications(
    rows: List[Tuple[str, float, Tuple[float, float, float, float]]],
    frame_w: float,
    frame_h: float
) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    """
    Fix common YOLO misclassifications based on size, position, and context.
    
    Common issues:
    - Person detected as truck/car (when person fills most of frame)
    - Small objects detected as large vehicles
    - Indoor objects detected as vehicles
    - Misclassified objects based on aspect ratio and position
    """
    fixed_rows = []
    
    # First pass: collect all labels for context analysis
    all_labels = [row[0] for row in rows]
    indoor_indicators = ['laptop', 'keyboard', 'mouse', 'book', 'cup', 'bottle', 'cell phone', 'tv', 'clock']
    outdoor_indicators = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic light', 'stop sign']
    
    indoor_count = sum(1 for label in all_labels if label in indoor_indicators)
    outdoor_count = sum(1 for label in all_labels if label in outdoor_indicators)
    
    # Determine likely environment context
    likely_indoor = indoor_count > outdoor_count and indoor_count >= 2
    likely_outdoor = outdoor_count > indoor_count and outdoor_count >= 1
    
    for label, conf, bbox in rows:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / max(height, 1)
        
        # Calculate relative size (fraction of frame)
        area_fraction = area / (frame_w * frame_h)
        width_fraction = width / frame_w
        height_fraction = height / frame_h
        
        # Center position (normalized)
        center_x = (x1 + x2) / 2 / frame_w
        center_y = (y1 + y2) / 2 / frame_h
        
        original_label = label
        
        # Fix 1: Large objects detected as vehicles are likely persons
        if label in ['truck', 'bus', 'car'] and area_fraction > 0.12:
            # If object takes up >12% of frame, likely a person close to camera
            if aspect_ratio < 2.2 and height_fraction > 0.3:  # Tall enough to be person
                label = 'person'
                conf *= 0.9  # Slightly reduce confidence for corrected label
        
        # Fix 2: Very small "vehicles" are likely other objects
        elif label in ['truck', 'bus', 'car', 'motorcycle'] and area_fraction < 0.008:
            # Tiny vehicles are suspicious - could be phones, books, etc.
            if aspect_ratio > 1.8:
                label = 'cell phone'  # Wide small object
            elif aspect_ratio < 0.8:
                label = 'book'  # Tall small object
            else:
                label = 'mouse'  # Square small object
            conf *= 0.75
        
        # Fix 3: Tall narrow "vehicles" are likely persons
        elif label in ['truck', 'bus', 'car'] and aspect_ratio < 0.65:
            # Tall narrow objects are usually people
            if height_fraction > 0.25:  # Must be reasonably tall
                label = 'person'
                conf *= 0.85
        
        # Fix 4: Objects in center of frame with high area are likely persons
        elif label in ['truck', 'bus'] and area_fraction > 0.06:
            if 0.25 < center_x < 0.75 and 0.15 < center_y < 0.85:  # Centered
                if aspect_ratio < 1.8:  # Not too wide
                    label = 'person'
                    conf *= 0.9
        
        # Fix 5: Very wide objects detected as persons are likely vehicles
        elif label == 'person' and aspect_ratio > 2.8 and area_fraction < 0.15:
            # Wide, not too large = likely a car
            if likely_outdoor or outdoor_count > 0:
                label = 'car'
                conf *= 0.8
        
        # Fix 6: Indoor context - vehicles indoors are suspicious
        if label in ['truck', 'bus', 'car', 'motorcycle'] and likely_indoor:
            # We're clearly indoors, vehicles are suspicious
            if area_fraction > 0.08:  # Large "vehicle" indoors = person
                label = 'person'
                conf *= 0.7
            elif area_fraction < 0.025:  # Small "vehicle" indoors = object
                if aspect_ratio > 1.5:
                    label = 'cell phone'
                else:
                    label = 'laptop' if area_fraction > 0.01 else 'mouse'
                conf *= 0.6
        
        # Fix 7: Motorcycle/bicycle confusion with person
        elif label in ['motorcycle', 'bicycle'] and area_fraction > 0.1:
            if aspect_ratio < 1.2 and likely_indoor:  # Tall object indoors
                label = 'person'
                conf *= 0.8
        
        # Fix 8: Person detected as small vehicle (common with partial person in frame)
        elif label in ['motorcycle', 'bicycle'] and area_fraction < 0.03:
            if aspect_ratio < 1.5:  # Not too wide
                label = 'person'
                conf *= 0.85
        
        # Fix 9: Position-based corrections for edge cases
        if label in ['truck', 'bus']:
            # Very top or bottom of frame "trucks" are often misclassified
            if center_y < 0.15 or center_y > 0.85:
                if area_fraction < 0.05:
                    label = 'car'  # Smaller vehicle more likely
                    conf *= 0.8
        
        # Fix 10: Confidence-based corrections
        if conf < 0.4:  # Low confidence detections need stricter rules
            if label in ['truck', 'bus'] and area_fraction < 0.02:
                continue  # Skip very small, low-confidence large vehicles
            elif label in ['car', 'motorcycle'] and area_fraction > 0.2:
                label = 'person'  # Large, low-confidence vehicles = person
                conf *= 0.7
        
        # Only keep if confidence is still reasonable after corrections
        if conf >= 0.12:  # Minimum confidence threshold (lowered slightly)
            fixed_rows.append((label, conf, bbox))
    
    return fixed_rows


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "api_version": API_VERSION,
        "mode": "yolo" if MODEL is not None else "demo",
        "message": "Backend ready",
        "model": YOLO_MODEL_NAME if MODEL else None,
        "model_load_error": MODEL_LOAD_ERROR,
        "yolo_conf": YOLO_CONF,
        "max_det": YOLO_MAX_DET,
        "yolo_tta": YOLO_TTA,
        "disable_person_fp_filter": DISABLE_PERSON_FP_FILTER,
        "calibration": CALIBRATION.model_dump(),
    }


@app.get("/calibration", response_model=CalibrationConfig)
def get_calibration() -> CalibrationConfig:
    return CALIBRATION


@app.post("/calibration", response_model=CalibrationConfig)
def set_calibration(patch: CalibrationPatch) -> CalibrationConfig:
    global CALIBRATION
    data = CALIBRATION.model_dump()
    if patch.focal_like is not None:
        data["focal_like"] = float(max(200.0, min(2500.0, patch.focal_like)))
    if patch.meters_per_px is not None:
        data["meters_per_px"] = float(max(0.005, min(0.5, patch.meters_per_px)))
    if patch.default_object_height_m is not None:
        data["default_object_height_m"] = float(max(0.3, min(4.5, patch.default_object_height_m)))
    CALIBRATION = CalibrationConfig(**data)
    return CALIBRATION


@app.get("/demo-detections", response_model=AnalyzeResponse)
def demo_detections() -> AnalyzeResponse:
    """No random labels: empty list when only demo mode, or run real model if loaded."""
    now_s = time.time()
    return AnalyzeResponse(frame_time_s=now_s, detections=[], trip=trip_snapshot(now_s))


@app.get("/trip/stats", response_model=TripStatsOut)
def trip_stats() -> TripStatsOut:
    """Phase 4: session counters and recent near-miss / caution events (newest first)."""
    now_s = time.time()
    events = [TripEventOut(**e) for e in TRIP_EVENTS]
    return TripStatsOut(
        trip_started_s=TRIP_STARTED_S,
        frames=TRIP_FRAMES,
        danger_frames=TRIP_DANGER_FRAMES,
        caution_frames=TRIP_CAUTION_FRAMES,
        safe_frames=TRIP_SAFE_FRAMES,
        near_miss_count=TRIP_NEAR_MISS_COUNT,
        trip_elapsed_s=round(max(0.0, now_s - TRIP_STARTED_S), 1),
        events=events,
    )


@app.post("/trip/reset")
def trip_reset() -> dict:
    """Clear Phase 4 trip stats and event log for a new session."""
    reset_trip_state()
    return {"ok": True}


@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...), user_id: str = "default", 
                       location_lat: float = None, location_lon: float = None,
                       user_speed: float = 0.0, acceleration: float = 0.0) -> AnalyzeResponse:
    now_s = time.time()
    if np is None or Image is None:
        return AnalyzeResponse(frame_time_s=now_s, detections=[], trip=trip_snapshot(now_s))

    raw = await file.read()
    image = Image.open(io_bytes(raw)).convert("RGB")
    frame = np.array(image)
    fh, fw = float(frame.shape[0]), float(frame.shape[1])
    diagnostics = compute_frame_diagnostics(frame)

    # Night / low-light enhancement: lift dark vehicle bodies, suppress headlight halos.
    # Runs automatically when the scene is dark or low-contrast; no-op in daylight.
    frame = enhance_night_frame(frame, diagnostics)

    if MODEL is None:
        return AnalyzeResponse(
            frame_time_s=now_s,
            detections=[],
            trip=trip_snapshot(now_s),
            frame_diagnostics=diagnostics,
        )

    # Get personalized settings for this user
    user_location = (location_lat, location_lon) if location_lat and location_lon else None
    context = {
        'weather': diagnostics.quality_hint if diagnostics else 'unknown',
        'hour': int((now_s % 86400) / 3600),
        'lighting': 'night' if diagnostics and diagnostics.low_light else 'day'
    }
    
    personalized_settings = learning_service.get_personalized_settings(user_id, context)
    environment_adaptations = learning_service.adapt_to_environment(user_id, context)

    # Get dynamic YOLO parameters based on scene conditions and performance mode
    if should_skip_heavy_processing():
        # Ultra-fast mode - use optimized parameters
        yolo_params = get_yolo_params()
    else:
        # Heavy AI mode - use dynamic parameters
        yolo_params = get_dynamic_yolo_params(diagnostics)
        
        # Apply personalized sensitivity adjustments in heavy mode
        if 'confidence_threshold' in environment_adaptations:
            yolo_params['conf'] = environment_adaptations['confidence_threshold']
    
    results = MODEL.predict(
        frame,
        verbose=False,
        **yolo_params
    )

    raw_rows: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_idx = int(b.cls[0].item())
            label = MODEL.names.get(cls_idx, str(cls_idx))
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            bbox = (x1, y1, x2, y2)
            
            # Apply personalized sensitivity filtering
            user_sensitivity = personalized_settings.get('sensitivity', {}).get(label, 0.5)
            if conf < user_sensitivity:
                continue
                
            if not DISABLE_PERSON_FP_FILTER and not person_false_positive_filter(label, conf, bbox, fw, fh):
                continue
            if not keep_detection_for_ride_mount(label, bbox, fh):
                continue
            raw_rows.append((label, conf, bbox))

    raw_rows = resolve_person_cell_phone_conflicts(raw_rows)
    
    # Apply misclassification fixes based on size, position, and context
    raw_rows = fix_common_misclassifications(raw_rows, fw, fh)

    # Convert to detection dictionaries for advanced processing
    initial_detections = []
    for label, conf, bbox in raw_rows:
        initial_detections.append({
            "label": label,
            "confidence": conf,
            "bbox_xyxy": list(bbox)
        })

    # === ADVANCED RAG & AI ENHANCEMENT PIPELINE ===
    
    # Check if we should skip heavy processing for ultra-fast mode
    if should_skip_heavy_processing():
        # Ultra-fast mode - skip all heavy AI processing
        final_enhanced_detections = []
        for label, conf, bbox in raw_rows:
            final_enhanced_detections.append({
                "label": label,
                "confidence": conf,
                "bbox_xyxy": list(bbox)
            })
    else:
        # Heavy AI mode - run all enhancement layers
        
        # Step 1: RAG-Enhanced Detection Analysis (only if enabled)
        if should_use_rag():
            rag_enhanced_detections = rag_enhancer.enhance_detections_with_rag(
                initial_detections, 
                diagnostics.__dict__ if diagnostics else {}
            )
        else:
            rag_enhanced_detections = initial_detections
        
        # Step 2: Knowledge Graph Contextual Analysis (only if enabled)
        if should_use_knowledge_graph():
            scene_context = "indoor" if any(d.get("label") in ["laptop", "keyboard", "tv"] for d in rag_enhanced_detections) else "outdoor"
            kg_analysis = knowledge_graph.analyze_detection_context(rag_enhanced_detections, scene_context)
            
            # Apply knowledge graph corrections
            kg_corrected_detections = []
            for detection in rag_enhanced_detections:
                corrected = detection.copy()
                
                # Apply confidence adjustments from knowledge graph
                obj_type = detection.get("label", "")
                if obj_type in kg_analysis["confidence_adjustments"]:
                    adj_factor = kg_analysis["confidence_adjustments"][obj_type]
                    corrected["confidence"] = min(0.99, corrected["confidence"] * adj_factor)
                
                # Apply suggested corrections
                for suggestion in kg_analysis["suggested_corrections"]:
                    if (suggestion["action"] == "relabel_detection" and 
                        suggestion["current_label"] == obj_type and
                        suggestion["confidence"] > 0.7):
                        corrected["label"] = suggestion["suggested_label"]
                        corrected["confidence"] *= 0.9  # Slight confidence reduction for corrections
                        corrected["kg_correction"] = suggestion["reason"]
                
                kg_corrected_detections.append(corrected)
        else:
            kg_corrected_detections = rag_enhanced_detections
            kg_analysis = {"confidence_adjustments": {}, "context_consistency": {}}
        
        # Step 3: Ensemble Detection (if enabled)
        if should_use_ensemble():
            try:
                ensemble_result = await ensemble_system.ensemble_detect(frame, MODEL, diagnostics.__dict__ if diagnostics else {})
                final_enhanced_detections = ensemble_result.final_detections
                
                # Add ensemble metadata
                for detection in final_enhanced_detections:
                    detection["ensemble_consensus"] = ensemble_result.consensus_strength
                    detection["model_agreement"] = ensemble_result.model_agreements.get("overall", 1.0)
                    
            except Exception as e:
                # Fallback to knowledge graph corrected detections if ensemble fails
                print(f"Ensemble detection failed: {e}")
                final_enhanced_detections = kg_corrected_detections
        else:
            final_enhanced_detections = kg_corrected_detections

    # Convert back to the original pipeline format
    out: List[DetectionOut] = []
    global NEXT_ID

    for detection in final_enhanced_detections:
        label = detection.get("label", "")
        conf = detection.get("confidence", 0.5)
        bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
        
        cx, cy = centroid(tuple(bbox))
        tid = match_track(cx, cy, now_s, fw, fh)
        if tid is None:
            tid = NEXT_ID
            NEXT_ID += 1
        if tid not in TRACK_FIRST_SEEN_S:
            TRACK_FIRST_SEEN_S[tid] = now_s
        track_age_s = max(0.0, now_s - TRACK_FIRST_SEEN_S[tid])
        speed_mps = update_track(tid, cx, cy, now_s)
        speed_kmh_raw = max(0.0, speed_mps * 3.6)
        is_moving = speed_kmh_raw >= MOVING_SPEED_THRESHOLD_KMH
        speed_kmh = speed_kmh_raw if is_moving else 0.0
        smooth_label, smooth_conf = smooth_label_from_history(tid, label, conf)
        distance_m = estimate_distance_m(smooth_label, tuple(bbox), fw, fh)

        if smooth_label in VEHICLE_LABELS and is_moving:
            ttc_s, risk = ttc_and_risk(distance_m, speed_kmh)
        elif smooth_label == "person":
            # Static person still gets a proximity-based risk so they appear in the HUD
            risk = max(0.0, min(60.0, (1.0 - distance_m / 10.0) * 60.0)) if distance_m < 10.0 else 5.0
            ttc_s = 999.0
        else:
            ttc_s, risk = 999.0, 0.0

        x1, y1, x2, y2 = bbox
        detection_out = DetectionOut(
            track_id=tid,
            label=smooth_label,
            confidence=smooth_conf,
            bbox_xyxy=[x1, y1, x2, y2],
            distance_m=distance_m,
            speed_kmh=speed_kmh,
            is_moving=is_moving,
            ttc_s=ttc_s,
            risk_percent=risk,
            track_age_s=track_age_s,
        )
        
        # Add AI enhancement metadata
        if hasattr(detection_out, '__dict__'):
            detection_out.__dict__.update({
                "ai_enhanced": True,
                "rag_metadata": detection.get("rag_metadata", {}),
                "kg_analysis": kg_analysis.get("context_consistency", {}).get(smooth_label, 0.5),
                "ensemble_data": {
                    "consensus": detection.get("ensemble_consensus", 1.0),
                    "agreement": detection.get("model_agreement", 1.0)
                } if "ensemble_consensus" in detection else None
            })
        
        out.append(detection_out)

    out.sort(key=lambda d: d.confidence, reverse=True)
    
    # === SAFETY AND ANALYTICS INTEGRATION ===
    
    # Record detections for analytics
    for detection_out in out:
        detection_dict = {
            'label': detection_out.label,
            'confidence': detection_out.confidence,
            'distance_m': detection_out.distance_m,
            'speed_kmh': detection_out.speed_kmh,
            'risk_percent': detection_out.risk_percent,
            'user_id': user_id
        }
        
        analytics_service.record_detection(
            detection_dict, 
            ai_enhanced=True,
            correction_type="ensemble_rag_kg",
            scene_context=scene_context
        )
    
    # Check for collision predictions and safety alerts
    collision_predictions = safety_service.predict_collision(
        [d.dict() for d in out], user_speed, user_location
    )
    
    # Check for emergency conditions
    emergency_event = safety_service.check_emergency_conditions(
        [d.dict() for d in out], user_speed, acceleration, user_location
    )
    
    # Create safety alerts for high-risk situations
    for detection_out in out:
        if detection_out.risk_percent > 80:
            safety_service.create_safety_alert(
                alert_type="collision_warning",
                severity="HIGH" if detection_out.risk_percent > 90 else "MEDIUM",
                message=f"High risk {detection_out.label} detected at {detection_out.distance_m:.1f}m",
                object_involved=detection_out.dict(),
                auto_dismiss_seconds=10
            )
    
    # Record safety events for high-risk detections
    for detection_out in out:
        if detection_out.risk_percent > 75:
            analytics_service.record_safety_event(
                event_type="high_risk_detection",
                severity="DANGER" if detection_out.risk_percent > 90 else "CAUTION",
                detection=detection_out.dict(),
                weather=context.get('weather', 'unknown'),
                lighting=context.get('lighting', 'unknown'),
                location=user_location
            )
    
    update_trip_stats(out, now_s)
    
    # Enhanced response with AI metadata and safety information
    response = AnalyzeResponse(
        frame_time_s=now_s,
        detections=out,
        trip=trip_snapshot(now_s),
        frame_diagnostics=diagnostics,
    )
    
    # Add comprehensive enhancement summary to response
    if hasattr(response, '__dict__'):
        response.__dict__.update({
            "ai_enhancements": {
                "rag_corrections": len([d for d in final_enhanced_detections if "rag_metadata" in d]),
                "kg_corrections": len(kg_analysis["suggested_corrections"]),
                "relationship_violations": len(kg_analysis["relationship_violations"]),
                "ensemble_enabled": "ensemble_consensus" in (final_enhanced_detections[0] if final_enhanced_detections else {}),
                "scene_context": scene_context,
                "processing_pipeline": ["yolo", "personalization", "rag", "knowledge_graph", "ensemble", "temporal_smoothing"]
            },
            "safety_analysis": {
                "collision_predictions": len(collision_predictions),
                "emergency_event": emergency_event.event_id if emergency_event else None,
                "active_alerts": len(safety_service.get_active_alerts()),
                "personalized_settings_applied": True
            },
            "personalization": {
                "user_id": user_id,
                "settings_applied": personalized_settings,
                "environment_adaptations": environment_adaptations
            }
        })
    
    return response


@app.get("/ai-insights")
def get_ai_insights() -> dict:
    """Get insights from AI enhancement systems"""
    try:
        # Get RAG learning insights
        rag_insights = rag_enhancer.get_learning_insights()
        
        # Get knowledge graph statistics
        kg_stats = {
            "total_nodes": len(knowledge_graph.object_nodes),
            "total_relationships": len(knowledge_graph.relationships),
            "exclusion_rules": len(knowledge_graph.exclusion_matrix)
        }
        
        # Get ensemble performance metrics
        ensemble_metrics = {
            "temporal_tracking_active": len(ensemble_system.temporal_tracker.object_histories),
            "adaptive_learning_patterns": len(ensemble_system.adaptive_learner.performance_history),
            "model_weights": ensemble_system.model_weights
        }
        
        return {
            "ok": True,
            "rag_insights": rag_insights,
            "knowledge_graph_stats": kg_stats,
            "ensemble_metrics": ensemble_metrics,
            "ai_features_active": {
                "rag_enhancement": True,
                "knowledge_graph": True,
                "ensemble_detection": True,
                "temporal_consistency": True,
                "adaptive_learning": True
            }
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "ai_features_active": {
                "rag_enhancement": False,
                "knowledge_graph": False,
                "ensemble_detection": False,
                "temporal_consistency": False,
                "adaptive_learning": False
            }
        }

@app.post("/ai-feedback")
def submit_ai_feedback(feedback: LearningFeedback) -> dict:
    """Submit feedback for AI learning systems"""
    try:
        # Record feedback for adaptive learning
        if feedback.detection_result and feedback.ground_truth:
            ensemble_system.adaptive_learner.record_detection_result(
                feedback.detection_result,
                feedback.ground_truth
            )
        
        # Record in learning service
        learning_service.record_detection_feedback(
            user_id=feedback.detection_result.get('user_id', 'default'),
            detection=feedback.detection_result,
            user_correction=feedback.user_correction or '',
            feedback_type=feedback.feedback_type
        )
        
        return {"ok": True, "message": "Feedback recorded for AI learning"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# === ANALYTICS ENDPOINTS ===

@app.get("/analytics/performance")
def get_performance_analytics(hours: int = 24) -> dict:
    """Get comprehensive performance analytics"""
    try:
        metrics = analytics_service.get_performance_metrics(hours)
        return {
            "ok": True,
            "performance_metrics": asdict(metrics),
            "time_period_hours": hours
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/analytics/safety")
def get_safety_analytics(hours: int = 24) -> dict:
    """Get safety metrics and trends"""
    try:
        metrics = analytics_service.get_safety_metrics(hours)
        return {
            "ok": True,
            "safety_metrics": asdict(metrics),
            "time_period_hours": hours
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/analytics/ai-enhancements")
def get_ai_enhancement_analytics(hours: int = 24) -> dict:
    """Get AI enhancement statistics"""
    try:
        stats = analytics_service.get_ai_enhancement_stats(hours)
        return {
            "ok": True,
            "ai_enhancement_stats": asdict(stats),
            "time_period_hours": hours
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/analytics/trends")
def get_historical_trends(days: int = 7) -> dict:
    """Get historical trends for charts and graphs"""
    try:
        trends = analytics_service.get_historical_trends(days)
        return {
            "ok": True,
            "trends": trends,
            "time_period_days": days
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/analytics/trip/start")
def start_trip_analytics(user_id: str = "default") -> dict:
    """Start a new trip for analytics tracking"""
    try:
        trip_id = analytics_service.start_trip(user_id)
        return {
            "ok": True,
            "trip_id": trip_id,
            "message": "Trip analytics started"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/analytics/trip/end")
def end_trip_analytics(trip_id: str = None) -> dict:
    """End current trip and get analytics"""
    try:
        trip_analytics = analytics_service.end_trip(trip_id)
        if trip_analytics:
            return {
                "ok": True,
                "trip_analytics": asdict(trip_analytics)
            }
        else:
            return {"ok": False, "error": "Trip not found"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# === SAFETY ENDPOINTS ===

@app.get("/safety/alerts")
def get_active_safety_alerts() -> dict:
    """Get all active safety alerts"""
    try:
        alerts = safety_service.get_active_alerts()
        return {
            "ok": True,
            "alerts": [asdict(alert) for alert in alerts]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/safety/alerts/{alert_id}/dismiss")
def dismiss_safety_alert(alert_id: str) -> dict:
    """Dismiss a safety alert"""
    try:
        success = safety_service.dismiss_alert(alert_id)
        return {
            "ok": success,
            "message": "Alert dismissed" if success else "Alert not found"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/safety/emergency-contact")
def add_emergency_contact(contact: EmergencyContactRequest) -> dict:
    """Add an emergency contact"""
    try:
        contact_id = safety_service.add_emergency_contact(
            name=contact.name,
            phone=contact.phone,
            email=contact.email,
            relationship=contact.relationship,
            priority=contact.priority
        )
        return {
            "ok": True,
            "contact_id": contact_id,
            "message": "Emergency contact added"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/safety/zone")
def add_safety_zone(zone: SafetyZoneRequest) -> dict:
    """Add a safety zone"""
    try:
        zone_id = safety_service.add_safety_zone(
            name=zone.name,
            center=(zone.center_lat, zone.center_lon),
            radius_m=zone.radius_m,
            zone_type=zone.zone_type,
            speed_limit=zone.speed_limit
        )
        return {
            "ok": True,
            "zone_id": zone_id,
            "message": "Safety zone added"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/safety/recommendations")
def get_safety_recommendations(user_id: str = "default") -> dict:
    """Get personalized safety recommendations"""
    try:
        recommendations = analytics_service.generate_safety_recommendations(user_id)
        return {
            "ok": True,
            "recommendations": recommendations
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# === LEARNING & PERSONALIZATION ENDPOINTS ===

@app.post("/learning/user-profile")
def create_user_profile(user_id: str, mobility_mode: str = "cycling") -> dict:
    """Create a new user profile"""
    try:
        profile = learning_service.create_user_profile(user_id, mobility_mode)
        return {
            "ok": True,
            "profile": asdict(profile),
            "message": "User profile created"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/learning/behavior")
def learn_user_behavior(behavior_data: UserBehaviorData) -> dict:
    """Submit user behavior data for learning"""
    try:
        learning_service.learn_user_behavior(
            behavior_data.user_id,
            behavior_data.session_data
        )
        return {
            "ok": True,
            "message": "Behavior data processed for learning"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/learning/custom-object")
def train_custom_object(training_request: CustomObjectTraining) -> dict:
    """Train a custom object detector"""
    try:
        custom_object = learning_service.train_custom_object(
            training_request.user_id,
            training_request.object_name,
            training_request.training_images
        )
        return {
            "ok": True,
            "custom_object": asdict(custom_object),
            "message": "Custom object training completed"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/learning/personalized-settings")
def get_personalized_settings(request: PersonalizedSettingsRequest) -> dict:
    """Get personalized detection settings for user"""
    try:
        settings = learning_service.get_personalized_settings(
            request.user_id,
            request.context
        )
        return {
            "ok": True,
            "settings": settings
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/learning/environment-adaptation")
def adapt_to_environment(user_id: str, environment_data: dict) -> dict:
    """Get environment-adapted detection parameters"""
    try:
        adaptations = learning_service.adapt_to_environment(user_id, environment_data)
        return {
            "ok": True,
            "adaptations": adaptations
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def io_bytes(raw: bytes):
    import io

    return io.BytesIO(raw)


# One-server deploy option: serve exported web build from backend process.
WEB_BUILD_DIR = Path(
    os.environ.get(
        "CAR_VISION_WEB_DIR",
        str((Path(__file__).resolve().parent.parent / "web-dist")),
    )
).resolve()

if WEB_BUILD_DIR.exists():
    for static_subdir in ("assets", "_expo", "static"):
        static_path = WEB_BUILD_DIR / static_subdir
        if static_path.exists():
            app.mount(f"/{static_subdir}", StaticFiles(directory=str(static_path)), name=f"web_{static_subdir}")

    @app.get("/", include_in_schema=False)
    def web_index() -> FileResponse:
        return FileResponse(str(WEB_BUILD_DIR / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def web_spa_fallback(full_path: str) -> FileResponse:
        # Never shadow API/documentation routes.
        if full_path.startswith(("health", "calibration", "demo-detections", "trip", "analyze-image", "docs", "openapi.json", "redoc")):
            raise HTTPException(status_code=404, detail="Not found")
        target = (WEB_BUILD_DIR / full_path).resolve()
        try:
            target.relative_to(WEB_BUILD_DIR)
        except ValueError:
            return FileResponse(str(WEB_BUILD_DIR / "index.html"))
        if target.exists() and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(WEB_BUILD_DIR / "index.html"))

# Motorcycle 360° Vision System Endpoints

class Motorcycle360Request(BaseModel):
    """Request for 360° motorcycle vision processing"""
    bike_speed: Optional[float] = None
    bike_heading: Optional[float] = None
    fallback_mode: bool = False

@app.post("/motorcycle-360/process-frames")
async def process_motorcycle_360_frames(
    request: Motorcycle360Request,
    front_camera: UploadFile = File(...),
    left_camera: Optional[UploadFile] = File(None),
    right_camera: Optional[UploadFile] = File(None),
    rear_camera: Optional[UploadFile] = File(None)
):
    """
    Process 4 camera feeds for Tesla-style 360° motorcycle vision
    Returns comprehensive situational awareness data
    """
    if not motorcycle_vision:
        raise HTTPException(status_code=503, detail="Motorcycle 360° Vision system not available")
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Process camera frames
        frames = {}
        
        # Front camera (required)
        front_image = Image.open(front_camera.file)
        frames['front'] = cv2.cvtColor(np.array(front_image), cv2.COLOR_RGB2BGR)
        
        # Optional side cameras
        if left_camera:
            left_image = Image.open(left_camera.file)
            frames['left'] = cv2.cvtColor(np.array(left_image), cv2.COLOR_RGB2BGR)
        
        if right_camera:
            right_image = Image.open(right_camera.file)
            frames['right'] = cv2.cvtColor(np.array(right_image), cv2.COLOR_RGB2BGR)
        
        if rear_camera:
            rear_image = Image.open(rear_camera.file)
            frames['rear'] = cv2.cvtColor(np.array(rear_image), cv2.COLOR_RGB2BGR)
        
        # Process with motorcycle vision system
        if len(frames) == 1 or request.fallback_mode:
            # Fallback mode with front camera only
            result = motorcycle_vision.process_single_camera_fallback(
                frames['front'],
                bike_speed=request.bike_speed,
                bike_heading=request.bike_heading
            )
        else:
            # Full 4-camera processing
            result = motorcycle_vision.process_frame_set(
                frames,
                bike_speed=request.bike_speed,
                bike_heading=request.bike_heading
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing 360° vision: {str(e)}")

@app.post("/motorcycle-360/process-single")
async def process_motorcycle_360_single(
    request: Motorcycle360Request,
    camera_frame: UploadFile = File(...)
):
    """
    Process single camera frame in fallback mode
    """
    if not motorcycle_vision:
        raise HTTPException(status_code=503, detail="Motorcycle 360° Vision system not available")
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Process single camera frame
        image = Image.open(camera_frame.file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        result = motorcycle_vision.process_single_camera_fallback(
            frame,
            bike_speed=request.bike_speed,
            bike_heading=request.bike_heading
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing single camera: {str(e)}")

@app.get("/motorcycle-360/status")
def get_motorcycle_360_status():
    """Get status of the motorcycle 360° vision system"""
    return {
        "available": motorcycle_vision is not None,
        "system_info": {
            "version": "1.0.0",
            "features": [
                "4-camera processing",
                "fallback mode",
                "hazard detection",
                "distance estimation",
                "motion analysis",
                "lane detection",
                "voice alerts"
            ],
            "supported_modes": [
                "4-camera surround",
                "single camera fallback"
            ]
        },
        "performance": {
            "frames_processed": getattr(motorcycle_vision, 'frame_count', 0) if motorcycle_vision else 0,
            "uptime_seconds": time.time() - getattr(motorcycle_vision, 'start_time', time.time()) if motorcycle_vision else 0
        }
    }

# Surround Vision Renderer Endpoints

@app.post("/surround-vision/render")
async def render_surround_vision(
    request: SurroundVisionRequest,
    front_camera: UploadFile = File(...)
):
    """
    Generate 360° surround vision scenes based on front camera and detection data
    Returns animated scene data for left, right, and rear panels
    """
    if not surround_renderer:
        raise HTTPException(status_code=503, detail="Surround Vision Renderer not available")
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Process front camera frame
        front_image = Image.open(front_camera.file)
        front_frame = cv2.cvtColor(np.array(front_image), cv2.COLOR_RGB2BGR)
        
        # Convert detected objects to proper format
        detected_objects = []
        for obj_data in request.detected_objects:
            detected_objects.append(DetectedObject(
                label=obj_data.get('label', 'unknown'),
                confidence=obj_data.get('confidence', 0.5),
                bbox=obj_data.get('bbox', [0, 0, 50, 50]),
                distance_m=obj_data.get('distance_m', 10.0),
                position=obj_data.get('position', 'center'),
                is_moving=obj_data.get('is_moving', False)
            ))
        
        # Generate surround vision scene
        scene_data = surround_renderer.render_frame(
            front_frame=front_frame,
            detected_objects=detected_objects,
            road_type=request.road_type,
            speed=request.speed,
            turn_direction=request.turn_direction
        )
        
        return scene_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rendering surround vision: {str(e)}")

@app.post("/surround-vision/render-mock")
def render_surround_vision_mock(request: SurroundVisionRequest):
    """
    Generate mock surround vision scenes without camera input (for testing)
    """
    if not surround_renderer:
        raise HTTPException(status_code=503, detail="Surround Vision Renderer not available")
    
    try:
        # Convert detected objects to proper format
        detected_objects = []
        for obj_data in request.detected_objects:
            detected_objects.append(DetectedObject(
                label=obj_data.get('label', 'unknown'),
                confidence=obj_data.get('confidence', 0.5),
                bbox=obj_data.get('bbox', [0, 0, 50, 50]),
                distance_m=obj_data.get('distance_m', 10.0),
                position=obj_data.get('position', 'center'),
                is_moving=obj_data.get('is_moving', False)
            ))
        
        # Generate surround vision scene without front frame
        scene_data = surround_renderer.render_frame(
            front_frame=None,  # No camera input
            detected_objects=detected_objects,
            road_type=request.road_type,
            speed=request.speed,
            turn_direction=request.turn_direction
        )
        
        return scene_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rendering mock surround vision: {str(e)}")

@app.get("/surround-vision/status")
def get_surround_vision_status():
    """Get status of the surround vision renderer"""
    return {
        "available": surround_renderer is not None,
        "system_info": {
            "version": "1.0.0",
            "features": [
                "360° scene generation",
                "animated road markings",
                "weather effects",
                "object inference",
                "parallax scrolling",
                "seamless stitching"
            ],
            "supported_road_types": ["urban", "highway", "rural", "parking", "offroad"],
            "supported_weather": ["clear", "rain", "fog", "night"]
        },
        "performance": {
            "frames_rendered": getattr(surround_renderer, 'frame_count', 0) if surround_renderer else 0,
            "scene_elements_active": 0,  # Would track active elements
            "animation_fps": 60
        }
    }