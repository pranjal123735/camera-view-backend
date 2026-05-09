"""
Performance Controller for Car Vision Backend
Handles switching between Ultra-Fast and Heavy AI modes
Works in both local and deployed environments
"""

import os
import json
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Initialize performance state based on environment variables
def init_performance_state():
    """Initialize performance state from environment variables"""
    ultra_fast = os.environ.get("CAR_VISION_ULTRA_FAST", "1").lower() in ("1", "true", "yes")
    
    return {
        "ultra_fast_mode": ultra_fast,
        "rag_enabled": not ultra_fast,
        "knowledge_graph_enabled": not ultra_fast,
        "ensemble_enabled": not ultra_fast,
        "learning_enabled": not ultra_fast,
        "yolo_model": os.environ.get("CAR_VISION_YOLO_MODEL", "yolov8n.pt" if ultra_fast else "yolov8s.pt"),
        "yolo_conf": float(os.environ.get("CAR_VISION_YOLO_CONF", "0.4" if ultra_fast else "0.3")),
        "yolo_iou": float(os.environ.get("CAR_VISION_YOLO_IOU", "0.6" if ultra_fast else "0.5")),
        "yolo_max_det": int(os.environ.get("CAR_VISION_YOLO_MAX_DET", "20" if ultra_fast else "60")),
        "yolo_imgsz": int(os.environ.get("CAR_VISION_YOLO_IMGSZ", "416" if ultra_fast else "640")),
        "yolo_tta": os.environ.get("CAR_VISION_YOLO_TTA", "false").lower() in ("1", "true", "yes")
    }

# Global performance state - initialized from environment
PERFORMANCE_STATE = init_performance_state()

class PerformanceSettings(BaseModel):
    enable_ultra_fast: bool = False
    disable_rag: bool = False
    disable_knowledge_graph: bool = False
    disable_ensemble: bool = False
    disable_learning: bool = False
    yolo_model: str = "yolov8s.pt"
    yolo_conf: float = 0.3
    yolo_iou: float = 0.5
    yolo_max_det: int = 60
    yolo_imgsz: int = 640
    yolo_tta: bool = False

router = APIRouter(prefix="/performance", tags=["performance"])

@router.get("/status")
def get_performance_status() -> Dict[str, Any]:
    """Get current performance configuration"""
    return {
        "mode": "ultra_fast" if PERFORMANCE_STATE["ultra_fast_mode"] else "heavy_ai",
        "ultra_fast_mode": PERFORMANCE_STATE["ultra_fast_mode"],
        "features": {
            "rag_enabled": PERFORMANCE_STATE["rag_enabled"],
            "knowledge_graph_enabled": PERFORMANCE_STATE["knowledge_graph_enabled"],
            "ensemble_enabled": PERFORMANCE_STATE["ensemble_enabled"],
            "learning_enabled": PERFORMANCE_STATE["learning_enabled"]
        },
        "yolo_config": {
            "model": PERFORMANCE_STATE["yolo_model"],
            "confidence": PERFORMANCE_STATE["yolo_conf"],
            "iou": PERFORMANCE_STATE["yolo_iou"],
            "max_detections": PERFORMANCE_STATE["yolo_max_det"],
            "image_size": PERFORMANCE_STATE["yolo_imgsz"],
            "tta": PERFORMANCE_STATE["yolo_tta"]
        },
        "expected_latency": "50-200ms" if PERFORMANCE_STATE["ultra_fast_mode"] else "10-50 seconds",
        "deployment_info": {
            "environment": os.environ.get("DEPLOYMENT_ENV", "local"),
            "platform": os.environ.get("RAILWAY_ENVIRONMENT", os.environ.get("RENDER", "local")),
            "supports_runtime_switching": True
        }
    }

@router.post("/set-ultra-fast")
def set_ultra_fast_mode(settings: PerformanceSettings) -> Dict[str, Any]:
    """Enable ultra-fast mode for real-time performance"""
    
    # Update global state
    PERFORMANCE_STATE.update({
        "ultra_fast_mode": True,
        "rag_enabled": False,
        "knowledge_graph_enabled": False,
        "ensemble_enabled": False,
        "learning_enabled": False,
        "yolo_model": "yolov8n.pt",  # Force nano model for speed
        "yolo_conf": 0.4,            # Higher confidence for fewer false positives
        "yolo_iou": 0.6,             # Higher IoU for better separation
        "yolo_max_det": 20,          # Limit detections for speed
        "yolo_imgsz": 416,           # Smaller image size for speed
        "yolo_tta": False            # Disable test time augmentation
    })
    
    # Update environment variables
    os.environ.update({
        "CAR_VISION_ULTRA_FAST": "1",
        "CAR_VISION_DISABLE_RAG": "1",
        "CAR_VISION_DISABLE_KNOWLEDGE_GRAPH": "1",
        "CAR_VISION_DISABLE_ENSEMBLE": "1",
        "CAR_VISION_DISABLE_LEARNING": "1",
        "CAR_VISION_YOLO_MODEL": "yolov8n.pt",
        "CAR_VISION_YOLO_CONF": "0.4",
        "CAR_VISION_YOLO_IOU": "0.6",
        "CAR_VISION_YOLO_MAX_DET": "20",
        "CAR_VISION_YOLO_IMGSZ": "416",
        "CAR_VISION_YOLO_TTA": "false"
    })
    
    return {
        "success": True,
        "message": "Ultra-fast mode enabled",
        "mode": "ultra_fast",
        "expected_latency": "50-200ms",
        "features_disabled": ["RAG", "Knowledge Graph", "Ensemble", "Learning"],
        "optimizations": [
            "Nano YOLO model (yolov8n.pt)",
            "Reduced image size (416px)",
            "Limited detections (20 max)",
            "Higher confidence threshold (0.4)",
            "Disabled test time augmentation"
        ]
    }

@router.post("/set-heavy-ai")
def set_heavy_ai_mode(settings: PerformanceSettings) -> Dict[str, Any]:
    """Enable heavy AI mode with all advanced features"""
    
    # Update global state
    PERFORMANCE_STATE.update({
        "ultra_fast_mode": False,
        "rag_enabled": True,
        "knowledge_graph_enabled": True,
        "ensemble_enabled": True,
        "learning_enabled": True,
        "yolo_model": "yolov8s.pt",  # Standard model
        "yolo_conf": 0.3,            # Lower confidence for more detections
        "yolo_iou": 0.5,             # Standard IoU
        "yolo_max_det": 60,          # More detections
        "yolo_imgsz": 640,           # Standard image size
        "yolo_tta": False            # Keep TTA disabled for now
    })
    
    # Update environment variables
    os.environ.update({
        "CAR_VISION_ULTRA_FAST": "0",
        "CAR_VISION_DISABLE_RAG": "0",
        "CAR_VISION_DISABLE_KNOWLEDGE_GRAPH": "0",
        "CAR_VISION_DISABLE_ENSEMBLE": "0",
        "CAR_VISION_DISABLE_LEARNING": "0",
        "CAR_VISION_YOLO_MODEL": "yolov8s.pt",
        "CAR_VISION_YOLO_CONF": "0.3",
        "CAR_VISION_YOLO_IOU": "0.5",
        "CAR_VISION_YOLO_MAX_DET": "60",
        "CAR_VISION_YOLO_IMGSZ": "640",
        "CAR_VISION_YOLO_TTA": "false"
    })
    
    return {
        "success": True,
        "message": "Heavy AI mode enabled",
        "mode": "heavy_ai",
        "expected_latency": "10-50 seconds",
        "features_enabled": ["RAG", "Knowledge Graph", "Ensemble", "Learning"],
        "warning": "High latency - NOT suitable for real-time driving!"
    }

@router.post("/apply-settings")
def apply_custom_settings(settings: PerformanceSettings) -> Dict[str, Any]:
    """Apply custom performance settings"""
    
    # Update global state with custom settings
    PERFORMANCE_STATE.update({
        "ultra_fast_mode": settings.enable_ultra_fast,
        "rag_enabled": not settings.disable_rag,
        "knowledge_graph_enabled": not settings.disable_knowledge_graph,
        "ensemble_enabled": not settings.disable_ensemble,
        "learning_enabled": not settings.disable_learning,
        "yolo_model": settings.yolo_model,
        "yolo_conf": settings.yolo_conf,
        "yolo_iou": settings.yolo_iou,
        "yolo_max_det": settings.yolo_max_det,
        "yolo_imgsz": settings.yolo_imgsz,
        "yolo_tta": settings.yolo_tta
    })
    
    # Update environment variables
    os.environ.update({
        "CAR_VISION_ULTRA_FAST": "1" if settings.enable_ultra_fast else "0",
        "CAR_VISION_DISABLE_RAG": "1" if settings.disable_rag else "0",
        "CAR_VISION_DISABLE_KNOWLEDGE_GRAPH": "1" if settings.disable_knowledge_graph else "0",
        "CAR_VISION_DISABLE_ENSEMBLE": "1" if settings.disable_ensemble else "0",
        "CAR_VISION_DISABLE_LEARNING": "1" if settings.disable_learning else "0",
        "CAR_VISION_YOLO_MODEL": settings.yolo_model,
        "CAR_VISION_YOLO_CONF": str(settings.yolo_conf),
        "CAR_VISION_YOLO_IOU": str(settings.yolo_iou),
        "CAR_VISION_YOLO_MAX_DET": str(settings.yolo_max_det),
        "CAR_VISION_YOLO_IMGSZ": str(settings.yolo_imgsz),
        "CAR_VISION_YOLO_TTA": "true" if settings.yolo_tta else "false"
    })
    
    return {
        "success": True,
        "message": "Custom settings applied",
        "current_state": PERFORMANCE_STATE
    }

def should_skip_heavy_processing() -> bool:
    """Check if heavy AI processing should be skipped"""
    return PERFORMANCE_STATE["ultra_fast_mode"]

def should_use_rag() -> bool:
    """Check if RAG enhancement should be used"""
    return PERFORMANCE_STATE["rag_enabled"] and not PERFORMANCE_STATE["ultra_fast_mode"]

def should_use_knowledge_graph() -> bool:
    """Check if knowledge graph should be used"""
    return PERFORMANCE_STATE["knowledge_graph_enabled"] and not PERFORMANCE_STATE["ultra_fast_mode"]

def should_use_ensemble() -> bool:
    """Check if ensemble detection should be used"""
    return PERFORMANCE_STATE["ensemble_enabled"] and not PERFORMANCE_STATE["ultra_fast_mode"]

def should_use_learning() -> bool:
    """Check if learning systems should be used"""
    return PERFORMANCE_STATE["learning_enabled"] and not PERFORMANCE_STATE["ultra_fast_mode"]

def get_yolo_params() -> Dict[str, Any]:
    """Get current YOLO parameters"""
    return {
        "model": PERFORMANCE_STATE["yolo_model"],
        "conf": PERFORMANCE_STATE["yolo_conf"],
        "iou": PERFORMANCE_STATE["yolo_iou"],
        "max_det": PERFORMANCE_STATE["yolo_max_det"],
        "imgsz": PERFORMANCE_STATE["yolo_imgsz"],
        "augment": PERFORMANCE_STATE["yolo_tta"]
    }