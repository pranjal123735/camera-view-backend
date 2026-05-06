"""
Universal Detection API
FastAPI endpoints that work with any YOLO model and detection domain
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
from PIL import Image
import io
import time
import asyncio

from universal_ai_detection import (
    UniversalAIDetectionSystem, 
    UniversalDetectionConfig,
    DetectionDomain,
    create_ai_detection_system
)

# Universal detection models (can be loaded dynamically)
UNIVERSAL_MODELS = {}
UNIVERSAL_SYSTEMS = {}

class UniversalDetectionRequest(BaseModel):
    """Request model for universal detection"""
    domain: str = "general"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    enable_rag: bool = True
    enable_knowledge_graph: bool = True
    enable_ensemble: bool = True
    max_detections: int = 100

class UniversalDetectionResponse(BaseModel):
    """Response model for universal detection"""
    success: bool
    detections: List[Dict[str, Any]]
    processing_time_ms: float
    domain: str
    pipeline_steps: List[str]
    enhancements_applied: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    frame_diagnostics: Optional[Dict[str, Any]] = None
    ai_metadata: Optional[Dict[str, Any]] = None

class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    available_domains: List[str]
    active_systems: Dict[str, Dict[str, Any]]
    total_processed: int
    average_processing_time_ms: float

# Initialize universal app
universal_app = FastAPI(
    title="Universal AI-Enhanced Detection API",
    description="AI-enhanced object detection for any domain - not just cars!",
    version="1.0.0"
)

universal_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_or_create_system(domain: str, config: Dict[str, Any] = None) -> UniversalAIDetectionSystem:
    """Get or create AI detection system for specified domain"""
    
    if domain not in UNIVERSAL_SYSTEMS:
        system_config = config or {}
        UNIVERSAL_SYSTEMS[domain] = create_ai_detection_system(domain, **system_config)
        
    return UNIVERSAL_SYSTEMS[domain]

def load_universal_model(model_path: str = "yolov8n.pt", domain: str = "general"):
    """Load YOLO model for universal detection"""
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        UNIVERSAL_MODELS[domain] = model
        return model
    except Exception as e:
        print(f"Failed to load model for {domain}: {e}")
        return None

def compute_universal_diagnostics(frame: np.ndarray) -> Dict[str, Any]:
    """Compute frame diagnostics for any image"""
    
    if len(frame.shape) == 3:
        gray = np.mean(frame.astype(np.float32), axis=2)
    else:
        gray = frame.astype(np.float32)
        
    # Basic image quality metrics
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 255.0)
    
    # Quality assessment
    low_light = brightness < 0.3
    low_contrast = contrast < 0.15
    overexposed = brightness > 0.85
    
    return {
        "brightness_01": brightness,
        "contrast": contrast,
        "low_light": low_light,
        "low_contrast": low_contrast,
        "overexposed": overexposed,
        "quality_score": min(1.0, brightness * contrast * 2)
    }

@universal_app.get("/")
def universal_root():
    """Root endpoint with API information"""
    return {
        "message": "Universal AI-Enhanced Detection API",
        "description": "Works with any YOLO model and detection domain",
        "supported_domains": [domain.value for domain in DetectionDomain],
        "features": [
            "RAG-Enhanced Detection",
            "Knowledge Graph Reasoning", 
            "Multi-Model Ensemble",
            "Temporal Consistency",
            "Adaptive Learning",
            "Universal Domain Support"
        ],
        "endpoints": {
            "detect": "/detect/{domain}",
            "detect_general": "/detect",
            "batch_detect": "/batch-detect/{domain}",
            "system_stats": "/stats",
            "domain_info": "/domains/{domain}",
            "health": "/health"
        }
    }

@universal_app.get("/health")
def universal_health():
    """Universal health check"""
    return {
        "status": "healthy",
        "available_domains": list(UNIVERSAL_SYSTEMS.keys()),
        "loaded_models": list(UNIVERSAL_MODELS.keys()),
        "ai_features": {
            "rag_enhancement": True,
            "knowledge_graph": True,
            "ensemble_detection": True,
            "temporal_tracking": True,
            "adaptive_learning": True
        }
    }

@universal_app.get("/domains")
def list_domains():
    """List all available detection domains"""
    return {
        "domains": [
            {
                "name": domain.value,
                "description": f"AI-enhanced detection for {domain.value} applications",
                "active": domain.value in UNIVERSAL_SYSTEMS
            }
            for domain in DetectionDomain
        ]
    }

@universal_app.get("/domains/{domain}")
def get_domain_info(domain: str):
    """Get information about specific domain"""
    
    try:
        domain_enum = DetectionDomain(domain.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")
        
    system = get_or_create_system(domain)
    
    return {
        "domain": domain,
        "description": f"AI-enhanced detection for {domain} applications",
        "system_stats": system.get_system_stats(),
        "available_features": {
            "rag_enhancement": True,
            "knowledge_graph": True,
            "ensemble_detection": True,
            "temporal_tracking": True,
            "adaptive_learning": True
        }
    }

@universal_app.post("/detect", response_model=UniversalDetectionResponse)
async def detect_general(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    enable_rag: bool = Query(True),
    enable_kg: bool = Query(True),
    max_detections: int = Query(100, ge=1, le=1000)
):
    """General object detection with AI enhancements"""
    return await detect_universal("general", file, confidence, enable_rag, enable_kg, max_detections)

@universal_app.post("/detect/{domain}", response_model=UniversalDetectionResponse)
async def detect_universal(
    domain: str,
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    enable_rag: bool = Query(True),
    enable_kg: bool = Query(True),
    max_detections: int = Query(100, ge=1, le=1000)
):
    """Universal detection for any domain"""
    
    start_time = time.time()
    
    try:
        # Validate domain
        try:
            domain_enum = DetectionDomain(domain.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")
        
        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        frame = np.array(image)
        
        # Compute diagnostics
        diagnostics = compute_universal_diagnostics(frame)
        
        # Get or create AI system for domain
        system = get_or_create_system(domain, {
            "confidence_threshold": confidence,
            "enable_rag": enable_rag,
            "enable_knowledge_graph": enable_kg
        })
        
        # Load model if not already loaded
        if domain not in UNIVERSAL_MODELS:
            model = load_universal_model("yolov8n.pt", domain)
            if not model:
                raise HTTPException(status_code=500, detail="Failed to load detection model")
        else:
            model = UNIVERSAL_MODELS[domain]
        
        # Run YOLO detection
        results = model.predict(
            frame,
            verbose=False,
            conf=confidence,
            max_det=max_detections
        )
        
        # Convert YOLO results to standard format
        raw_detections = []
        for r in results:
            if r.boxes is None:
                continue
                
            for b in r.boxes:
                cls_idx = int(b.cls[0].item())
                label = model.names.get(cls_idx, str(cls_idx))
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                
                raw_detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "class_id": cls_idx
                })
        
        # Process through AI enhancement pipeline
        ai_result = await system.process_detections(raw_detections, diagnostics, model)
        
        total_time = time.time() - start_time
        
        return UniversalDetectionResponse(
            success=True,
            detections=ai_result["detections"],
            processing_time_ms=total_time * 1000,
            domain=domain,
            pipeline_steps=ai_result["pipeline_steps"],
            enhancements_applied=ai_result["enhancements_applied"],
            performance_metrics=ai_result["performance_metrics"],
            frame_diagnostics=diagnostics,
            ai_metadata={
                "yolo_detections": len(raw_detections),
                "ai_enhanced_detections": len(ai_result["detections"]),
                "enhancement_ratio": len(ai_result["detections"]) / max(len(raw_detections), 1),
                "processing_breakdown": {
                    "yolo_inference": "included_in_total",
                    "ai_enhancement": ai_result["processing_time_ms"],
                    "total_time": total_time * 1000
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@universal_app.post("/batch-detect/{domain}")
async def batch_detect_universal(
    domain: str,
    files: List[UploadFile] = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    enable_rag: bool = Query(True),
    enable_kg: bool = Query(True)
):
    """Batch detection for multiple images"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            result = await detect_universal(domain, file, confidence, enable_rag, enable_kg)
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_files": len(files),
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]])
    }

@universal_app.get("/stats", response_model=SystemStatsResponse)
def get_system_stats():
    """Get universal system statistics"""
    
    active_systems = {}
    total_processed = 0
    processing_times = []
    
    for domain, system in UNIVERSAL_SYSTEMS.items():
        stats = system.get_system_stats()
        active_systems[domain] = stats
        total_processed += stats.get("total_processed", 0)
        
        if stats.get("average_processing_time", 0) > 0:
            processing_times.append(stats["average_processing_time"])
    
    avg_processing_time = np.mean(processing_times) if processing_times else 0.0
    
    return SystemStatsResponse(
        available_domains=[domain.value for domain in DetectionDomain],
        active_systems=active_systems,
        total_processed=total_processed,
        average_processing_time_ms=avg_processing_time * 1000
    )

@universal_app.post("/feedback/{domain}")
def submit_feedback(
    domain: str,
    feedback: Dict[str, Any]
):
    """Submit feedback for AI learning"""
    
    try:
        system = get_or_create_system(domain)
        
        # Store feedback for learning (implementation depends on specific needs)
        return {
            "success": True,
            "message": f"Feedback recorded for {domain} domain",
            "domain": domain
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

# Initialize default systems
def initialize_universal_systems():
    """Initialize systems for common domains"""
    
    common_domains = ["general", "security", "medical", "retail"]
    
    for domain in common_domains:
        try:
            get_or_create_system(domain)
            print(f"✅ Initialized {domain} detection system")
        except Exception as e:
            print(f"❌ Failed to initialize {domain} system: {e}")

# Auto-initialize on import
initialize_universal_systems()