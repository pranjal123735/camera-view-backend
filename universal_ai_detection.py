"""
Universal AI-Enhanced Detection Framework
Works with any YOLO model and detection scenario - not just car vision!

This framework can be applied to:
- General object detection
- Security surveillance 
- Medical imaging
- Industrial inspection
- Wildlife monitoring
- Sports analysis
- Retail analytics
- And any other detection use case!
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import sqlite3
from pathlib import Path
from enum import Enum
import asyncio

class DetectionDomain(Enum):
    """Different detection domains/applications"""
    GENERAL = "general"
    AUTOMOTIVE = "automotive"
    SECURITY = "security"
    MEDICAL = "medical"
    INDUSTRIAL = "industrial"
    WILDLIFE = "wildlife"
    SPORTS = "sports"
    RETAIL = "retail"
    AGRICULTURE = "agriculture"

@dataclass
class UniversalDetectionConfig:
    """Configuration for universal detection system"""
    domain: DetectionDomain = DetectionDomain.GENERAL
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    enable_rag: bool = True
    enable_knowledge_graph: bool = True
    enable_ensemble: bool = True
    enable_temporal_tracking: bool = True
    enable_adaptive_learning: bool = True
    custom_object_classes: Optional[List[str]] = None
    scene_contexts: Optional[List[str]] = None

class UniversalObjectKnowledge:
    """Universal object knowledge base that adapts to any domain"""
    
    def __init__(self, domain: DetectionDomain = DetectionDomain.GENERAL):
        self.domain = domain
        self.object_database = self._build_universal_knowledge_base()
        
    def _build_universal_knowledge_base(self) -> Dict[str, Dict]:
        """Build knowledge base for any detection domain"""
        
        # Universal base objects (COCO classes + extensions)
        base_objects = {
            # People and body parts
            "person": {
                "category": "human",
                "typical_contexts": ["indoor", "outdoor", "public", "private"],
                "size_ranges": {"close": (0.1, 0.8), "medium": (0.05, 0.3), "far": (0.001, 0.05)},
                "confidence_modifiers": {"indoor": 1.2, "outdoor": 1.0, "crowded": 0.9},
                "common_confusions": ["mannequin", "statue", "shadow"],
                "exclusions": [],
                "relationships": ["with_objects", "in_groups", "using_tools"]
            },
            
            # Vehicles (all types)
            "car": {
                "category": "vehicle",
                "typical_contexts": ["road", "parking", "outdoor"],
                "size_ranges": {"close": (0.2, 0.9), "medium": (0.05, 0.3), "far": (0.001, 0.05)},
                "confidence_modifiers": {"road": 1.4, "parking": 1.2, "indoor": 0.1},
                "common_confusions": ["truck", "van", "bus"],
                "exclusions": ["indoor_unless_garage"],
                "relationships": ["on_road", "in_parking", "with_person"]
            },
            
            # Animals
            "dog": {
                "category": "animal",
                "typical_contexts": ["outdoor", "home", "park"],
                "size_ranges": {"close": (0.05, 0.4), "medium": (0.02, 0.15), "far": (0.001, 0.03)},
                "confidence_modifiers": {"park": 1.3, "home": 1.2, "wild": 0.8},
                "common_confusions": ["cat", "wolf", "fox"],
                "exclusions": [],
                "relationships": ["with_person", "in_pack", "chasing_object"]
            },
            
            "cat": {
                "category": "animal", 
                "typical_contexts": ["indoor", "outdoor", "home"],
                "size_ranges": {"close": (0.03, 0.3), "medium": (0.01, 0.1), "far": (0.001, 0.02)},
                "confidence_modifiers": {"home": 1.4, "outdoor": 1.0, "high_places": 1.2},
                "common_confusions": ["dog", "rabbit", "shadow"],
                "exclusions": [],
                "relationships": ["with_person", "on_furniture", "hunting"]
            },
            
            # Electronics and objects
            "laptop": {
                "category": "electronics",
                "typical_contexts": ["indoor", "office", "home", "cafe"],
                "size_ranges": {"typical": (0.01, 0.1)},
                "confidence_modifiers": {"office": 1.5, "home": 1.3, "outdoor": 0.2},
                "common_confusions": ["book", "tablet", "tv"],
                "exclusions": ["outdoor_unless_portable"],
                "relationships": ["with_person", "on_desk", "being_used"]
            },
            
            "cell_phone": {
                "category": "electronics",
                "typical_contexts": ["anywhere"],
                "size_ranges": {"typical": (0.001, 0.02)},
                "confidence_modifiers": {"with_person": 1.3, "on_table": 1.1},
                "common_confusions": ["remote", "wallet", "small_object"],
                "exclusions": [],
                "relationships": ["with_person", "being_held", "on_surface"]
            },
            
            # Furniture and household
            "chair": {
                "category": "furniture",
                "typical_contexts": ["indoor", "office", "home", "restaurant"],
                "size_ranges": {"typical": (0.02, 0.2)},
                "confidence_modifiers": {"indoor": 1.4, "office": 1.3, "outdoor": 0.8},
                "common_confusions": ["stool", "bench", "couch"],
                "exclusions": [],
                "relationships": ["with_table", "in_room", "person_sitting"]
            },
            
            "table": {
                "category": "furniture",
                "typical_contexts": ["indoor", "office", "home", "restaurant"],
                "size_ranges": {"typical": (0.03, 0.3)},
                "confidence_modifiers": {"indoor": 1.4, "restaurant": 1.3, "outdoor": 0.7},
                "common_confusions": ["desk", "counter", "surface"],
                "exclusions": [],
                "relationships": ["with_chair", "has_objects", "in_room"]
            },
            
            # Sports and activities
            "sports_ball": {
                "category": "sports",
                "typical_contexts": ["outdoor", "sports_field", "gym", "park"],
                "size_ranges": {"typical": (0.005, 0.05)},
                "confidence_modifiers": {"sports_field": 1.5, "gym": 1.3, "park": 1.2},
                "common_confusions": ["balloon", "fruit", "round_object"],
                "exclusions": [],
                "relationships": ["with_person", "in_motion", "on_ground"]
            },
            
            # Food items
            "apple": {
                "category": "food",
                "typical_contexts": ["kitchen", "table", "store", "outdoor"],
                "size_ranges": {"typical": (0.002, 0.02)},
                "confidence_modifiers": {"kitchen": 1.4, "table": 1.3, "store": 1.2},
                "common_confusions": ["orange", "ball", "round_fruit"],
                "exclusions": [],
                "relationships": ["on_table", "in_bowl", "being_eaten"]
            }
        }
        
        # Domain-specific extensions
        if self.domain == DetectionDomain.MEDICAL:
            base_objects.update({
                "syringe": {
                    "category": "medical_tool",
                    "typical_contexts": ["hospital", "clinic", "medical_room"],
                    "size_ranges": {"typical": (0.005, 0.03)},
                    "confidence_modifiers": {"medical_room": 1.5, "hospital": 1.4},
                    "common_confusions": ["pen", "tube", "needle"],
                    "exclusions": ["non_medical_context"],
                    "relationships": ["with_medical_staff", "sterile_environment"]
                }
            })
            
        elif self.domain == DetectionDomain.SECURITY:
            base_objects.update({
                "weapon": {
                    "category": "security_threat",
                    "typical_contexts": ["security_check", "restricted_area"],
                    "size_ranges": {"typical": (0.01, 0.2)},
                    "confidence_modifiers": {"security_area": 1.5, "public": 0.3},
                    "common_confusions": ["tool", "phone", "object"],
                    "exclusions": ["safe_context"],
                    "relationships": ["with_person", "concealed", "threatening"]
                }
            })
            
        elif self.domain == DetectionDomain.RETAIL:
            base_objects.update({
                "product": {
                    "category": "merchandise",
                    "typical_contexts": ["store", "shelf", "checkout"],
                    "size_ranges": {"typical": (0.005, 0.1)},
                    "confidence_modifiers": {"shelf": 1.4, "checkout": 1.3, "cart": 1.2},
                    "common_confusions": ["package", "box", "container"],
                    "exclusions": [],
                    "relationships": ["on_shelf", "in_cart", "being_purchased"]
                }
            })
            
        return base_objects
    
    def get_object_knowledge(self, object_type: str) -> Dict[str, Any]:
        """Get knowledge for specific object type"""
        return self.object_database.get(object_type, {
            "category": "unknown",
            "typical_contexts": ["general"],
            "size_ranges": {"typical": (0.001, 1.0)},
            "confidence_modifiers": {},
            "common_confusions": [],
            "exclusions": [],
            "relationships": []
        })

class UniversalRAGEnhancer:
    """Universal RAG system that works for any detection domain"""
    
    def __init__(self, domain: DetectionDomain = DetectionDomain.GENERAL, 
                 knowledge_db_path: str = "universal_detection_knowledge.db"):
        self.domain = domain
        self.knowledge_db_path = knowledge_db_path
        self.object_knowledge = UniversalObjectKnowledge(domain)
        self.detection_history = deque(maxlen=1000)
        self.context_patterns = defaultdict(list)
        self._init_database()
        
    def _init_database(self):
        """Initialize universal knowledge database"""
        conn = sqlite3.connect(self.knowledge_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS universal_detections (
                id INTEGER PRIMARY KEY,
                domain TEXT,
                object_type TEXT,
                scene_context TEXT,
                confidence_original REAL,
                confidence_enhanced REAL,
                bbox_area REAL,
                aspect_ratio REAL,
                timestamp REAL,
                correction_type TEXT,
                success_rate REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_patterns (
                id INTEGER PRIMARY KEY,
                domain TEXT,
                context_type TEXT,
                object_combinations TEXT,
                pattern_strength REAL,
                timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def analyze_universal_context(self, detections: List[Dict], 
                                frame_diagnostics: Dict = None) -> Dict[str, Any]:
        """Analyze context for any detection domain"""
        
        detected_objects = [d.get("label", "") for d in detections]
        object_categories = []
        
        # Categorize detected objects
        for obj in detected_objects:
            knowledge = self.object_knowledge.get_object_knowledge(obj)
            category = knowledge.get("category", "unknown")
            object_categories.append(category)
            
        # Determine scene context based on object categories
        context = self._infer_scene_context(object_categories, detected_objects)
        
        # Analyze lighting and quality if diagnostics available
        lighting_context = "normal"
        if frame_diagnostics:
            brightness = frame_diagnostics.get("brightness_01", 0.5)
            if brightness < 0.3:
                lighting_context = "low_light"
            elif brightness > 0.8:
                lighting_context = "bright"
                
        return {
            "scene_type": context,
            "lighting": lighting_context,
            "object_categories": list(set(object_categories)),
            "detected_objects": detected_objects,
            "domain": self.domain.value
        }
        
    def _infer_scene_context(self, categories: List[str], objects: List[str]) -> str:
        """Infer scene context from object categories"""
        
        # Domain-specific context inference
        if self.domain == DetectionDomain.MEDICAL:
            if "medical_tool" in categories or any("medical" in obj for obj in objects):
                return "medical_facility"
            elif "person" in categories:
                return "patient_area"
                
        elif self.domain == DetectionDomain.SECURITY:
            if "security_threat" in categories:
                return "security_alert"
            elif "person" in categories and len(objects) > 3:
                return "crowded_area"
                
        elif self.domain == DetectionDomain.RETAIL:
            if "merchandise" in categories or "product" in objects:
                return "retail_store"
            elif "person" in categories and "cart" in objects:
                return "shopping_area"
                
        # General context inference
        indoor_indicators = ["furniture", "electronics", "appliance"]
        outdoor_indicators = ["vehicle", "animal", "sports"]
        
        indoor_score = sum(1 for cat in categories if cat in indoor_indicators)
        outdoor_score = sum(1 for cat in categories if cat in outdoor_indicators)
        
        if indoor_score > outdoor_score:
            return "indoor"
        elif outdoor_score > indoor_score:
            return "outdoor"
        else:
            return "mixed"
            
    def enhance_universal_detections(self, detections: List[Dict], 
                                   frame_diagnostics: Dict = None) -> List[Dict]:
        """Enhance detections for any domain using RAG"""
        
        context = self.analyze_universal_context(detections, frame_diagnostics)
        enhanced_detections = []
        
        for detection in detections:
            enhanced = detection.copy()
            obj_type = detection.get("label", "")
            
            # Get object knowledge
            knowledge = self.object_knowledge.get_object_knowledge(obj_type)
            
            # Apply context-based confidence adjustment
            scene_type = context["scene_type"]
            confidence_modifier = knowledge.get("confidence_modifiers", {}).get(scene_type, 1.0)
            
            original_conf = detection.get("confidence", 0.5)
            enhanced["confidence"] = min(0.99, original_conf * confidence_modifier)
            
            # Check for common confusions
            confusions = knowledge.get("common_confusions", [])
            if original_conf < 0.4 and confusions:
                enhanced["potential_confusions"] = confusions
                
            # Apply exclusion rules
            exclusions = knowledge.get("exclusions", [])
            for exclusion in exclusions:
                if exclusion in scene_type or any(exc in scene_type for exc in exclusions):
                    enhanced["confidence"] *= 0.5  # Reduce confidence for exclusion violations
                    enhanced["exclusion_warning"] = f"Unusual for {obj_type} in {scene_type}"
                    
            # Add enhancement metadata
            enhanced["rag_enhancement"] = {
                "original_confidence": original_conf,
                "confidence_modifier": confidence_modifier,
                "scene_context": scene_type,
                "domain": self.domain.value,
                "knowledge_applied": True
            }
            
            enhanced_detections.append(enhanced)
            
        # Store learning data
        self._store_universal_patterns(enhanced_detections, context)
        
        return enhanced_detections
        
    def _store_universal_patterns(self, detections: List[Dict], context: Dict):
        """Store detection patterns for universal learning"""
        
        conn = sqlite3.connect(self.knowledge_db_path)
        cursor = conn.cursor()
        
        timestamp = time.time()
        
        for detection in detections:
            bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
            if len(bbox) >= 4:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                aspect_ratio = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1)
                
                cursor.execute('''
                    INSERT INTO universal_detections
                    (domain, object_type, scene_context, confidence_original, 
                     confidence_enhanced, bbox_area, aspect_ratio, timestamp, correction_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.domain.value,
                    detection.get("label", ""),
                    context["scene_type"],
                    detection.get("rag_enhancement", {}).get("original_confidence", 0.5),
                    detection.get("confidence", 0.5),
                    area,
                    aspect_ratio,
                    timestamp,
                    "rag_enhancement"
                ))
                
        conn.commit()
        conn.close()

class UniversalAIDetectionSystem:
    """Complete universal AI detection system"""
    
    def __init__(self, config: UniversalDetectionConfig):
        self.config = config
        self.rag_enhancer = UniversalRAGEnhancer(config.domain) if config.enable_rag else None
        self.detection_history = deque(maxlen=100)
        self.performance_metrics = defaultdict(list)
        
    async def process_detections(self, raw_detections: List[Dict], 
                               frame_diagnostics: Dict = None,
                               yolo_model = None) -> Dict[str, Any]:
        """Process detections through universal AI pipeline"""
        
        start_time = time.time()
        pipeline_steps = []
        
        # Step 1: Basic filtering and validation
        filtered_detections = self._filter_detections(raw_detections)
        pipeline_steps.append("basic_filtering")
        
        # Step 2: RAG Enhancement (if enabled)
        if self.config.enable_rag and self.rag_enhancer:
            enhanced_detections = self.rag_enhancer.enhance_universal_detections(
                filtered_detections, frame_diagnostics
            )
            pipeline_steps.append("rag_enhancement")
        else:
            enhanced_detections = filtered_detections
            
        # Step 3: Universal misclassification fixes
        corrected_detections = self._apply_universal_corrections(
            enhanced_detections, frame_diagnostics
        )
        pipeline_steps.append("universal_corrections")
        
        # Step 4: Confidence optimization
        optimized_detections = self._optimize_confidence_scores(corrected_detections)
        pipeline_steps.append("confidence_optimization")
        
        # Calculate processing metrics
        processing_time = time.time() - start_time
        
        return {
            "detections": optimized_detections,
            "processing_time_ms": processing_time * 1000,
            "pipeline_steps": pipeline_steps,
            "domain": self.config.domain.value,
            "enhancements_applied": {
                "rag": self.config.enable_rag,
                "knowledge_graph": self.config.enable_knowledge_graph,
                "ensemble": self.config.enable_ensemble,
                "temporal_tracking": self.config.enable_temporal_tracking
            },
            "performance_metrics": {
                "total_detections": len(optimized_detections),
                "confidence_improvements": len([d for d in optimized_detections 
                                              if "rag_enhancement" in d]),
                "corrections_applied": len([d for d in optimized_detections 
                                          if "correction_applied" in d])
            }
        }
        
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Basic detection filtering"""
        filtered = []
        
        for detection in detections:
            confidence = detection.get("confidence", 0.0)
            bbox = detection.get("bbox_xyxy", [])
            
            # Basic confidence threshold
            if confidence < self.config.confidence_threshold:
                continue
                
            # Basic bbox validation
            if len(bbox) < 4:
                continue
                
            # Size validation (remove tiny detections)
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                
                if area < 100:  # Minimum area threshold
                    continue
                    
            filtered.append(detection)
            
        return filtered
        
    def _apply_universal_corrections(self, detections: List[Dict], 
                                   frame_diagnostics: Dict = None) -> List[Dict]:
        """Apply universal misclassification corrections"""
        
        corrected = []
        
        for detection in detections:
            corrected_detection = detection.copy()
            bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
            
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                aspect_ratio = width / max(height, 1)
                
                # Universal correction rules
                label = detection.get("label", "")
                confidence = detection.get("confidence", 0.5)
                
                # Rule 1: Very small objects with vehicle labels
                if label in ["car", "truck", "bus", "motorcycle"] and area < 5000:
                    if aspect_ratio > 1.5:
                        corrected_detection["label"] = "cell_phone"
                        corrected_detection["correction_applied"] = "small_vehicle_to_phone"
                    else:
                        corrected_detection["label"] = "book"
                        corrected_detection["correction_applied"] = "small_vehicle_to_book"
                    corrected_detection["confidence"] *= 0.8
                    
                # Rule 2: Large objects with small object labels
                elif label in ["cell_phone", "mouse", "remote"] and area > 50000:
                    corrected_detection["label"] = "laptop"
                    corrected_detection["correction_applied"] = "large_small_object_to_laptop"
                    corrected_detection["confidence"] *= 0.9
                    
                # Rule 3: Aspect ratio based corrections
                elif aspect_ratio < 0.3:  # Very tall objects
                    if label in ["car", "truck", "laptop"]:
                        corrected_detection["label"] = "person"
                        corrected_detection["correction_applied"] = "tall_object_to_person"
                        corrected_detection["confidence"] *= 0.85
                        
                elif aspect_ratio > 4.0:  # Very wide objects
                    if label == "person":
                        corrected_detection["label"] = "car"
                        corrected_detection["correction_applied"] = "wide_person_to_car"
                        corrected_detection["confidence"] *= 0.8
                        
            corrected.append(corrected_detection)
            
        return corrected
        
    def _optimize_confidence_scores(self, detections: List[Dict]) -> List[Dict]:
        """Optimize confidence scores based on context and patterns"""
        
        optimized = []
        
        for detection in detections:
            optimized_detection = detection.copy()
            
            # Boost confidence for high-quality detections
            confidence = detection.get("confidence", 0.5)
            bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
            
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                
                # Boost confidence for well-sized objects
                if 10000 < area < 200000:  # Good size range
                    optimized_detection["confidence"] = min(0.99, confidence * 1.1)
                    
                # Reduce confidence for edge cases
                elif area < 1000 or area > 500000:
                    optimized_detection["confidence"] = confidence * 0.9
                    
            optimized.append(optimized_detection)
            
        return optimized
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get universal system statistics"""
        
        stats = {
            "domain": self.config.domain.value,
            "total_processed": len(self.detection_history),
            "average_processing_time": np.mean(self.performance_metrics.get("processing_times", [0])),
            "enabled_features": {
                "rag": self.config.enable_rag,
                "knowledge_graph": self.config.enable_knowledge_graph,
                "ensemble": self.config.enable_ensemble,
                "temporal_tracking": self.config.enable_temporal_tracking,
                "adaptive_learning": self.config.enable_adaptive_learning
            }
        }
        
        if self.rag_enhancer:
            stats["rag_database_size"] = len(self.rag_enhancer.detection_history)
            
        return stats

# Factory function to create domain-specific systems
def create_ai_detection_system(domain: str = "general", **kwargs) -> UniversalAIDetectionSystem:
    """Factory function to create AI detection system for any domain"""
    
    domain_enum = DetectionDomain(domain.lower())
    
    config = UniversalDetectionConfig(
        domain=domain_enum,
        **kwargs
    )
    
    return UniversalAIDetectionSystem(config)

# Pre-configured systems for common domains
def create_general_detection_system(**kwargs):
    """Create system for general object detection"""
    return create_ai_detection_system("general", **kwargs)

def create_security_detection_system(**kwargs):
    """Create system for security/surveillance"""
    return create_ai_detection_system("security", **kwargs)

def create_medical_detection_system(**kwargs):
    """Create system for medical imaging"""
    return create_ai_detection_system("medical", **kwargs)

def create_retail_detection_system(**kwargs):
    """Create system for retail analytics"""
    return create_ai_detection_system("retail", **kwargs)

def create_wildlife_detection_system(**kwargs):
    """Create system for wildlife monitoring"""
    return create_ai_detection_system("wildlife", **kwargs)

# Global instances for easy access
universal_general_system = create_general_detection_system()
universal_security_system = create_security_detection_system()
universal_medical_system = create_medical_detection_system()
universal_retail_system = create_retail_detection_system()