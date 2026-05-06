"""
RAG-Enhanced Detection System for Car Vision
Implements Retrieval-Augmented Generation for improved object detection accuracy
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import sqlite3
from pathlib import Path

@dataclass
class DetectionContext:
    """Context information for RAG-enhanced detection"""
    scene_type: str  # indoor, outdoor, highway, city, parking
    lighting: str    # daylight, dusk, night, artificial
    weather: str     # clear, rain, fog, snow
    time_of_day: str # morning, afternoon, evening, night
    location_type: str # residential, commercial, highway, parking_lot
    
@dataclass
class ObjectKnowledge:
    """Knowledge base entry for object detection"""
    object_type: str
    typical_sizes: Dict[str, Tuple[float, float]]  # context -> (min_size, max_size)
    typical_locations: List[str]  # where this object is commonly found
    context_rules: Dict[str, float]  # context -> confidence_modifier
    confusion_matrix: Dict[str, float]  # commonly_confused_with -> probability
    temporal_patterns: Dict[str, List[str]]  # time -> likely_states

class RAGDetectionEnhancer:
    """RAG-enhanced detection system for improved accuracy"""
    
    def __init__(self, knowledge_db_path: str = "detection_knowledge.db"):
        self.knowledge_db_path = knowledge_db_path
        self.detection_history = deque(maxlen=100)  # Last 100 detections
        self.scene_context_history = deque(maxlen=50)  # Scene context history
        self.object_knowledge = self._load_object_knowledge()
        self.temporal_memory = defaultdict(list)  # Track object patterns over time
        self._init_knowledge_db()
        
    def _init_knowledge_db(self):
        """Initialize SQLite knowledge database"""
        conn = sqlite3.connect(self.knowledge_db_path)
        cursor = conn.cursor()
        
        # Create tables for knowledge storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_patterns (
                id INTEGER PRIMARY KEY,
                object_type TEXT,
                scene_context TEXT,
                confidence_original REAL,
                confidence_corrected REAL,
                bbox_size REAL,
                aspect_ratio REAL,
                position_x REAL,
                position_y REAL,
                timestamp REAL,
                correction_applied TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scene_contexts (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                scene_type TEXT,
                lighting TEXT,
                brightness REAL,
                contrast REAL,
                glare_present BOOLEAN,
                object_count INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS object_relationships (
                id INTEGER PRIMARY KEY,
                primary_object TEXT,
                related_object TEXT,
                relationship_type TEXT,
                co_occurrence_score REAL,
                spatial_relationship TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_object_knowledge(self) -> Dict[str, ObjectKnowledge]:
        """Load comprehensive object knowledge base"""
        knowledge = {
            "person": ObjectKnowledge(
                object_type="person",
                typical_sizes={
                    "close_up": (0.1, 0.8),      # Face/upper body shots
                    "full_body": (0.05, 0.3),    # Full person in frame
                    "distant": (0.001, 0.05),    # Person far away
                    "indoor": (0.08, 0.6),       # Person indoors
                    "outdoor": (0.01, 0.4)       # Person outdoors
                },
                typical_locations=["sidewalk", "crosswalk", "indoor", "parking_lot", "residential"],
                context_rules={
                    "indoor": 1.2,     # More likely indoors
                    "sidewalk": 1.3,   # Very likely on sidewalks
                    "highway": 0.3,    # Less likely on highways
                    "night": 0.8       # Slightly less visible at night
                },
                confusion_matrix={
                    "truck": 0.15,     # Sometimes confused with truck when close
                    "car": 0.08,       # Sometimes confused with car
                    "motorcycle": 0.05  # Rarely confused with motorcycle
                },
                temporal_patterns={
                    "morning": ["walking", "commuting"],
                    "evening": ["walking", "leisure"],
                    "night": ["rare", "suspicious"]
                }
            ),
            
            "car": ObjectKnowledge(
                object_type="car",
                typical_sizes={
                    "close": (0.2, 0.9),
                    "medium": (0.05, 0.3),
                    "distant": (0.001, 0.05),
                    "highway": (0.01, 0.15),
                    "parking": (0.1, 0.5)
                },
                typical_locations=["road", "highway", "parking_lot", "driveway"],
                context_rules={
                    "indoor": 0.1,     # Very unlikely indoors
                    "road": 1.4,       # Very likely on roads
                    "parking": 1.3,    # Likely in parking areas
                    "sidewalk": 0.2    # Unlikely on sidewalks
                },
                confusion_matrix={
                    "truck": 0.12,
                    "bus": 0.08,
                    "person": 0.03
                },
                temporal_patterns={
                    "morning": ["commuting", "moving"],
                    "evening": ["commuting", "moving"],
                    "night": ["parked", "moving_slow"]
                }
            ),
            
            "truck": ObjectKnowledge(
                object_type="truck",
                typical_sizes={
                    "close": (0.3, 0.95),
                    "medium": (0.08, 0.4),
                    "distant": (0.002, 0.08),
                    "highway": (0.02, 0.2)
                },
                typical_locations=["highway", "road", "loading_dock", "industrial"],
                context_rules={
                    "indoor": 0.05,    # Almost never indoors
                    "highway": 1.5,    # Very common on highways
                    "residential": 0.6, # Less common in residential
                    "industrial": 1.4   # Common in industrial areas
                },
                confusion_matrix={
                    "car": 0.18,
                    "bus": 0.25,
                    "person": 0.12     # Sometimes person misclassified as truck
                },
                temporal_patterns={
                    "morning": ["delivery", "commuting"],
                    "afternoon": ["delivery", "transport"],
                    "night": ["long_haul", "parked"]
                }
            ),
            
            "motorcycle": ObjectKnowledge(
                object_type="motorcycle",
                typical_sizes={
                    "close": (0.1, 0.6),
                    "medium": (0.02, 0.2),
                    "distant": (0.001, 0.03)
                },
                typical_locations=["road", "highway", "parking_lot"],
                context_rules={
                    "indoor": 0.15,    # Rarely indoors
                    "highway": 1.2,    # Common on highways
                    "parking": 1.1,    # Common in parking
                    "rain": 0.3        # Less likely in rain
                },
                confusion_matrix={
                    "bicycle": 0.2,
                    "person": 0.15,
                    "car": 0.1
                },
                temporal_patterns={
                    "morning": ["commuting"],
                    "evening": ["leisure", "commuting"],
                    "winter": ["rare"]
                }
            )
        }
        return knowledge
        
    def analyze_scene_context(self, frame_diagnostics: Dict, detections: List[Dict]) -> DetectionContext:
        """Analyze current scene context using RAG principles"""
        
        # Determine scene type based on detected objects and their patterns
        indoor_indicators = ["laptop", "keyboard", "mouse", "tv", "couch", "bed"]
        outdoor_indicators = ["car", "truck", "traffic_light", "stop_sign", "tree"]
        highway_indicators = ["multiple_cars", "high_speed", "lane_markings"]
        
        detected_labels = [d.get("label", "") for d in detections]
        
        # Scene type classification
        indoor_score = sum(1 for label in detected_labels if label in indoor_indicators)
        outdoor_score = sum(1 for label in detected_labels if label in outdoor_indicators)
        
        if indoor_score > outdoor_score:
            scene_type = "indoor"
        elif len([d for d in detections if d.get("label") == "car"]) > 3:
            scene_type = "highway"
        else:
            scene_type = "outdoor"
            
        # Lighting analysis
        brightness = frame_diagnostics.get("brightness_01", 0.5)
        if brightness < 0.2:
            lighting = "night"
        elif brightness < 0.4:
            lighting = "dusk"
        else:
            lighting = "daylight"
            
        # Weather inference (basic)
        low_contrast = frame_diagnostics.get("low_contrast", False)
        glare = frame_diagnostics.get("glare_risk", False)
        
        if low_contrast and not glare:
            weather = "fog"
        elif glare:
            weather = "clear"
        else:
            weather = "clear"
            
        # Time of day
        current_hour = time.localtime().tm_hour
        if 6 <= current_hour < 12:
            time_of_day = "morning"
        elif 12 <= current_hour < 18:
            time_of_day = "afternoon"
        elif 18 <= current_hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"
            
        context = DetectionContext(
            scene_type=scene_type,
            lighting=lighting,
            weather=weather,
            time_of_day=time_of_day,
            location_type=scene_type
        )
        
        # Store context in history
        self.scene_context_history.append({
            "timestamp": time.time(),
            "context": context,
            "detections_count": len(detections)
        })
        
        return context
        
    def retrieve_relevant_knowledge(self, detection: Dict, context: DetectionContext) -> Dict[str, Any]:
        """Retrieve relevant knowledge for a detection using RAG principles"""
        
        object_type = detection.get("label", "")
        knowledge = self.object_knowledge.get(object_type)
        
        if not knowledge:
            return {"confidence_modifier": 1.0, "suggestions": []}
            
        # Calculate context-based confidence modifier
        confidence_modifier = 1.0
        suggestions = []
        
        # Apply context rules
        context_key = context.scene_type
        if context_key in knowledge.context_rules:
            confidence_modifier *= knowledge.context_rules[context_key]
            
        # Check size consistency
        bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
        if len(bbox) >= 4:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area_fraction = (width * height) / (1280 * 720)  # Assume standard resolution
            
            # Check if size matches expected range for context
            expected_sizes = knowledge.typical_sizes.get(context.scene_type, (0, 1))
            if not (expected_sizes[0] <= area_fraction <= expected_sizes[1]):
                confidence_modifier *= 0.7
                suggestions.append(f"Size unusual for {object_type} in {context.scene_type}")
                
        # Check temporal patterns
        current_time = context.time_of_day
        if current_time in knowledge.temporal_patterns:
            expected_states = knowledge.temporal_patterns[current_time]
            if "rare" in expected_states:
                confidence_modifier *= 0.6
                suggestions.append(f"{object_type} rare at {current_time}")
                
        # Check for common confusions
        original_conf = detection.get("confidence", 0.5)
        for confused_with, confusion_prob in knowledge.confusion_matrix.items():
            if original_conf < 0.4 and confusion_prob > 0.1:
                suggestions.append(f"Might be confused with {confused_with}")
                
        return {
            "confidence_modifier": confidence_modifier,
            "suggestions": suggestions,
            "expected_size_range": knowledge.typical_sizes.get(context.scene_type, (0, 1)),
            "context_appropriateness": confidence_modifier
        }
        
    def enhance_detections_with_rag(self, detections: List[Dict], frame_diagnostics: Dict) -> List[Dict]:
        """Main RAG enhancement function"""
        
        # Analyze current scene context
        context = self.analyze_scene_context(frame_diagnostics, detections)
        
        enhanced_detections = []
        corrections_applied = []
        
        for detection in detections:
            # Retrieve relevant knowledge
            knowledge_result = self.retrieve_relevant_knowledge(detection, context)
            
            # Apply RAG-based corrections
            enhanced_detection = detection.copy()
            original_conf = detection.get("confidence", 0.5)
            original_label = detection.get("label", "")
            
            # Apply confidence modifier
            new_confidence = original_conf * knowledge_result["confidence_modifier"]
            enhanced_detection["confidence"] = min(0.99, max(0.01, new_confidence))
            
            # Check for label corrections based on context
            corrected_label = self._suggest_label_correction(
                detection, context, knowledge_result
            )
            
            if corrected_label != original_label:
                enhanced_detection["label"] = corrected_label
                enhanced_detection["confidence"] *= 0.85  # Reduce confidence for corrections
                corrections_applied.append({
                    "original": original_label,
                    "corrected": corrected_label,
                    "reason": "RAG context analysis"
                })
                
            # Add RAG metadata
            enhanced_detection["rag_metadata"] = {
                "context_score": knowledge_result.get("context_appropriateness", 0.5),
                "suggestions": knowledge_result["suggestions"],
                "confidence_modifier": knowledge_result["confidence_modifier"]
            }
            
            enhanced_detections.append(enhanced_detection)
            
        # Store learning data
        self._store_detection_patterns(enhanced_detections, context, corrections_applied)
        
        return enhanced_detections
        
    def _suggest_label_correction(self, detection: Dict, context: DetectionContext, knowledge: Dict) -> str:
        """Suggest label corrections based on RAG analysis"""
        
        original_label = detection.get("label", "")
        bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
        
        if len(bbox) < 4:
            return original_label
            
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area_fraction = (width * height) / (1280 * 720)
        aspect_ratio = width / max(height, 1)
        
        # RAG-based correction rules
        
        # Rule 1: Large objects in indoor scenes
        if context.scene_type == "indoor" and area_fraction > 0.1:
            if original_label in ["car", "truck", "bus"]:
                return "person"  # Large vehicle indoors is likely a person
                
        # Rule 2: Very small "vehicles" are likely objects
        if original_label in ["car", "truck", "motorcycle"] and area_fraction < 0.005:
            if aspect_ratio > 1.5:
                return "cell phone"
            else:
                return "book"
                
        # Rule 3: Temporal context corrections
        if context.time_of_day == "night" and context.scene_type == "indoor":
            if original_label in ["car", "motorcycle"]:
                return "person"
                
        # Rule 4: Size-based person detection
        if original_label in ["truck", "bus"] and 0.1 < area_fraction < 0.4:
            if aspect_ratio < 1.2:  # Tall object
                return "person"
                
        return original_label
        
    def _store_detection_patterns(self, detections: List[Dict], context: DetectionContext, corrections: List[Dict]):
        """Store detection patterns for continuous learning"""
        
        conn = sqlite3.connect(self.knowledge_db_path)
        cursor = conn.cursor()
        
        timestamp = time.time()
        
        # Store scene context
        cursor.execute('''
            INSERT INTO scene_contexts 
            (timestamp, scene_type, lighting, brightness, contrast, glare_present, object_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            context.scene_type,
            context.lighting,
            0.5,  # Default brightness
            0.5,  # Default contrast
            False,  # Default glare
            len(detections)
        ))
        
        # Store detection patterns
        for detection in detections:
            bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area_fraction = (width * height) / (1280 * 720)
                aspect_ratio = width / max(height, 1)
                
                cursor.execute('''
                    INSERT INTO detection_patterns
                    (object_type, scene_context, confidence_original, confidence_corrected,
                     bbox_size, aspect_ratio, position_x, position_y, timestamp, correction_applied)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection.get("label", ""),
                    context.scene_type,
                    detection.get("original_confidence", detection.get("confidence", 0.5)),
                    detection.get("confidence", 0.5),
                    area_fraction,
                    aspect_ratio,
                    (bbox[0] + bbox[2]) / 2,  # Center X
                    (bbox[1] + bbox[3]) / 2,  # Center Y
                    timestamp,
                    json.dumps(corrections)
                ))
                
        conn.commit()
        conn.close()
        
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from stored learning data"""
        
        conn = sqlite3.connect(self.knowledge_db_path)
        cursor = conn.cursor()
        
        # Get most common corrections
        cursor.execute('''
            SELECT object_type, scene_context, COUNT(*) as correction_count
            FROM detection_patterns 
            WHERE correction_applied != '[]'
            GROUP BY object_type, scene_context
            ORDER BY correction_count DESC
            LIMIT 10
        ''')
        
        common_corrections = cursor.fetchall()
        
        # Get accuracy trends
        cursor.execute('''
            SELECT scene_context, AVG(confidence_corrected - confidence_original) as avg_improvement
            FROM detection_patterns
            GROUP BY scene_context
        ''')
        
        accuracy_trends = cursor.fetchall()
        
        conn.close()
        
        return {
            "common_corrections": common_corrections,
            "accuracy_trends": accuracy_trends,
            "total_patterns": len(self.detection_history)
        }

# Global RAG enhancer instance
rag_enhancer = RAGDetectionEnhancer()