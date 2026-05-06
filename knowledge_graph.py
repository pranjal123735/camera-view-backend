"""
Knowledge Graph System for Object Detection
Implements semantic relationships and contextual reasoning for improved detection accuracy
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from enum import Enum
import time

class RelationType(Enum):
    """Types of relationships between objects"""
    SPATIAL_NEAR = "spatial_near"
    SPATIAL_INSIDE = "spatial_inside"
    SPATIAL_ON = "spatial_on"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    SEMANTIC_SIMILAR = "semantic_similar"
    FUNCTIONAL_RELATED = "functional_related"
    CAUSAL = "causal"
    EXCLUSION = "exclusion"  # Objects that rarely appear together

@dataclass
class ObjectNode:
    """Node in the knowledge graph representing an object type"""
    object_type: str
    typical_contexts: List[str] = field(default_factory=list)
    size_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    confidence_modifiers: Dict[str, float] = field(default_factory=dict)
    semantic_features: List[str] = field(default_factory=list)
    exclusion_rules: List[str] = field(default_factory=list)

@dataclass
class Relationship:
    """Relationship between two objects in the knowledge graph"""
    source: str
    target: str
    relation_type: RelationType
    strength: float  # 0.0 to 1.0
    context_conditions: List[str] = field(default_factory=list)
    spatial_constraints: Optional[Dict[str, Any]] = None

class SemanticKnowledgeGraph:
    """Knowledge graph for semantic object detection reasoning"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.object_nodes = {}
        self.relationships = []
        self.context_rules = {}
        self.exclusion_matrix = defaultdict(set)
        self._build_knowledge_graph()
        
    def _build_knowledge_graph(self):
        """Build the comprehensive knowledge graph"""
        
        # Define object nodes with rich semantic information
        objects = {
            "person": ObjectNode(
                object_type="person",
                typical_contexts=["indoor", "outdoor", "sidewalk", "crosswalk", "office"],
                size_ranges={
                    "close_up": (0.1, 0.8),
                    "full_body": (0.05, 0.4),
                    "distant": (0.001, 0.05)
                },
                confidence_modifiers={
                    "indoor": 1.2,
                    "sidewalk": 1.3,
                    "highway": 0.3,
                    "with_laptop": 1.4,
                    "with_phone": 1.2
                },
                semantic_features=["bipedal", "vertical", "mobile", "intelligent"],
                exclusion_rules=["rarely_with_large_vehicles_indoors"]
            ),
            
            "car": ObjectNode(
                object_type="car",
                typical_contexts=["road", "highway", "parking_lot", "driveway", "outdoor"],
                size_ranges={
                    "close": (0.2, 0.9),
                    "medium": (0.05, 0.3),
                    "distant": (0.001, 0.05)
                },
                confidence_modifiers={
                    "road": 1.4,
                    "parking": 1.3,
                    "indoor": 0.1,
                    "with_person_inside": 1.2
                },
                semantic_features=["wheeled", "motorized", "transport", "rectangular"],
                exclusion_rules=["not_indoors_unless_garage"]
            ),
            
            "truck": ObjectNode(
                object_type="truck",
                typical_contexts=["highway", "road", "industrial", "loading_dock"],
                size_ranges={
                    "close": (0.3, 0.95),
                    "medium": (0.08, 0.4),
                    "distant": (0.002, 0.08)
                },
                confidence_modifiers={
                    "highway": 1.5,
                    "industrial": 1.4,
                    "residential": 0.6,
                    "indoor": 0.05
                },
                semantic_features=["large", "wheeled", "motorized", "commercial"],
                exclusion_rules=["almost_never_indoors", "not_with_office_items"]
            ),
            
            "motorcycle": ObjectNode(
                object_type="motorcycle",
                typical_contexts=["road", "highway", "parking_lot"],
                size_ranges={
                    "close": (0.1, 0.6),
                    "medium": (0.02, 0.2),
                    "distant": (0.001, 0.03)
                },
                confidence_modifiers={
                    "highway": 1.2,
                    "parking": 1.1,
                    "indoor": 0.15,
                    "rain": 0.3
                },
                semantic_features=["two_wheeled", "motorized", "fast", "exposed_rider"],
                exclusion_rules=["rare_in_rain", "not_indoors"]
            ),
            
            "bicycle": ObjectNode(
                object_type="bicycle",
                typical_contexts=["road", "sidewalk", "park", "bike_lane"],
                size_ranges={
                    "close": (0.08, 0.5),
                    "medium": (0.02, 0.15),
                    "distant": (0.001, 0.02)
                },
                confidence_modifiers={
                    "bike_lane": 1.4,
                    "park": 1.3,
                    "sidewalk": 1.2,
                    "highway": 0.4
                },
                semantic_features=["two_wheeled", "human_powered", "eco_friendly"],
                exclusion_rules=["rare_on_highways"]
            ),
            
            "laptop": ObjectNode(
                object_type="laptop",
                typical_contexts=["indoor", "office", "home", "cafe"],
                size_ranges={
                    "typical": (0.01, 0.1)
                },
                confidence_modifiers={
                    "indoor": 1.4,
                    "office": 1.5,
                    "outdoor": 0.2
                },
                semantic_features=["electronic", "rectangular", "portable"],
                exclusion_rules=["not_outdoors_unless_portable_use"]
            ),
            
            "cell_phone": ObjectNode(
                object_type="cell_phone",
                typical_contexts=["anywhere"],
                size_ranges={
                    "typical": (0.001, 0.02)
                },
                confidence_modifiers={
                    "with_person": 1.3,
                    "indoor": 1.1
                },
                semantic_features=["small", "rectangular", "handheld"],
                exclusion_rules=[]
            )
        }
        
        # Add nodes to graph
        for obj_type, node in objects.items():
            self.object_nodes[obj_type] = node
            self.graph.add_node(obj_type, **node.__dict__)
            
        # Define relationships
        relationships = [
            # Spatial relationships
            Relationship("person", "laptop", RelationType.SPATIAL_NEAR, 0.8, 
                        ["indoor", "office"], {"max_distance": 2.0}),
            Relationship("person", "cell_phone", RelationType.SPATIAL_NEAR, 0.9, 
                        ["anywhere"], {"max_distance": 1.0}),
            Relationship("person", "car", RelationType.SPATIAL_INSIDE, 0.7, 
                        ["driving"], {"overlap_threshold": 0.6}),
            Relationship("person", "bicycle", RelationType.SPATIAL_ON, 0.8, 
                        ["riding"], {"overlap_threshold": 0.4}),
            
            # Exclusion relationships
            Relationship("car", "laptop", RelationType.EXCLUSION, 0.9, 
                        ["indoor"], None),
            Relationship("truck", "laptop", RelationType.EXCLUSION, 0.95, 
                        ["indoor"], None),
            Relationship("motorcycle", "laptop", RelationType.EXCLUSION, 0.9, 
                        ["indoor"], None),
            
            # Semantic similarities
            Relationship("car", "truck", RelationType.SEMANTIC_SIMILAR, 0.7, 
                        ["road_context"], None),
            Relationship("motorcycle", "bicycle", RelationType.SEMANTIC_SIMILAR, 0.6, 
                        ["two_wheeled"], None),
            
            # Functional relationships
            Relationship("person", "car", RelationType.FUNCTIONAL_RELATED, 0.8, 
                        ["transportation"], None),
            Relationship("person", "motorcycle", RelationType.FUNCTIONAL_RELATED, 0.8, 
                        ["transportation"], None),
            Relationship("person", "bicycle", RelationType.FUNCTIONAL_RELATED, 0.9, 
                        ["transportation"], None),
        ]
        
        # Add relationships to graph
        for rel in relationships:
            self.relationships.append(rel)
            self.graph.add_edge(rel.source, rel.target, 
                              relation_type=rel.relation_type,
                              strength=rel.strength,
                              context_conditions=rel.context_conditions,
                              spatial_constraints=rel.spatial_constraints)
            
            # Build exclusion matrix
            if rel.relation_type == RelationType.EXCLUSION:
                self.exclusion_matrix[rel.source].add(rel.target)
                self.exclusion_matrix[rel.target].add(rel.source)
                
    def analyze_detection_context(self, detections: List[Dict], scene_context: str) -> Dict[str, Any]:
        """Analyze detection context using knowledge graph"""
        
        analysis = {
            "context_consistency": {},
            "relationship_violations": [],
            "confidence_adjustments": {},
            "suggested_corrections": []
        }
        
        detected_objects = [d.get("label", "") for d in detections]
        
        # Check context consistency for each detection
        for detection in detections:
            obj_type = detection.get("label", "")
            if obj_type in self.object_nodes:
                node = self.object_nodes[obj_type]
                
                # Check if object fits the scene context
                context_score = self._calculate_context_score(obj_type, scene_context, detected_objects)
                analysis["context_consistency"][obj_type] = context_score
                
                # Calculate confidence adjustment
                confidence_adj = node.confidence_modifiers.get(scene_context, 1.0)
                analysis["confidence_adjustments"][obj_type] = confidence_adj
                
        # Check for relationship violations
        violations = self._check_relationship_violations(detections, scene_context)
        analysis["relationship_violations"] = violations
        
        # Generate correction suggestions
        suggestions = self._generate_correction_suggestions(detections, scene_context, violations)
        analysis["suggested_corrections"] = suggestions
        
        return analysis
        
    def _calculate_context_score(self, obj_type: str, scene_context: str, co_occurring_objects: List[str]) -> float:
        """Calculate how well an object fits the current context"""
        
        if obj_type not in self.object_nodes:
            return 0.5  # Neutral score for unknown objects
            
        node = self.object_nodes[obj_type]
        score = 0.5  # Base score
        
        # Check if object typically appears in this context
        if scene_context in node.typical_contexts:
            score += 0.3
        elif any(ctx in scene_context for ctx in node.typical_contexts):
            score += 0.1
            
        # Check relationships with co-occurring objects
        for co_obj in co_occurring_objects:
            if co_obj != obj_type and self.graph.has_edge(obj_type, co_obj):
                edge_data = self.graph[obj_type][co_obj]
                rel_type = edge_data.get("relation_type")
                strength = edge_data.get("strength", 0.5)
                
                if rel_type == RelationType.EXCLUSION:
                    score -= 0.2 * strength  # Penalize exclusion violations
                else:
                    score += 0.1 * strength  # Reward positive relationships
                    
        return max(0.0, min(1.0, score))
        
    def _check_relationship_violations(self, detections: List[Dict], scene_context: str) -> List[Dict]:
        """Check for violations of known object relationships"""
        
        violations = []
        detected_objects = [(d.get("label", ""), d) for d in detections]
        
        for i, (obj1, det1) in enumerate(detected_objects):
            for j, (obj2, det2) in enumerate(detected_objects):
                if i >= j or obj1 == obj2:
                    continue
                    
                # Check for exclusion violations
                if obj2 in self.exclusion_matrix.get(obj1, set()):
                    # Check if exclusion applies in current context
                    if self._exclusion_applies(obj1, obj2, scene_context):
                        violations.append({
                            "type": "exclusion_violation",
                            "object1": obj1,
                            "object2": obj2,
                            "context": scene_context,
                            "severity": "high",
                            "detection1": det1,
                            "detection2": det2
                        })
                        
                # Check spatial relationship violations
                spatial_violation = self._check_spatial_violation(det1, det2, obj1, obj2)
                if spatial_violation:
                    violations.append(spatial_violation)
                    
        return violations
        
    def _exclusion_applies(self, obj1: str, obj2: str, scene_context: str) -> bool:
        """Check if exclusion rule applies in current context"""
        
        if not self.graph.has_edge(obj1, obj2):
            return False
            
        edge_data = self.graph[obj1][obj2]
        if edge_data.get("relation_type") != RelationType.EXCLUSION:
            return False
            
        context_conditions = edge_data.get("context_conditions", [])
        if not context_conditions:
            return True  # Always applies
            
        return any(condition in scene_context for condition in context_conditions)
        
    def _check_spatial_violation(self, det1: Dict, det2: Dict, obj1: str, obj2: str) -> Optional[Dict]:
        """Check for spatial relationship violations"""
        
        if not self.graph.has_edge(obj1, obj2):
            return None
            
        edge_data = self.graph[obj1][obj2]
        spatial_constraints = edge_data.get("spatial_constraints")
        
        if not spatial_constraints:
            return None
            
        bbox1 = det1.get("bbox_xyxy", [0, 0, 100, 100])
        bbox2 = det2.get("bbox_xyxy", [0, 0, 100, 100])
        
        if len(bbox1) < 4 or len(bbox2) < 4:
            return None
            
        # Calculate spatial metrics
        distance = self._calculate_bbox_distance(bbox1, bbox2)
        overlap = self._calculate_bbox_overlap(bbox1, bbox2)
        
        # Check constraints
        max_distance = spatial_constraints.get("max_distance")
        overlap_threshold = spatial_constraints.get("overlap_threshold")
        
        violation = None
        
        if max_distance and distance > max_distance:
            violation = {
                "type": "spatial_distance_violation",
                "object1": obj1,
                "object2": obj2,
                "expected_max_distance": max_distance,
                "actual_distance": distance,
                "severity": "medium"
            }
        elif overlap_threshold and overlap < overlap_threshold:
            violation = {
                "type": "spatial_overlap_violation",
                "object1": obj1,
                "object2": obj2,
                "expected_min_overlap": overlap_threshold,
                "actual_overlap": overlap,
                "severity": "medium"
            }
            
        return violation
        
    def _calculate_bbox_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bounding boxes"""
        
        # Calculate centers
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        # Euclidean distance
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        
        return (dx * dx + dy * dy) ** 0.5
        
    def _calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Return intersection over smaller area
        smaller_area = min(area1, area2)
        return intersection / smaller_area if smaller_area > 0 else 0.0
        
    def _generate_correction_suggestions(self, detections: List[Dict], scene_context: str, violations: List[Dict]) -> List[Dict]:
        """Generate correction suggestions based on knowledge graph analysis"""
        
        suggestions = []
        
        for violation in violations:
            if violation["type"] == "exclusion_violation":
                # Suggest removing the less likely object
                obj1 = violation["object1"]
                obj2 = violation["object2"]
                det1 = violation["detection1"]
                det2 = violation["detection2"]
                
                conf1 = det1.get("confidence", 0.5)
                conf2 = det2.get("confidence", 0.5)
                
                # Remove the one with lower confidence or less context fit
                context_score1 = self._calculate_context_score(obj1, scene_context, [])
                context_score2 = self._calculate_context_score(obj2, scene_context, [])
                
                combined_score1 = conf1 * context_score1
                combined_score2 = conf2 * context_score2
                
                if combined_score1 < combined_score2:
                    suggestions.append({
                        "action": "remove_detection",
                        "target_object": obj1,
                        "reason": f"Exclusion violation with {obj2} in {scene_context}",
                        "confidence": 0.8
                    })
                else:
                    suggestions.append({
                        "action": "remove_detection",
                        "target_object": obj2,
                        "reason": f"Exclusion violation with {obj1} in {scene_context}",
                        "confidence": 0.8
                    })
                    
        # Suggest label corrections based on context
        for detection in detections:
            obj_type = detection.get("label", "")
            context_score = self._calculate_context_score(obj_type, scene_context, [])
            
            if context_score < 0.3:  # Low context fit
                # Find better alternatives
                alternatives = self._find_alternative_labels(detection, scene_context)
                if alternatives:
                    suggestions.append({
                        "action": "relabel_detection",
                        "current_label": obj_type,
                        "suggested_label": alternatives[0]["label"],
                        "reason": f"Better context fit for {scene_context}",
                        "confidence": alternatives[0]["confidence"]
                    })
                    
        return suggestions
        
    def _find_alternative_labels(self, detection: Dict, scene_context: str) -> List[Dict]:
        """Find alternative labels that better fit the context"""
        
        alternatives = []
        bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
        
        if len(bbox) < 4:
            return alternatives
            
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        aspect_ratio = width / max(height, 1)
        
        # Check each object type for better fit
        for obj_type, node in self.object_nodes.items():
            if obj_type == detection.get("label"):
                continue
                
            # Check context fit
            context_score = self._calculate_context_score(obj_type, scene_context, [])
            
            # Check size fit
            size_fit = self._check_size_fit(area, aspect_ratio, node, scene_context)
            
            combined_score = context_score * size_fit
            
            if combined_score > 0.6:  # Good alternative
                alternatives.append({
                    "label": obj_type,
                    "confidence": combined_score,
                    "context_score": context_score,
                    "size_fit": size_fit
                })
                
        # Sort by confidence
        alternatives.sort(key=lambda x: x["confidence"], reverse=True)
        
        return alternatives[:3]  # Return top 3 alternatives
        
    def _check_size_fit(self, area: float, aspect_ratio: float, node: ObjectNode, context: str) -> float:
        """Check how well the size fits the object type in given context"""
        
        # Normalize area (assuming 1280x720 frame)
        area_fraction = area / (1280 * 720)
        
        # Find appropriate size range
        size_range = None
        if context in node.size_ranges:
            size_range = node.size_ranges[context]
        elif "typical" in node.size_ranges:
            size_range = node.size_ranges["typical"]
        elif node.size_ranges:
            size_range = list(node.size_ranges.values())[0]
            
        if not size_range:
            return 0.5  # Neutral if no size info
            
        min_size, max_size = size_range
        
        if min_size <= area_fraction <= max_size:
            return 1.0  # Perfect fit
        elif area_fraction < min_size:
            # Too small
            ratio = area_fraction / min_size
            return max(0.0, ratio)
        else:
            # Too large
            ratio = max_size / area_fraction
            return max(0.0, ratio)

# Global knowledge graph instance
knowledge_graph = SemanticKnowledgeGraph()