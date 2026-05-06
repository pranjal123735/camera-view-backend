"""
Multi-Model Ensemble Detection System
Combines multiple YOLO models and detection strategies for improved accuracy
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from collections import defaultdict, deque
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class EnsembleResult:
    """Result from ensemble detection"""
    final_detections: List[Dict]
    model_agreements: Dict[str, float]
    confidence_scores: Dict[str, List[float]]
    processing_times: Dict[str, float]
    consensus_strength: float

class TemporalConsistencyTracker:
    """Tracks object consistency across frames for temporal smoothing"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.object_histories = defaultdict(lambda: deque(maxlen=max_history))
        self.confidence_histories = defaultdict(lambda: deque(maxlen=max_history))
        self.position_histories = defaultdict(lambda: deque(maxlen=max_history))
        
    def update_object_history(self, track_id: int, label: str, confidence: float, bbox: List[float]):
        """Update object history for temporal consistency"""
        self.object_histories[track_id].append(label)
        self.confidence_histories[track_id].append(confidence)
        self.position_histories[track_id].append(bbox)
        
    def get_temporal_consensus(self, track_id: int) -> Tuple[str, float]:
        """Get temporal consensus for object label and confidence"""
        if track_id not in self.object_histories:
            return "", 0.0
            
        labels = list(self.object_histories[track_id])
        confidences = list(self.confidence_histories[track_id])
        
        if not labels:
            return "", 0.0
            
        # Count label occurrences
        label_counts = defaultdict(int)
        label_conf_sums = defaultdict(float)
        
        for label, conf in zip(labels, confidences):
            label_counts[label] += 1
            label_conf_sums[label] += conf
            
        # Find most frequent label
        most_frequent = max(label_counts.keys(), key=lambda x: label_counts[x])
        avg_confidence = label_conf_sums[most_frequent] / label_counts[most_frequent]
        
        # Calculate stability score
        stability = label_counts[most_frequent] / len(labels)
        
        return most_frequent, avg_confidence * stability
        
    def get_position_prediction(self, track_id: int) -> Optional[List[float]]:
        """Predict next position based on movement history"""
        if track_id not in self.position_histories:
            return None
            
        positions = list(self.position_histories[track_id])
        if len(positions) < 2:
            return positions[-1] if positions else None
            
        # Simple linear prediction
        last_pos = positions[-1]
        prev_pos = positions[-2]
        
        # Calculate velocity
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        
        # Predict next position
        predicted = [
            last_pos[0] + dx,
            last_pos[1] + dy,
            last_pos[2] + dx,
            last_pos[3] + dy
        ]
        
        return predicted

class AdaptiveLearningSystem:
    """Adaptive learning system that improves detection over time"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.error_patterns = defaultdict(list)
        self.success_patterns = defaultdict(list)
        self.adaptation_rules = {}
        
    def record_detection_result(self, detection: Dict, ground_truth: Optional[Dict] = None):
        """Record detection result for learning"""
        timestamp = time.time()
        
        result = {
            "timestamp": timestamp,
            "detection": detection,
            "ground_truth": ground_truth
        }
        
        self.performance_history.append(result)
        
        # Analyze patterns if ground truth is available
        if ground_truth:
            self._analyze_detection_pattern(detection, ground_truth)
            
    def _analyze_detection_pattern(self, detection: Dict, ground_truth: Dict):
        """Analyze detection patterns for learning"""
        detected_label = detection.get("label", "")
        true_label = ground_truth.get("label", "")
        
        if detected_label != true_label:
            # Record error pattern
            error_key = f"{detected_label}_as_{true_label}"
            self.error_patterns[error_key].append({
                "detection": detection,
                "ground_truth": ground_truth,
                "timestamp": time.time()
            })
        else:
            # Record success pattern
            success_key = detected_label
            self.success_patterns[success_key].append({
                "detection": detection,
                "timestamp": time.time()
            })
            
    def get_adaptive_corrections(self, detections: List[Dict]) -> List[Dict]:
        """Apply adaptive corrections based on learned patterns"""
        corrected_detections = []
        
        for detection in detections:
            corrected = detection.copy()
            
            # Apply learned correction rules
            label = detection.get("label", "")
            confidence = detection.get("confidence", 0.5)
            
            # Check for known error patterns
            for error_pattern, examples in self.error_patterns.items():
                if len(examples) > 5:  # Enough examples to be confident
                    detected_as, should_be = error_pattern.split("_as_")
                    
                    if label == detected_as:
                        # Check if this detection matches the error pattern
                        if self._matches_error_pattern(detection, examples):
                            corrected["label"] = should_be
                            corrected["confidence"] *= 0.8  # Reduce confidence for corrections
                            corrected["correction_applied"] = f"adaptive_learning_{error_pattern}"
                            
            corrected_detections.append(corrected)
            
        return corrected_detections
        
    def _matches_error_pattern(self, detection: Dict, error_examples: List[Dict]) -> bool:
        """Check if detection matches known error pattern"""
        # Simple pattern matching based on size and position
        bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
        if len(bbox) < 4:
            return False
            
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        aspect_ratio = width / max(height, 1)
        
        # Check similarity with error examples
        similar_count = 0
        for example in error_examples[-10:]:  # Check last 10 examples
            ex_detection = example["detection"]
            ex_bbox = ex_detection.get("bbox_xyxy", [0, 0, 100, 100])
            
            if len(ex_bbox) >= 4:
                ex_width = ex_bbox[2] - ex_bbox[0]
                ex_height = ex_bbox[3] - ex_bbox[1]
                ex_area = ex_width * ex_height
                ex_aspect_ratio = ex_width / max(ex_height, 1)
                
                # Check similarity
                area_similarity = min(area, ex_area) / max(area, ex_area)
                ratio_similarity = min(aspect_ratio, ex_aspect_ratio) / max(aspect_ratio, ex_aspect_ratio)
                
                if area_similarity > 0.7 and ratio_similarity > 0.8:
                    similar_count += 1
                    
        return similar_count >= 3  # At least 3 similar examples

class EnsembleDetectionSystem:
    """Multi-model ensemble detection system"""
    
    def __init__(self):
        self.temporal_tracker = TemporalConsistencyTracker()
        self.adaptive_learner = AdaptiveLearningSystem()
        self.model_weights = {
            "yolo_primary": 0.4,
            "yolo_secondary": 0.3,
            "temporal_consensus": 0.2,
            "adaptive_learning": 0.1
        }
        self.performance_metrics = defaultdict(list)
        
    async def ensemble_detect(self, frame: np.ndarray, primary_model, frame_diagnostics: Dict) -> EnsembleResult:
        """Perform ensemble detection using multiple strategies"""
        
        start_time = time.time()
        
        # Strategy 1: Primary YOLO model (already done)
        primary_results = []  # This would be passed in from main detection
        
        # Strategy 2: Secondary YOLO with different parameters
        secondary_results = await self._secondary_yolo_detection(frame, primary_model, frame_diagnostics)
        
        # Strategy 3: Temporal consistency check
        temporal_results = self._apply_temporal_consistency(primary_results)
        
        # Strategy 4: Adaptive learning corrections
        adaptive_results = self.adaptive_learner.get_adaptive_corrections(primary_results)
        
        # Combine results using weighted voting
        final_detections = self._weighted_ensemble_fusion([
            ("primary", primary_results, self.model_weights["yolo_primary"]),
            ("secondary", secondary_results, self.model_weights["yolo_secondary"]),
            ("temporal", temporal_results, self.model_weights["temporal_consensus"]),
            ("adaptive", adaptive_results, self.model_weights["adaptive_learning"])
        ])
        
        # Calculate ensemble metrics
        processing_time = time.time() - start_time
        model_agreements = self._calculate_model_agreements([
            primary_results, secondary_results, temporal_results, adaptive_results
        ])
        
        consensus_strength = self._calculate_consensus_strength(final_detections)
        
        return EnsembleResult(
            final_detections=final_detections,
            model_agreements=model_agreements,
            confidence_scores={},
            processing_times={"total": processing_time},
            consensus_strength=consensus_strength
        )
        
    async def _secondary_yolo_detection(self, frame: np.ndarray, model, frame_diagnostics: Dict) -> List[Dict]:
        """Run secondary YOLO detection with different parameters"""
        
        # Use more conservative parameters for secondary detection
        secondary_params = {
            "conf": 0.2,  # Lower confidence threshold
            "iou": 0.4,   # Lower IoU threshold
            "max_det": 100,  # More detections
            "augment": True  # Always use TTA for secondary
        }
        
        try:
            results = model.predict(frame, verbose=False, **secondary_params)
            
            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                    
                for b in r.boxes:
                    cls_idx = int(b.cls[0].item())
                    label = model.names.get(cls_idx, str(cls_idx))
                    conf = float(b.conf[0].item())
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "source": "secondary_yolo"
                    })
                    
            return detections
            
        except Exception as e:
            print(f"Secondary YOLO detection failed: {e}")
            return []
            
    def _apply_temporal_consistency(self, detections: List[Dict]) -> List[Dict]:
        """Apply temporal consistency to detections"""
        
        consistent_detections = []
        
        for detection in detections:
            track_id = detection.get("track_id", 0)
            
            # Update temporal history
            self.temporal_tracker.update_object_history(
                track_id,
                detection.get("label", ""),
                detection.get("confidence", 0.5),
                detection.get("bbox_xyxy", [0, 0, 100, 100])
            )
            
            # Get temporal consensus
            consensus_label, consensus_conf = self.temporal_tracker.get_temporal_consensus(track_id)
            
            if consensus_label:
                # Use temporal consensus if available
                temporal_detection = detection.copy()
                temporal_detection["label"] = consensus_label
                temporal_detection["confidence"] = consensus_conf
                temporal_detection["source"] = "temporal_consensus"
                consistent_detections.append(temporal_detection)
            else:
                consistent_detections.append(detection)
                
        return consistent_detections
        
    def _weighted_ensemble_fusion(self, model_results: List[Tuple[str, List[Dict], float]]) -> List[Dict]:
        """Fuse results from multiple models using weighted voting"""
        
        # Group detections by spatial proximity
        detection_groups = []
        
        for model_name, detections, weight in model_results:
            for detection in detections:
                detection["model_weight"] = weight
                detection["model_source"] = model_name
                
                # Find matching group or create new one
                matched_group = None
                bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
                
                for group in detection_groups:
                    if self._detections_overlap(bbox, group[0].get("bbox_xyxy", [0, 0, 100, 100])):
                        matched_group = group
                        break
                        
                if matched_group:
                    matched_group.append(detection)
                else:
                    detection_groups.append([detection])
                    
        # Fuse each group
        fused_detections = []
        for group in detection_groups:
            if len(group) == 1:
                fused_detections.append(group[0])
            else:
                fused_detection = self._fuse_detection_group(group)
                fused_detections.append(fused_detection)
                
        return fused_detections
        
    def _detections_overlap(self, bbox1: List[float], bbox2: List[float], threshold: float = 0.3) -> bool:
        """Check if two bounding boxes overlap significantly"""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return False
            
        # Calculate IoU
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou >= threshold
        
    def _fuse_detection_group(self, group: List[Dict]) -> Dict:
        """Fuse a group of overlapping detections"""
        
        # Weighted voting for label
        label_votes = defaultdict(float)
        total_weight = 0
        
        for detection in group:
            weight = detection.get("model_weight", 1.0)
            confidence = detection.get("confidence", 0.5)
            label = detection.get("label", "")
            
            vote_strength = weight * confidence
            label_votes[label] += vote_strength
            total_weight += weight
            
        # Choose label with highest weighted vote
        winning_label = max(label_votes.keys(), key=lambda x: label_votes[x])
        
        # Calculate fused confidence
        fused_confidence = label_votes[winning_label] / total_weight if total_weight > 0 else 0.5
        
        # Calculate fused bounding box (weighted average)
        fused_bbox = [0, 0, 0, 0]
        for i in range(4):
            weighted_sum = 0
            for detection in group:
                bbox = detection.get("bbox_xyxy", [0, 0, 100, 100])
                weight = detection.get("model_weight", 1.0)
                if len(bbox) > i:
                    weighted_sum += bbox[i] * weight
            fused_bbox[i] = weighted_sum / total_weight if total_weight > 0 else 0
            
        # Create fused detection
        fused_detection = {
            "label": winning_label,
            "confidence": min(0.99, fused_confidence),
            "bbox_xyxy": fused_bbox,
            "source": "ensemble_fusion",
            "contributing_models": [d.get("model_source", "unknown") for d in group],
            "vote_strength": label_votes[winning_label]
        }
        
        # Copy other attributes from highest confidence detection
        highest_conf_detection = max(group, key=lambda x: x.get("confidence", 0))
        for key, value in highest_conf_detection.items():
            if key not in fused_detection:
                fused_detection[key] = value
                
        return fused_detection
        
    def _calculate_model_agreements(self, model_results: List[List[Dict]]) -> Dict[str, float]:
        """Calculate agreement between different models"""
        
        if len(model_results) < 2:
            return {"overall": 1.0}
            
        agreements = {}
        
        # Compare each pair of models
        for i in range(len(model_results)):
            for j in range(i + 1, len(model_results)):
                model1_results = model_results[i]
                model2_results = model_results[j]
                
                agreement = self._calculate_pairwise_agreement(model1_results, model2_results)
                agreements[f"model_{i}_vs_{j}"] = agreement
                
        # Calculate overall agreement
        overall_agreement = sum(agreements.values()) / len(agreements) if agreements else 1.0
        agreements["overall"] = overall_agreement
        
        return agreements
        
    def _calculate_pairwise_agreement(self, results1: List[Dict], results2: List[Dict]) -> float:
        """Calculate agreement between two sets of detection results"""
        
        if not results1 and not results2:
            return 1.0
        if not results1 or not results2:
            return 0.0
            
        matches = 0
        total_comparisons = 0
        
        for det1 in results1:
            for det2 in results2:
                total_comparisons += 1
                
                # Check if detections match
                if (det1.get("label") == det2.get("label") and
                    self._detections_overlap(
                        det1.get("bbox_xyxy", []), 
                        det2.get("bbox_xyxy", [])
                    )):
                    matches += 1
                    
        return matches / total_comparisons if total_comparisons > 0 else 0.0
        
    def _calculate_consensus_strength(self, detections: List[Dict]) -> float:
        """Calculate overall consensus strength of ensemble"""
        
        if not detections:
            return 0.0
            
        total_strength = 0
        for detection in detections:
            vote_strength = detection.get("vote_strength", detection.get("confidence", 0.5))
            total_strength += vote_strength
            
        return total_strength / len(detections)

# Global ensemble system instance
ensemble_system = EnsembleDetectionSystem()