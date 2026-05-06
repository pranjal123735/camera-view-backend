"""
Tesla-style 360° Surround Vision System for Motorcycles
Processes 4 camera feeds and provides comprehensive situational awareness
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HazardLevel(Enum):
    CLEAR = 0
    WATCH = 1
    WARNING = 2
    DANGER = 3

class Distance(Enum):
    NEAR = "near"    # <3m
    MID = "mid"      # 3-10m
    FAR = "far"      # >10m

class Motion(Enum):
    STATIC = "static"
    SLOW = "slow"
    FAST = "fast"
    APPROACHING = "approaching"
    RECEDING = "receding"

class Position(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

@dataclass
class DetectedObject:
    label: str
    confidence: float
    position_in_frame: str
    distance: str
    motion: str
    bbox: List[int]
    track_id: Optional[int] = None
    velocity: Optional[Tuple[float, float]] = None

@dataclass
class CameraData:
    objects: List[DetectedObject]
    lane_detected: bool
    road_surface: str
    hazard_level: int
    hazard_note: str
    frame_quality: float = 1.0
    timestamp: float = 0.0

@dataclass
class GlobalHazard:
    level: int
    direction: str
    note: str
    alert_color: str

class Motorcycle360Vision:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.object_tracker = {}
        self.previous_frames = {
            'front': None,
            'left': None,
            'right': None,
            'rear': None
        }
        
        # Initialize YOLO model (you can replace with your preferred model)
        try:
            self.net = cv2.dnn.readNet('yolov8n.pt')  # Adjust path as needed
            self.output_layers = self.net.getUnconnectedOutLayersNames()
        except:
            logger.warning("YOLO model not found, using mock detection")
            self.net = None
            
        # Load class names
        self.classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]

    def detect_objects(self, frame: np.ndarray, camera_position: str) -> List[DetectedObject]:
        """Detect objects in a single camera frame"""
        if self.net is None:
            # Mock detection for testing
            return self._mock_detection(frame, camera_position)
        
        height, width = frame.shape[:2]
        
        # Prepare frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        objects = []
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                
                # Determine position in frame
                center_x = x + w/2
                if center_x < width/3:
                    position = Position.LEFT.value
                elif center_x > 2*width/3:
                    position = Position.RIGHT.value
                else:
                    position = Position.CENTER.value
                
                # Estimate distance based on object size and type
                distance = self._estimate_distance(w, h, label, width, height)
                
                # Estimate motion (simplified - would need tracking for accuracy)
                motion = self._estimate_motion(x, y, w, h, label, camera_position)
                
                objects.append(DetectedObject(
                    label=label,
                    confidence=confidence,
                    position_in_frame=position,
                    distance=distance,
                    motion=motion,
                    bbox=[x, y, w, h]
                ))
        
        return objects

    def _mock_detection(self, frame: np.ndarray, camera_position: str) -> List[DetectedObject]:
        """Mock object detection for testing when YOLO is not available"""
        height, width = frame.shape[:2]
        
        # Generate some mock objects based on camera position
        mock_objects = []
        
        if camera_position == 'front':
            # Mock a car ahead
            mock_objects.append(DetectedObject(
                label="car",
                confidence=0.91,
                position_in_frame="center",
                distance="mid",
                motion="static",
                bbox=[width//3, height//3, width//3, height//4]
            ))
        elif camera_position == 'left':
            # Mock a person on the left
            mock_objects.append(DetectedObject(
                label="person",
                confidence=0.85,
                position_in_frame="left",
                distance="near",
                motion="slow",
                bbox=[50, height//2, 80, 150]
            ))
        
        return mock_objects

    def _estimate_distance(self, w: int, h: int, label: str, frame_width: int, frame_height: int) -> str:
        """Estimate distance based on object size and type"""
        # Object size relative to frame
        relative_size = (w * h) / (frame_width * frame_height)
        
        # Distance estimation based on object type and size
        if label in ["person", "bicycle", "motorcycle"]:
            if relative_size > 0.15:
                return Distance.NEAR.value
            elif relative_size > 0.05:
                return Distance.MID.value
            else:
                return Distance.FAR.value
        elif label in ["car", "truck", "bus"]:
            if relative_size > 0.25:
                return Distance.NEAR.value
            elif relative_size > 0.08:
                return Distance.MID.value
            else:
                return Distance.FAR.value
        else:
            # Default estimation
            if relative_size > 0.1:
                return Distance.NEAR.value
            elif relative_size > 0.03:
                return Distance.MID.value
            else:
                return Distance.FAR.value

    def _estimate_motion(self, x: int, y: int, w: int, h: int, label: str, camera_position: str) -> str:
        """Estimate object motion (simplified version)"""
        # This is a simplified version - real implementation would use tracking
        # For now, return static for most objects
        if label in ["car", "truck", "bus", "motorcycle"]:
            return Motion.STATIC.value
        elif label == "person":
            return Motion.SLOW.value
        else:
            return Motion.STATIC.value

    def analyze_lane_detection(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Detect lane markings and road surface"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Hough line detection for lane markings
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
        
        lane_detected = lines is not None and len(lines) > 0
        
        # Simple road surface detection based on texture
        # This is simplified - real implementation would use more sophisticated methods
        mean_intensity = np.mean(gray)
        if mean_intensity > 120:
            road_surface = "concrete"
        elif mean_intensity > 80:
            road_surface = "asphalt"
        else:
            road_surface = "dirt"
        
        return lane_detected, road_surface

    def calculate_hazard_level(self, objects: List[DetectedObject], camera_position: str) -> Tuple[int, str]:
        """Calculate hazard level for a camera view"""
        max_hazard = HazardLevel.CLEAR.value
        hazard_note = "All clear"
        
        for obj in objects:
            hazard = HazardLevel.CLEAR.value
            
            # High-risk objects
            if obj.label in ["car", "truck", "bus", "motorcycle"]:
                if obj.distance == Distance.NEAR.value:
                    if obj.motion == Motion.APPROACHING.value:
                        hazard = HazardLevel.DANGER.value
                        hazard_note = f"{obj.label.title()} approaching fast from {camera_position}"
                    elif obj.motion in [Motion.FAST.value, Motion.STATIC.value]:
                        hazard = HazardLevel.WARNING.value
                        hazard_note = f"{obj.label.title()} {obj.distance} on {camera_position}"
                    else:
                        hazard = HazardLevel.WATCH.value
                        hazard_note = f"{obj.label.title()} detected on {camera_position}"
                elif obj.distance == Distance.MID.value:
                    if obj.motion == Motion.APPROACHING.value:
                        hazard = HazardLevel.WARNING.value
                        hazard_note = f"{obj.label.title()} approaching from {camera_position}"
                    else:
                        hazard = HazardLevel.WATCH.value
                        hazard_note = f"{obj.label.title()} {obj.distance} on {camera_position}"
            
            # Pedestrians and cyclists
            elif obj.label in ["person", "bicycle"]:
                if obj.distance == Distance.NEAR.value:
                    hazard = HazardLevel.WARNING.value
                    hazard_note = f"{obj.label.title()} close on {camera_position}"
                elif obj.distance == Distance.MID.value:
                    hazard = HazardLevel.WATCH.value
                    hazard_note = f"{obj.label.title()} on {camera_position}"
            
            max_hazard = max(max_hazard, hazard)
        
        return max_hazard, hazard_note

    def calculate_global_hazard(self, camera_data: Dict[str, CameraData]) -> GlobalHazard:
        """Calculate overall hazard level from all cameras"""
        max_hazard = HazardLevel.CLEAR.value
        hazard_direction = "none"
        hazard_note = "All clear"
        
        # Priority order for directions (front is most critical)
        direction_priority = ["front", "left", "right", "rear"]
        
        for direction in direction_priority:
            if direction in camera_data:
                data = camera_data[direction]
                if data.hazard_level > max_hazard:
                    max_hazard = data.hazard_level
                    hazard_direction = direction
                    hazard_note = data.hazard_note
        
        # Determine alert color
        alert_colors = {
            HazardLevel.CLEAR.value: "green",
            HazardLevel.WATCH.value: "blue",
            HazardLevel.WARNING.value: "yellow",
            HazardLevel.DANGER.value: "red"
        }
        
        return GlobalHazard(
            level=max_hazard,
            direction=hazard_direction,
            note=hazard_note,
            alert_color=alert_colors[max_hazard]
        )

    def process_frame_set(self, frames: Dict[str, np.ndarray], 
                         bike_speed: Optional[float] = None,
                         bike_heading: Optional[float] = None) -> Dict:
        """Process a complete set of 4 camera frames"""
        timestamp = int((time.time() - self.start_time) * 1000)
        self.frame_count += 1
        
        # Analyze each camera feed
        camera_data = {}
        
        for camera_position, frame in frames.items():
            if frame is not None:
                # Detect objects
                objects = self.detect_objects(frame, camera_position)
                
                # Analyze lane detection
                lane_detected, road_surface = self.analyze_lane_detection(frame)
                
                # Calculate hazard level
                hazard_level, hazard_note = self.calculate_hazard_level(objects, camera_position)
                
                camera_data[camera_position] = CameraData(
                    objects=objects,
                    lane_detected=lane_detected,
                    road_surface=road_surface,
                    hazard_level=hazard_level,
                    hazard_note=hazard_note,
                    timestamp=timestamp
                )
        
        # Calculate global hazard
        global_hazard = self.calculate_global_hazard(camera_data)
        
        # Determine road type and weather (simplified)
        road_type = self._determine_road_type(camera_data)
        weather = self._determine_weather(camera_data)
        
        # Build output JSON
        output = {
            "timestamp": str(timestamp),
            "speed": bike_speed,
            "road_type": road_type,
            "weather": weather,
            "bike": {
                "position": "center",
                "heading": bike_heading or 0
            },
            "cameras": {},
            "global_hazard": {
                "level": global_hazard.level,
                "direction": global_hazard.direction,
                "note": global_hazard.note,
                "alert_color": global_hazard.alert_color
            }
        }
        
        # Add camera data to output
        for camera_position, data in camera_data.items():
            output["cameras"][camera_position] = {
                "objects": [
                    {
                        "label": obj.label,
                        "confidence": obj.confidence,
                        "position_in_frame": obj.position_in_frame,
                        "distance": obj.distance,
                        "motion": obj.motion,
                        "bbox": obj.bbox
                    }
                    for obj in data.objects
                ],
                "lane_detected": data.lane_detected,
                "road_surface": data.road_surface,
                "hazard_level": data.hazard_level,
                "hazard_note": data.hazard_note
            }
        
        return output

    def _determine_road_type(self, camera_data: Dict[str, CameraData]) -> str:
        """Determine road type based on camera analysis"""
        # Simplified road type detection
        # Real implementation would analyze lane markings, signs, etc.
        return "urban"  # Default

    def _determine_weather(self, camera_data: Dict[str, CameraData]) -> str:
        """Determine weather conditions based on camera analysis"""
        # Simplified weather detection
        # Real implementation would analyze brightness, contrast, etc.
        return "clear"  # Default

    def process_single_camera_fallback(self, front_frame: np.ndarray, 
                                     bike_speed: Optional[float] = None,
                                     bike_heading: Optional[float] = None) -> Dict:
        """Fallback mode for single camera (front only)"""
        timestamp = int((time.time() - self.start_time) * 1000)
        
        # Process front camera
        front_objects = self.detect_objects(front_frame, "front")
        lane_detected, road_surface = self.analyze_lane_detection(front_frame)
        hazard_level, hazard_note = self.calculate_hazard_level(front_objects, "front")
        
        # Create estimated data for other cameras
        estimated_cameras = {}
        for camera_pos in ["left", "right", "rear"]:
            estimated_cameras[camera_pos] = {
                "objects": [],  # No objects in estimated views
                "lane_detected": False,
                "road_surface": "unknown",
                "hazard_level": 0,
                "hazard_note": "Estimated view - no camera feed",
                "estimated": True
            }
        
        output = {
            "timestamp": str(timestamp),
            "speed": bike_speed,
            "road_type": "urban",
            "weather": "clear",
            "bike": {
                "position": "center",
                "heading": bike_heading or 0
            },
            "cameras": {
                "front": {
                    "objects": [
                        {
                            "label": obj.label,
                            "confidence": obj.confidence,
                            "position_in_frame": obj.position_in_frame,
                            "distance": obj.distance,
                            "motion": obj.motion,
                            "bbox": obj.bbox
                        }
                        for obj in front_objects
                    ],
                    "lane_detected": lane_detected,
                    "road_surface": road_surface,
                    "hazard_level": hazard_level,
                    "hazard_note": hazard_note
                },
                **estimated_cameras
            },
            "global_hazard": {
                "level": hazard_level,
                "direction": "front" if hazard_level > 0 else "none",
                "note": hazard_note,
                "alert_color": ["green", "blue", "yellow", "red"][hazard_level]
            },
            "fallback_mode": True
        }
        
        return output

# Example usage and testing
if __name__ == "__main__":
    vision_system = Motorcycle360Vision()
    
    # Mock camera frames (replace with actual camera feeds)
    def create_mock_frame(width=640, height=480):
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Test with 4 cameras
    frames = {
        "front": create_mock_frame(),
        "left": create_mock_frame(),
        "right": create_mock_frame(),
        "rear": create_mock_frame()
    }
    
    result = vision_system.process_frame_set(frames, bike_speed=45.0, bike_heading=90.0)
    print("4-Camera Result:")
    print(json.dumps(result, indent=2))
    
    # Test fallback mode
    front_only = create_mock_frame()
    fallback_result = vision_system.process_single_camera_fallback(front_only, bike_speed=30.0)
    print("\nFallback Mode Result:")
    print(json.dumps(fallback_result, indent=2))