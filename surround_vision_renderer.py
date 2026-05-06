"""
360° Surround Vision Renderer for Motorcycle HUD
Generates animated canvas scenes for left, right, and rear panels based on front camera feed
"""

import cv2
import numpy as np
import json
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random

class RoadType(Enum):
    URBAN = "urban"
    HIGHWAY = "highway"
    RURAL = "rural"
    PARKING = "parking"
    OFFROAD = "offroad"

class TurnDirection(Enum):
    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"

class Weather(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    NIGHT = "night"

@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: List[int]  # [x, y, w, h]
    distance_m: float
    position: str  # left, center, right
    is_moving: bool

@dataclass
class SceneElement:
    type: str  # building, tree, barrier, car, person
    position: Tuple[float, float]  # x, y in panel coordinates
    size: Tuple[float, float]  # width, height
    color: str
    speed: float  # scroll speed multiplier
    distance_layer: int  # 0=far, 1=mid, 2=near for parallax

@dataclass
class PanelData:
    bg_color: str
    objects: List[Dict]
    scroll_speed: float
    scene_elements: List[Dict]
    road_markings: List[Dict]
    weather_effects: List[Dict]

class SurroundVisionRenderer:
    def __init__(self):
        self.frame_count = 0
        self.time_offset = 0
        self.road_texture_offset = 0
        self.last_speed = 0
        self.last_turn = TurnDirection.STRAIGHT
        
        # Scene memory for consistency
        self.scene_context = {
            'road_color': '#404040',
            'sky_color': '#87CEEB',
            'lighting': 'day',
            'weather': Weather.CLEAR,
            'road_type': RoadType.URBAN
        }
        
        # Object tracking for smooth animations
        self.tracked_objects = {}
        self.object_id_counter = 0
        
    def analyze_front_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze front camera frame to extract scene context"""
        if frame is None:
            return self.scene_context
            
        height, width = frame.shape[:2]
        
        # Analyze colors
        # Sky region (top 30% of frame)
        sky_region = frame[:int(height * 0.3), :]
        sky_color = self._get_dominant_color(sky_region)
        
        # Road region (bottom 50% of frame)
        road_region = frame[int(height * 0.5):, :]
        road_color = self._get_dominant_color(road_region)
        
        # Lighting analysis
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        lighting = 'night' if brightness < 80 else 'day'
        
        # Weather detection (simplified)
        weather = Weather.CLEAR
        if brightness < 60:
            weather = Weather.NIGHT
        elif self._detect_rain_pattern(frame):
            weather = Weather.RAIN
        elif brightness < 100 and np.std(frame) < 30:
            weather = Weather.FOG
            
        self.scene_context.update({
            'road_color': self._rgb_to_hex(road_color),
            'sky_color': self._rgb_to_hex(sky_color),
            'lighting': lighting,
            'weather': weather,
            'brightness': brightness
        })
        
        return self.scene_context
    
    def _get_dominant_color(self, region: np.ndarray) -> Tuple[int, int, int]:
        """Get dominant color from image region"""
        if region.size == 0:
            return (128, 128, 128)
            
        # Reshape and get mean color
        pixels = region.reshape(-1, 3)
        mean_color = np.mean(pixels, axis=0)
        return tuple(map(int, mean_color))
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def _detect_rain_pattern(self, frame: np.ndarray) -> bool:
        """Simple rain detection based on texture patterns"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Look for vertical streaks (rain drops)
        kernel = np.array([[1], [1], [1], [1], [1]], dtype=np.float32) / 5
        filtered = cv2.filter2D(gray, -1, kernel)
        variance = np.var(filtered - gray)
        return variance > 100  # Threshold for rain detection
    
    def generate_road_markings(self, panel: str, speed: float, turn_direction: TurnDirection) -> List[Dict]:
        """Generate animated road markings for a panel"""
        markings = []
        
        # Calculate scroll speed based on bike speed
        base_scroll_speed = speed * 0.1  # Adjust multiplier as needed
        
        if panel == 'rear':
            base_scroll_speed *= -1  # Reverse direction for rear
            
        # Lane markings
        for i in range(5):  # 5 dashed lines
            y_pos = (i * 100 + self.road_texture_offset) % 500
            
            if panel == 'left' and turn_direction == TurnDirection.LEFT:
                # Curve markings into view
                x_offset = math.sin(y_pos * 0.01) * 20
            elif panel == 'right' and turn_direction == TurnDirection.RIGHT:
                x_offset = -math.sin(y_pos * 0.01) * 20
            else:
                x_offset = 0
                
            markings.append({
                'type': 'lane_marking',
                'x': 160 + x_offset,  # Center of 320px panel
                'y': y_pos,
                'width': 4,
                'height': 40,
                'color': '#FFFFFF'
            })
            
        return markings
    
    def generate_scene_elements(self, panel: str, road_type: RoadType, speed: float) -> List[Dict]:
        """Generate background scene elements for a panel"""
        elements = []
        
        if road_type == RoadType.URBAN:
            elements.extend(self._generate_urban_elements(panel, speed))
        elif road_type == RoadType.HIGHWAY:
            elements.extend(self._generate_highway_elements(panel, speed))
        elif road_type == RoadType.RURAL:
            elements.extend(self._generate_rural_elements(panel, speed))
            
        return elements
    
    def _generate_urban_elements(self, panel: str, speed: float) -> List[Dict]:
        """Generate urban scene elements (buildings, shops, etc.)"""
        elements = []
        
        # Buildings in background
        for i in range(3):
            x_pos = i * 120 + random.randint(-20, 20)
            height = random.randint(80, 150)
            
            elements.append({
                'type': 'building',
                'x': x_pos,
                'y': 0,
                'width': 100,
                'height': height,
                'color': f"#{random.randint(100, 150):02x}{random.randint(100, 150):02x}{random.randint(100, 150):02x}",
                'distance_layer': 0,  # Far background
                'scroll_speed': speed * 0.2
            })
            
        # Street elements
        if panel in ['left', 'right']:
            # Sidewalk
            elements.append({
                'type': 'sidewalk',
                'x': 0 if panel == 'left' else 280,
                'y': 200,
                'width': 40,
                'height': 200,
                'color': '#CCCCCC',
                'distance_layer': 1,
                'scroll_speed': speed * 0.5
            })
            
        return elements
    
    def _generate_highway_elements(self, panel: str, speed: float) -> List[Dict]:
        """Generate highway scene elements (barriers, trees, etc.)"""
        elements = []
        
        # Highway barriers
        if panel in ['left', 'right']:
            for i in range(10):
                y_pos = i * 50 + (self.road_texture_offset * 0.3) % 500
                elements.append({
                    'type': 'barrier',
                    'x': 20 if panel == 'left' else 280,
                    'y': y_pos,
                    'width': 20,
                    'height': 30,
                    'color': '#888888',
                    'distance_layer': 1,
                    'scroll_speed': speed * 0.4
                })
                
        # Trees in far background
        for i in range(5):
            x_pos = random.randint(0, 320)
            y_pos = (i * 100 + self.road_texture_offset * 0.1) % 500
            
            elements.append({
                'type': 'tree',
                'x': x_pos,
                'y': y_pos,
                'width': 30,
                'height': 60,
                'color': '#228B22',
                'distance_layer': 0,
                'scroll_speed': speed * 0.1
            })
            
        return elements
    
    def _generate_rural_elements(self, panel: str, speed: float) -> List[Dict]:
        """Generate rural scene elements (fields, fences, etc.)"""
        elements = []
        
        # Fields
        elements.append({
            'type': 'field',
            'x': 0,
            'y': 0,
            'width': 320,
            'height': 150,
            'color': '#90EE90',
            'distance_layer': 0,
            'scroll_speed': speed * 0.1
        })
        
        # Fence posts
        if panel in ['left', 'right']:
            for i in range(8):
                y_pos = i * 60 + (self.road_texture_offset * 0.4) % 480
                elements.append({
                    'type': 'fence_post',
                    'x': 30 if panel == 'left' else 290,
                    'y': y_pos,
                    'width': 5,
                    'height': 40,
                    'color': '#8B4513',
                    'distance_layer': 1,
                    'scroll_speed': speed * 0.3
                })
                
        return elements
    
    def infer_side_objects(self, front_objects: List[DetectedObject], panel: str, speed: float) -> List[Dict]:
        """Infer objects that might be visible in side/rear panels"""
        inferred_objects = []
        
        for obj in front_objects:
            # Probability of object appearing in side panels
            if panel == 'left' and obj.position == 'left':
                probability = 0.4
            elif panel == 'right' and obj.position == 'right':
                probability = 0.4
            elif panel == 'rear':
                probability = 0.2  # Objects might be behind
            else:
                probability = 0.1  # Random chance
                
            if random.random() < probability:
                # Create inferred object
                inferred_obj = {
                    'label': obj.label,
                    'confidence': obj.confidence * 0.7,  # Lower confidence for inferred
                    'x': random.randint(50, 270),
                    'y': random.randint(150, 300),
                    'width': random.randint(40, 80),
                    'height': random.randint(60, 120),
                    'distance_m': obj.distance_m + random.uniform(-5, 5),
                    'is_inferred': True,
                    'scroll_speed': speed * 0.6
                }
                
                # Adjust for panel type
                if panel == 'rear':
                    inferred_obj['scroll_speed'] *= -1  # Reverse direction
                    
                inferred_objects.append(inferred_obj)
                
        return inferred_objects
    
    def generate_weather_effects(self, weather: Weather, speed: float) -> List[Dict]:
        """Generate weather effect particles"""
        effects = []
        
        if weather == Weather.RAIN:
            # Rain drops
            for i in range(20):
                effects.append({
                    'type': 'raindrop',
                    'x': random.randint(0, 320),
                    'y': random.randint(0, 400),
                    'width': 2,
                    'height': 15,
                    'color': '#ADD8E6',
                    'velocity_y': speed * 2 + 10,
                    'opacity': 0.6
                })
                
        elif weather == Weather.FOG:
            # Fog overlay
            effects.append({
                'type': 'fog',
                'x': 0,
                'y': 0,
                'width': 320,
                'height': 400,
                'color': '#F5F5F5',
                'opacity': 0.3
            })
            
        return effects
    
    def calculate_bike_state(self, speed: float, turn_direction: TurnDirection, road_type: RoadType) -> Dict:
        """Calculate bike model state for center display"""
        # Wheel rotation speed
        wheel_speed = speed * 0.1  # Adjust for visual effect
        
        # Lean angle based on turn and speed
        lean_angle = 0
        if turn_direction == TurnDirection.LEFT:
            lean_angle = -min(15, speed * 0.3)  # Lean left
        elif turn_direction == TurnDirection.RIGHT:
            lean_angle = min(15, speed * 0.3)   # Lean right
            
        # Headlight based on lighting conditions
        headlight = self.scene_context.get('lighting') == 'night'
        
        # Suspension bounce for rough roads
        bounce = 0
        if road_type == RoadType.OFFROAD:
            bounce = math.sin(self.frame_count * 0.2) * 2
            
        return {
            'lean_angle': lean_angle,
            'wheel_speed': wheel_speed,
            'headlight': headlight,
            'bounce': bounce,
            'speed_display': speed
        }
    
    def render_frame(self, 
                    front_frame: Optional[np.ndarray],
                    detected_objects: List[DetectedObject],
                    road_type: str,
                    speed: float,
                    turn_direction: str) -> Dict[str, Any]:
        """Main rendering function - generates 360° scene data"""
        
        self.frame_count += 1
        self.road_texture_offset += speed * 0.5  # Update road animation
        
        # Convert string enums
        road_type_enum = RoadType(road_type.lower())
        turn_direction_enum = TurnDirection(turn_direction.lower())
        
        # Analyze front frame for scene context
        if front_frame is not None:
            scene_context = self.analyze_front_frame(front_frame)
        else:
            scene_context = self.scene_context
            
        # Generate panel data
        panels = {}
        
        for panel in ['left', 'right', 'rear']:
            # Generate road markings
            road_markings = self.generate_road_markings(panel, speed, turn_direction_enum)
            
            # Generate scene elements
            scene_elements = self.generate_scene_elements(panel, road_type_enum, speed)
            
            # Infer objects for this panel
            inferred_objects = self.infer_side_objects(detected_objects, panel, speed)
            
            # Generate weather effects
            weather_effects = self.generate_weather_effects(scene_context['weather'], speed)
            
            panels[panel] = {
                'bg_color': scene_context['sky_color'],
                'objects': inferred_objects,
                'scroll_speed': speed * 0.1,
                'scene_elements': [elem.__dict__ if hasattr(elem, '__dict__') else elem for elem in scene_elements],
                'road_markings': road_markings,
                'weather_effects': weather_effects,
                'road_color': scene_context['road_color']
            }
        
        # Calculate bike state
        bike_state = self.calculate_bike_state(speed, turn_direction_enum, road_type_enum)
        
        # Stitch information for seamless 360° view
        stitch_info = {
            'sky_color': scene_context['sky_color'],
            'road_color': scene_context['road_color'],
            'weather': scene_context['weather'].value,
            'lighting': scene_context['lighting'],
            'brightness': scene_context.get('brightness', 128)
        }
        
        return {
            'front': 'use_real_video',
            'left': panels['left'],
            'right': panels['right'],
            'rear': panels['rear'],
            'bike': bike_state,
            'stitch': stitch_info,
            'frame_count': self.frame_count,
            'timestamp': time.time()
        }

# Example usage and testing
if __name__ == "__main__":
    renderer = SurroundVisionRenderer()
    
    # Mock detected objects
    mock_objects = [
        DetectedObject(
            label="car",
            confidence=0.9,
            bbox=[100, 150, 80, 60],
            distance_m=15.0,
            position="center",
            is_moving=True
        ),
        DetectedObject(
            label="person",
            confidence=0.8,
            bbox=[50, 200, 40, 80],
            distance_m=8.0,
            position="left",
            is_moving=False
        )
    ]
    
    # Generate frame
    result = renderer.render_frame(
        front_frame=None,  # Would be actual camera frame
        detected_objects=mock_objects,
        road_type="urban",
        speed=45.0,
        turn_direction="straight"
    )
    
    print("Generated 360° Scene Data:")
    print(json.dumps(result, indent=2, default=str))