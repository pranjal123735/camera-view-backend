"""
Smart Learning & Personalization Service for Car Vision System
Provides user behavior learning, custom object training, and personalized optimization.
"""

import json
import sqlite3
import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import hashlib

@dataclass
class UserProfile:
    user_id: str
    mobility_mode: str  # 'walking', 'cycling', 'driving'
    preferred_sensitivity: Dict[str, float]
    common_routes: List[Dict]
    safety_preferences: Dict[str, Any]
    learning_data: Dict[str, Any]
    created_at: float
    last_updated: float

@dataclass
class RoutePattern:
    route_id: str
    route_hash: str
    frequency: int
    average_duration: float
    common_objects: List[str]
    risk_hotspots: List[Tuple[float, float]]
    optimal_settings: Dict[str, float]
    weather_adaptations: Dict[str, Dict]

@dataclass
class CustomObject:
    object_id: str
    user_id: str
    object_name: str
    training_images: List[str]
    detection_confidence: float
    false_positive_rate: float
    last_trained: float
    active: bool

@dataclass
class BehaviorPattern:
    pattern_id: str
    user_id: str
    pattern_type: str  # 'speed', 'route', 'time', 'reaction'
    pattern_data: Dict[str, Any]
    confidence: float
    frequency: int
    last_observed: float

class LearningService:
    def __init__(self, db_path: str = "learning.db"):
        self.db_path = db_path
        self.user_profiles = {}
        self.route_patterns = {}
        self.custom_objects = {}
        self.behavior_patterns = defaultdict(list)
        self.learning_sessions = {}
        
        # Learning parameters
        self.min_route_frequency = 3
        self.pattern_confidence_threshold = 0.7
        self.adaptation_learning_rate = 0.1
        
        self._init_database()
        self._load_existing_data()
    
    def _init_database(self):
        """Initialize learning database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                mobility_mode TEXT,
                preferred_sensitivity TEXT,
                common_routes TEXT,
                safety_preferences TEXT,
                learning_data TEXT,
                created_at REAL,
                last_updated REAL
            )
        """)
        
        # Route patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS route_patterns (
                route_id TEXT PRIMARY KEY,
                user_id TEXT,
                route_hash TEXT,
                frequency INTEGER,
                average_duration REAL,
                common_objects TEXT,
                risk_hotspots TEXT,
                optimal_settings TEXT,
                weather_adaptations TEXT,
                last_used REAL
            )
        """)
        
        # Custom objects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_objects (
                object_id TEXT PRIMARY KEY,
                user_id TEXT,
                object_name TEXT,
                training_images TEXT,
                detection_confidence REAL,
                false_positive_rate REAL,
                last_trained REAL,
                active BOOLEAN
            )
        """)
        
        # Behavior patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS behavior_patterns (
                pattern_id TEXT PRIMARY KEY,
                user_id TEXT,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                frequency INTEGER,
                last_observed REAL
            )
        """)
        
        # Learning sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                session_type TEXT,
                start_time REAL,
                end_time REAL,
                data_points INTEGER,
                improvements_made TEXT,
                success_rate REAL
            )
        """)
        
        # Detection feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_feedback (
                feedback_id TEXT PRIMARY KEY,
                user_id TEXT,
                timestamp REAL,
                detection_data TEXT,
                user_correction TEXT,
                feedback_type TEXT,
                confidence_before REAL,
                confidence_after REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_existing_data(self):
        """Load existing learning data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load user profiles
        cursor.execute("SELECT * FROM user_profiles")
        for row in cursor.fetchall():
            profile = UserProfile(
                user_id=row[0],
                mobility_mode=row[1],
                preferred_sensitivity=json.loads(row[2]) if row[2] else {},
                common_routes=json.loads(row[3]) if row[3] else [],
                safety_preferences=json.loads(row[4]) if row[4] else {},
                learning_data=json.loads(row[5]) if row[5] else {},
                created_at=row[6],
                last_updated=row[7]
            )
            self.user_profiles[profile.user_id] = profile
        
        # Load route patterns
        cursor.execute("SELECT * FROM route_patterns")
        for row in cursor.fetchall():
            pattern = RoutePattern(
                route_id=row[0],
                route_hash=row[2],
                frequency=row[3],
                average_duration=row[4],
                common_objects=json.loads(row[5]) if row[5] else [],
                risk_hotspots=json.loads(row[6]) if row[6] else [],
                optimal_settings=json.loads(row[7]) if row[7] else {},
                weather_adaptations=json.loads(row[8]) if row[8] else {}
            )
            self.route_patterns[pattern.route_id] = pattern
        
        conn.close()
    
    def create_user_profile(self, user_id: str, mobility_mode: str = "cycling") -> UserProfile:
        """Create a new user profile with default settings"""
        current_time = time.time()
        
        # Default sensitivity settings based on mobility mode
        default_sensitivity = {
            "walking": {
                "person": 0.8, "bicycle": 0.9, "motorcycle": 0.95, 
                "car": 0.95, "truck": 0.98, "bus": 0.98
            },
            "cycling": {
                "person": 0.85, "bicycle": 0.7, "motorcycle": 0.9, 
                "car": 0.95, "truck": 0.98, "bus": 0.98
            },
            "driving": {
                "person": 0.95, "bicycle": 0.95, "motorcycle": 0.9, 
                "car": 0.8, "truck": 0.85, "bus": 0.85
            }
        }
        
        profile = UserProfile(
            user_id=user_id,
            mobility_mode=mobility_mode,
            preferred_sensitivity=default_sensitivity.get(mobility_mode, default_sensitivity["cycling"]),
            common_routes=[],
            safety_preferences={
                "voice_alerts": True,
                "haptic_feedback": True,
                "alert_distance": 15.0,
                "emergency_contacts": []
            },
            learning_data={
                "total_sessions": 0,
                "total_detections": 0,
                "accuracy_improvements": 0,
                "custom_objects_trained": 0
            },
            created_at=current_time,
            last_updated=current_time
        )
        
        self.user_profiles[user_id] = profile
        self._save_user_profile(profile)
        
        return profile
    
    def learn_user_behavior(self, user_id: str, session_data: Dict[str, Any]):
        """Learn from user behavior during a session"""
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        # Analyze speed patterns
        if 'speed_data' in session_data:
            self._learn_speed_patterns(user_id, session_data['speed_data'])
        
        # Analyze route patterns
        if 'route_data' in session_data:
            self._learn_route_patterns(user_id, session_data['route_data'])
        
        # Analyze reaction patterns
        if 'reaction_data' in session_data:
            self._learn_reaction_patterns(user_id, session_data['reaction_data'])
        
        # Update learning statistics
        profile.learning_data['total_sessions'] += 1
        profile.last_updated = time.time()
        
        self._save_user_profile(profile)
    
    def _learn_speed_patterns(self, user_id: str, speed_data: List[Dict]):
        """Learn user's speed patterns and preferences"""
        if not speed_data:
            return
        
        speeds = [d['speed'] for d in speed_data if 'speed' in d]
        times = [d['time'] for d in speed_data if 'time' in d]
        
        if len(speeds) < 10:
            return
        
        # Calculate speed statistics
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        speed_variance = np.var(speeds)
        
        # Detect speed patterns by time of day
        hourly_speeds = defaultdict(list)
        for i, timestamp in enumerate(times):
            hour = int((timestamp % 86400) / 3600)  # Hour of day
            if i < len(speeds):
                hourly_speeds[hour].append(speeds[i])
        
        # Create behavior pattern
        pattern_data = {
            'average_speed': avg_speed,
            'max_speed': max_speed,
            'speed_variance': speed_variance,
            'hourly_patterns': {str(h): np.mean(speeds) for h, speeds in hourly_speeds.items() if speeds}
        }
        
        pattern = BehaviorPattern(
            pattern_id=f"speed_{user_id}_{int(time.time())}",
            user_id=user_id,
            pattern_type="speed",
            pattern_data=pattern_data,
            confidence=min(1.0, len(speeds) / 100.0),
            frequency=len(speeds),
            last_observed=time.time()
        )
        
        self.behavior_patterns[user_id].append(pattern)
        self._save_behavior_pattern(pattern)
    
    def _learn_route_patterns(self, user_id: str, route_data: Dict):
        """Learn user's common routes and optimize settings for them"""
        if not route_data or 'waypoints' not in route_data:
            return
        
        # Create route hash from waypoints
        waypoints_str = json.dumps(route_data['waypoints'], sort_keys=True)
        route_hash = hashlib.md5(waypoints_str.encode()).hexdigest()
        
        # Check if this route exists
        existing_route = None
        for route in self.route_patterns.values():
            if route.route_hash == route_hash:
                existing_route = route
                break
        
        if existing_route:
            # Update existing route
            existing_route.frequency += 1
            existing_route.average_duration = (
                (existing_route.average_duration * (existing_route.frequency - 1) + 
                 route_data.get('duration', 0)) / existing_route.frequency
            )
        else:
            # Create new route pattern
            route_id = f"route_{user_id}_{int(time.time())}"
            new_route = RoutePattern(
                route_id=route_id,
                route_hash=route_hash,
                frequency=1,
                average_duration=route_data.get('duration', 0),
                common_objects=route_data.get('objects_seen', []),
                risk_hotspots=route_data.get('risk_points', []),
                optimal_settings={},
                weather_adaptations={}
            )
            
            self.route_patterns[route_id] = new_route
            
            # Add to user's common routes if frequent enough
            if new_route.frequency >= self.min_route_frequency:
                profile = self.user_profiles[user_id]
                profile.common_routes.append({
                    'route_id': route_id,
                    'name': route_data.get('name', f'Route {len(profile.common_routes) + 1}'),
                    'frequency': new_route.frequency
                })
                self._save_user_profile(profile)
    
    def _learn_reaction_patterns(self, user_id: str, reaction_data: List[Dict]):
        """Learn user's reaction patterns to different threats"""
        if not reaction_data:
            return
        
        # Analyze reaction times by object type
        reaction_times = defaultdict(list)
        for reaction in reaction_data:
            if 'object_type' in reaction and 'reaction_time' in reaction:
                reaction_times[reaction['object_type']].append(reaction['reaction_time'])
        
        # Calculate average reaction times
        avg_reactions = {}
        for obj_type, times in reaction_times.items():
            if times:
                avg_reactions[obj_type] = {
                    'average_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'samples': len(times)
                }
        
        if avg_reactions:
            pattern = BehaviorPattern(
                pattern_id=f"reaction_{user_id}_{int(time.time())}",
                user_id=user_id,
                pattern_type="reaction",
                pattern_data=avg_reactions,
                confidence=min(1.0, sum(len(times) for times in reaction_times.values()) / 50.0),
                frequency=len(reaction_data),
                last_observed=time.time()
            )
            
            self.behavior_patterns[user_id].append(pattern)
            self._save_behavior_pattern(pattern)
    
    def train_custom_object(self, user_id: str, object_name: str, 
                           training_images: List[str]) -> CustomObject:
        """Train a custom object detector for the user"""
        object_id = f"custom_{user_id}_{object_name}_{int(time.time())}"
        
        # Simulate training process (in real implementation, this would train a model)
        detection_confidence = 0.75 + (len(training_images) * 0.02)  # More images = better confidence
        false_positive_rate = max(0.05, 0.2 - (len(training_images) * 0.01))
        
        custom_object = CustomObject(
            object_id=object_id,
            user_id=user_id,
            object_name=object_name,
            training_images=training_images,
            detection_confidence=min(0.95, detection_confidence),
            false_positive_rate=false_positive_rate,
            last_trained=time.time(),
            active=True
        )
        
        self.custom_objects[object_id] = custom_object
        
        # Update user profile
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.learning_data['custom_objects_trained'] += 1
            self._save_user_profile(profile)
        
        # Save custom object
        self._save_custom_object(custom_object)
        
        return custom_object
    
    def get_personalized_settings(self, user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get personalized detection settings for the user"""
        if user_id not in self.user_profiles:
            return self._get_default_settings()
        
        profile = self.user_profiles[user_id]
        settings = {
            'sensitivity': profile.preferred_sensitivity.copy(),
            'alert_distance': profile.safety_preferences.get('alert_distance', 15.0),
            'voice_alerts': profile.safety_preferences.get('voice_alerts', True),
            'haptic_feedback': profile.safety_preferences.get('haptic_feedback', True)
        }
        
        # Adjust based on context
        if context:
            # Weather adaptations
            if context.get('weather') == 'rain':
                for obj_type in settings['sensitivity']:
                    settings['sensitivity'][obj_type] = min(0.98, settings['sensitivity'][obj_type] * 1.1)
                settings['alert_distance'] *= 1.3
            
            # Time of day adaptations
            hour = context.get('hour', 12)
            if hour < 6 or hour > 20:  # Night time
                for obj_type in settings['sensitivity']:
                    settings['sensitivity'][obj_type] = min(0.98, settings['sensitivity'][obj_type] * 1.05)
                settings['alert_distance'] *= 1.2
            
            # Route-specific adaptations
            route_id = context.get('route_id')
            if route_id and route_id in self.route_patterns:
                route = self.route_patterns[route_id]
                if route.optimal_settings:
                    settings.update(route.optimal_settings)
        
        return settings
    
    def record_detection_feedback(self, user_id: str, detection: Dict, 
                                 user_correction: str, feedback_type: str):
        """Record user feedback on detection accuracy"""
        feedback_id = f"feedback_{user_id}_{int(time.time() * 1000)}"
        
        feedback_data = {
            'feedback_id': feedback_id,
            'user_id': user_id,
            'timestamp': time.time(),
            'detection_data': detection,
            'user_correction': user_correction,
            'feedback_type': feedback_type,  # 'false_positive', 'false_negative', 'correct', 'improve'
            'confidence_before': detection.get('confidence', 0.0),
            'confidence_after': 0.0  # Will be updated after retraining
        }
        
        # Store feedback
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detection_feedback 
            (feedback_id, user_id, timestamp, detection_data, user_correction, 
             feedback_type, confidence_before, confidence_after)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback_id, user_id, time.time(), json.dumps(detection),
            user_correction, feedback_type, feedback_data['confidence_before'], 0.0
        ))
        conn.commit()
        conn.close()
        
        # Update user learning statistics
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.learning_data['total_detections'] += 1
            if feedback_type in ['false_positive', 'false_negative']:
                profile.learning_data['accuracy_improvements'] += 1
            self._save_user_profile(profile)
    
    def adapt_to_environment(self, user_id: str, environment_data: Dict[str, Any]) -> Dict[str, float]:
        """Adapt detection parameters based on environmental conditions"""
        base_settings = self.get_personalized_settings(user_id, environment_data)
        adaptations = {}
        
        # Weather adaptations
        weather = environment_data.get('weather', 'clear')
        if weather in ['rain', 'snow', 'fog']:
            adaptations['confidence_threshold'] = max(0.3, base_settings['sensitivity'].get('car', 0.8) - 0.1)
            adaptations['detection_range'] = base_settings['alert_distance'] * 0.8
        
        # Lighting adaptations
        lighting = environment_data.get('lighting', 'day')
        if lighting in ['night', 'dawn', 'dusk']:
            adaptations['brightness_boost'] = 1.2
            adaptations['contrast_enhancement'] = 1.3
        
        # Traffic density adaptations
        traffic_density = environment_data.get('traffic_density', 'medium')
        if traffic_density == 'high':
            adaptations['update_frequency'] = 1.5  # Faster updates
            adaptations['max_objects'] = 20
        elif traffic_density == 'low':
            adaptations['update_frequency'] = 0.8  # Slower updates to save battery
            adaptations['max_objects'] = 10
        
        return adaptations
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings for new users"""
        return {
            'sensitivity': {
                'person': 0.8, 'bicycle': 0.8, 'motorcycle': 0.9,
                'car': 0.85, 'truck': 0.9, 'bus': 0.9
            },
            'alert_distance': 15.0,
            'voice_alerts': True,
            'haptic_feedback': True
        }
    
    def _save_user_profile(self, profile: UserProfile):
        """Save user profile to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_profiles 
            (user_id, mobility_mode, preferred_sensitivity, common_routes, 
             safety_preferences, learning_data, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile.user_id, profile.mobility_mode,
            json.dumps(profile.preferred_sensitivity),
            json.dumps(profile.common_routes),
            json.dumps(profile.safety_preferences),
            json.dumps(profile.learning_data),
            profile.created_at, profile.last_updated
        ))
        conn.commit()
        conn.close()
    
    def _save_behavior_pattern(self, pattern: BehaviorPattern):
        """Save behavior pattern to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO behavior_patterns 
            (pattern_id, user_id, pattern_type, pattern_data, confidence, frequency, last_observed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_id, pattern.user_id, pattern.pattern_type,
            json.dumps(pattern.pattern_data), pattern.confidence,
            pattern.frequency, pattern.last_observed
        ))
        conn.commit()
        conn.close()
    
    def _save_custom_object(self, custom_object: CustomObject):
        """Save custom object to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO custom_objects 
            (object_id, user_id, object_name, training_images, detection_confidence,
             false_positive_rate, last_trained, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            custom_object.object_id, custom_object.user_id, custom_object.object_name,
            json.dumps(custom_object.training_images), custom_object.detection_confidence,
            custom_object.false_positive_rate, custom_object.last_trained, custom_object.active
        ))
        conn.commit()
        conn.close()

# Global learning service instance
learning_service = LearningService()