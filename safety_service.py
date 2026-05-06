"""
Advanced Safety Service for Car Vision System
Provides predictive collision warnings, emergency features, and safety analytics.
"""

import json
import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
import numpy as np
import sqlite3
from datetime import datetime, timedelta

@dataclass
class CollisionPrediction:
    object_id: int
    object_type: str
    time_to_collision: float
    collision_probability: float
    predicted_impact_point: Tuple[float, float]
    recommended_action: str
    severity: str

@dataclass
class SafetyAlert:
    alert_id: str
    timestamp: float
    alert_type: str
    severity: str
    message: str
    object_involved: Optional[Dict]
    recommended_actions: List[str]
    auto_dismiss_time: Optional[float]

@dataclass
class EmergencyEvent:
    event_id: str
    timestamp: float
    event_type: str
    severity: str
    location: Optional[Tuple[float, float]]
    speed_at_event: float
    objects_involved: List[Dict]
    emergency_contacts_notified: bool

class SafetyService:
    def __init__(self, db_path: str = "safety.db"):
        self.db_path = db_path
        self.active_alerts = {}
        self.collision_predictions = {}
        self.emergency_contacts = []
        self.safety_zones = []
        self.object_trajectories = defaultdict(deque)
        self.speed_history = deque(maxlen=100)
        self.location_history = deque(maxlen=1000)
        
        # Safety thresholds
        self.collision_time_threshold = 3.0  # seconds
        self.emergency_speed_threshold = 50.0  # km/h
        self.rapid_deceleration_threshold = -8.0  # m/s²
        
        self._init_database()
    
    def _init_database(self):
        """Initialize safety database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Safety alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp REAL,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                object_data TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time REAL
            )
        """)
        
        # Emergency events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emergency_events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL,
                event_type TEXT,
                severity TEXT,
                location_lat REAL,
                location_lon REAL,
                speed_kmh REAL,
                objects_data TEXT,
                contacts_notified BOOLEAN,
                response_time REAL
            )
        """)
        
        # Collision predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collision_predictions (
                prediction_id TEXT PRIMARY KEY,
                timestamp REAL,
                object_id INTEGER,
                object_type TEXT,
                time_to_collision REAL,
                collision_probability REAL,
                impact_point_x REAL,
                impact_point_y REAL,
                recommended_action TEXT,
                severity TEXT,
                actual_collision BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Safety zones table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_zones (
                zone_id TEXT PRIMARY KEY,
                name TEXT,
                center_lat REAL,
                center_lon REAL,
                radius_m REAL,
                zone_type TEXT,
                speed_limit REAL,
                special_rules TEXT
            )
        """)
        
        # Emergency contacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emergency_contacts (
                contact_id TEXT PRIMARY KEY,
                name TEXT,
                phone TEXT,
                email TEXT,
                relationship TEXT,
                priority INTEGER,
                notification_methods TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def predict_collision(self, detections: List[Dict], user_speed: float, 
                         user_location: Tuple[float, float] = None) -> List[CollisionPrediction]:
        """Predict potential collisions based on object trajectories"""
        predictions = []
        current_time = time.time()
        
        for detection in detections:
            if not detection.get('is_moving', False):
                continue
            
            object_id = detection.get('track_id')
            if not object_id:
                continue
            
            # Update trajectory history
            trajectory = self.object_trajectories[object_id]
            trajectory.append({
                'timestamp': current_time,
                'position': detection.get('bbox_xyxy', [0, 0, 0, 0]),
                'distance': detection.get('distance_m', 0),
                'speed': detection.get('speed_kmh', 0)
            })
            
            # Need at least 3 points for prediction
            if len(trajectory) < 3:
                continue
            
            # Calculate trajectory and predict collision
            collision_pred = self._calculate_collision_probability(
                trajectory, user_speed, detection
            )
            
            if collision_pred and collision_pred.collision_probability > 0.3:
                predictions.append(collision_pred)
                
                # Store prediction in database
                self._store_collision_prediction(collision_pred)
        
        return predictions
    
    def _calculate_collision_probability(self, trajectory: deque, user_speed: float, 
                                       detection: Dict) -> Optional[CollisionPrediction]:
        """Calculate collision probability based on trajectory analysis"""
        if len(trajectory) < 3:
            return None
        
        # Get recent trajectory points
        recent_points = list(trajectory)[-3:]
        
        # Calculate velocity and acceleration
        dt1 = recent_points[1]['timestamp'] - recent_points[0]['timestamp']
        dt2 = recent_points[2]['timestamp'] - recent_points[1]['timestamp']
        
        if dt1 <= 0 or dt2 <= 0:
            return None
        
        # Distance-based velocity calculation
        v1 = (recent_points[0]['distance'] - recent_points[1]['distance']) / dt1
        v2 = (recent_points[1]['distance'] - recent_points[2]['distance']) / dt2
        
        # Relative velocity (object approaching if positive)
        relative_velocity = v2 + (user_speed / 3.6)  # Convert km/h to m/s
        
        if relative_velocity <= 0:
            return None  # Objects moving away from each other
        
        # Time to collision
        current_distance = recent_points[-1]['distance']
        ttc = current_distance / relative_velocity
        
        if ttc > self.collision_time_threshold:
            return None
        
        # Calculate collision probability based on multiple factors
        distance_factor = max(0, 1 - (current_distance / 50.0))  # Higher prob at closer distances
        speed_factor = min(1, relative_velocity / 20.0)  # Higher prob at higher speeds
        trajectory_consistency = self._calculate_trajectory_consistency(trajectory)
        
        collision_probability = (distance_factor * 0.4 + 
                               speed_factor * 0.4 + 
                               trajectory_consistency * 0.2)
        
        # Determine severity and recommended action
        if ttc < 1.0:
            severity = "CRITICAL"
            action = "EMERGENCY_BRAKE"
        elif ttc < 2.0:
            severity = "HIGH"
            action = "BRAKE_NOW"
        else:
            severity = "MEDIUM"
            action = "PREPARE_TO_BRAKE"
        
        # Predict impact point (simplified)
        bbox = detection.get('bbox_xyxy', [0, 0, 0, 0])
        impact_point = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        return CollisionPrediction(
            object_id=detection.get('track_id'),
            object_type=detection.get('label', 'unknown'),
            time_to_collision=ttc,
            collision_probability=collision_probability,
            predicted_impact_point=impact_point,
            recommended_action=action,
            severity=severity
        )
    
    def _calculate_trajectory_consistency(self, trajectory: deque) -> float:
        """Calculate how consistent the object's trajectory is"""
        if len(trajectory) < 3:
            return 0.5
        
        distances = [point['distance'] for point in trajectory]
        
        # Calculate variance in distance changes
        distance_changes = []
        for i in range(1, len(distances)):
            distance_changes.append(distances[i-1] - distances[i])
        
        if not distance_changes:
            return 0.5
        
        variance = np.var(distance_changes)
        consistency = max(0, 1 - (variance / 10.0))  # Normalize variance
        
        return consistency
    
    def create_safety_alert(self, alert_type: str, severity: str, message: str,
                           object_involved: Dict = None, auto_dismiss_seconds: int = None) -> SafetyAlert:
        """Create a new safety alert"""
        alert_id = f"alert_{int(time.time() * 1000)}"
        timestamp = time.time()
        
        auto_dismiss_time = None
        if auto_dismiss_seconds:
            auto_dismiss_time = timestamp + auto_dismiss_seconds
        
        # Determine recommended actions based on alert type and severity
        recommended_actions = self._get_recommended_actions(alert_type, severity, object_involved)
        
        alert = SafetyAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            alert_type=alert_type,
            severity=severity,
            message=message,
            object_involved=object_involved,
            recommended_actions=recommended_actions,
            auto_dismiss_time=auto_dismiss_time
        )
        
        self.active_alerts[alert_id] = alert
        
        # Store in database
        self._store_safety_alert(alert)
        
        return alert
    
    def _get_recommended_actions(self, alert_type: str, severity: str, 
                                object_involved: Dict = None) -> List[str]:
        """Get recommended actions based on alert parameters"""
        actions = []
        
        if alert_type == "collision_warning":
            if severity == "CRITICAL":
                actions = ["EMERGENCY_BRAKE", "SOUND_HORN", "EVASIVE_MANEUVER"]
            elif severity == "HIGH":
                actions = ["BRAKE_NOW", "INCREASE_AWARENESS", "SOUND_HORN"]
            else:
                actions = ["PREPARE_TO_BRAKE", "MAINTAIN_DISTANCE", "STAY_ALERT"]
        
        elif alert_type == "blind_spot":
            actions = ["CHECK_MIRRORS", "SIGNAL_INTENTION", "WAIT_FOR_CLEAR"]
        
        elif alert_type == "speed_warning":
            actions = ["REDUCE_SPEED", "CHECK_SPEED_LIMIT", "INCREASE_FOLLOWING_DISTANCE"]
        
        elif alert_type == "weather_warning":
            actions = ["REDUCE_SPEED", "INCREASE_FOLLOWING_DISTANCE", "USE_LIGHTS"]
        
        elif alert_type == "fatigue_detection":
            actions = ["TAKE_BREAK", "FIND_SAFE_PARKING", "CALL_SOMEONE"]
        
        else:
            actions = ["STAY_ALERT", "ASSESS_SITUATION", "TAKE_APPROPRIATE_ACTION"]
        
        return actions
    
    def check_emergency_conditions(self, detections: List[Dict], user_speed: float,
                                  acceleration: float, location: Tuple[float, float] = None) -> Optional[EmergencyEvent]:
        """Check for emergency conditions and create emergency event if needed"""
        current_time = time.time()
        
        # Check for rapid deceleration (potential crash)
        if acceleration < self.rapid_deceleration_threshold:
            return self._create_emergency_event(
                "rapid_deceleration", "HIGH", location, user_speed, detections,
                f"Rapid deceleration detected: {acceleration:.1f} m/s²"
            )
        
        # Check for high-speed collision warnings
        collision_predictions = self.predict_collision(detections, user_speed, location)
        critical_predictions = [p for p in collision_predictions if p.severity == "CRITICAL"]
        
        if critical_predictions and user_speed > self.emergency_speed_threshold:
            return self._create_emergency_event(
                "imminent_collision", "CRITICAL", location, user_speed, detections,
                f"Imminent collision at {user_speed:.1f} km/h"
            )
        
        # Check for multiple simultaneous high-risk objects
        high_risk_objects = [d for d in detections if d.get('risk_percent', 0) > 80]
        if len(high_risk_objects) >= 3:
            return self._create_emergency_event(
                "multiple_threats", "HIGH", location, user_speed, detections,
                f"Multiple high-risk objects detected: {len(high_risk_objects)}"
            )
        
        return None
    
    def _create_emergency_event(self, event_type: str, severity: str, 
                               location: Tuple[float, float], speed: float,
                               objects: List[Dict], description: str) -> EmergencyEvent:
        """Create an emergency event and trigger notifications"""
        event_id = f"emergency_{int(time.time() * 1000)}"
        timestamp = time.time()
        
        event = EmergencyEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            severity=severity,
            location=location,
            speed_at_event=speed,
            objects_involved=objects,
            emergency_contacts_notified=False
        )
        
        # Store in database
        self._store_emergency_event(event)
        
        # Trigger emergency notifications if severity is high enough
        if severity in ["HIGH", "CRITICAL"]:
            asyncio.create_task(self._notify_emergency_contacts(event))
        
        return event
    
    async def _notify_emergency_contacts(self, event: EmergencyEvent):
        """Notify emergency contacts about the event"""
        # This would integrate with SMS/email services
        # For now, we'll just log the notification
        print(f"EMERGENCY NOTIFICATION: {event.event_type} - {event.severity}")
        print(f"Location: {event.location}")
        print(f"Speed: {event.speed_at_event} km/h")
        
        # Mark as notified
        event.emergency_contacts_notified = True
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE emergency_events 
            SET contacts_notified = TRUE, response_time = ?
            WHERE event_id = ?
        """, (time.time(), event.event_id))
        conn.commit()
        conn.close()
    
    def add_emergency_contact(self, name: str, phone: str, email: str = None,
                            relationship: str = "emergency", priority: int = 1) -> str:
        """Add an emergency contact"""
        contact_id = f"contact_{int(time.time())}"
        
        contact = {
            'contact_id': contact_id,
            'name': name,
            'phone': phone,
            'email': email,
            'relationship': relationship,
            'priority': priority,
            'notification_methods': json.dumps(['sms', 'call'] + (['email'] if email else []))
        }
        
        self.emergency_contacts.append(contact)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO emergency_contacts 
            (contact_id, name, phone, email, relationship, priority, notification_methods)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (contact_id, name, phone, email, relationship, priority, 
              contact['notification_methods']))
        conn.commit()
        conn.close()
        
        return contact_id
    
    def add_safety_zone(self, name: str, center: Tuple[float, float], 
                       radius_m: float, zone_type: str = "general",
                       speed_limit: float = None) -> str:
        """Add a safety zone (school zone, construction, etc.)"""
        zone_id = f"zone_{int(time.time())}"
        
        zone = {
            'zone_id': zone_id,
            'name': name,
            'center_lat': center[0],
            'center_lon': center[1],
            'radius_m': radius_m,
            'zone_type': zone_type,
            'speed_limit': speed_limit,
            'special_rules': json.dumps([])
        }
        
        self.safety_zones.append(zone)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO safety_zones 
            (zone_id, name, center_lat, center_lon, radius_m, zone_type, speed_limit, special_rules)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (zone_id, name, center[0], center[1], radius_m, zone_type, speed_limit, zone['special_rules']))
        conn.commit()
        conn.close()
        
        return zone_id
    
    def check_safety_zones(self, location: Tuple[float, float]) -> List[Dict]:
        """Check if current location is in any safety zones"""
        active_zones = []
        
        for zone in self.safety_zones:
            # Calculate distance to zone center (simplified)
            lat_diff = location[0] - zone['center_lat']
            lon_diff = location[1] - zone['center_lon']
            distance_m = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111000  # Rough conversion
            
            if distance_m <= zone['radius_m']:
                active_zones.append(zone)
        
        return active_zones
    
    def _store_safety_alert(self, alert: SafetyAlert):
        """Store safety alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO safety_alerts 
            (alert_id, timestamp, alert_type, severity, message, object_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id, alert.timestamp, alert.alert_type, 
            alert.severity, alert.message, 
            json.dumps(alert.object_involved) if alert.object_involved else None
        ))
        conn.commit()
        conn.close()
    
    def _store_collision_prediction(self, prediction: CollisionPrediction):
        """Store collision prediction in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO collision_predictions 
            (prediction_id, timestamp, object_id, object_type, time_to_collision,
             collision_probability, impact_point_x, impact_point_y, 
             recommended_action, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"pred_{int(time.time() * 1000)}", time.time(),
            prediction.object_id, prediction.object_type, prediction.time_to_collision,
            prediction.collision_probability, prediction.predicted_impact_point[0],
            prediction.predicted_impact_point[1], prediction.recommended_action,
            prediction.severity
        ))
        conn.commit()
        conn.close()
    
    def _store_emergency_event(self, event: EmergencyEvent):
        """Store emergency event in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO emergency_events 
            (event_id, timestamp, event_type, severity, location_lat, location_lon,
             speed_kmh, objects_data, contacts_notified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id, event.timestamp, event.event_type, event.severity,
            event.location[0] if event.location else None,
            event.location[1] if event.location else None,
            event.speed_at_event, json.dumps(event.objects_involved),
            event.emergency_contacts_notified
        ))
        conn.commit()
        conn.close()
    
    def get_active_alerts(self) -> List[SafetyAlert]:
        """Get all active safety alerts"""
        current_time = time.time()
        active = []
        
        for alert_id, alert in list(self.active_alerts.items()):
            # Check if alert should be auto-dismissed
            if alert.auto_dismiss_time and current_time > alert.auto_dismiss_time:
                del self.active_alerts[alert_id]
                continue
            
            active.append(alert)
        
        return active
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss a safety alert"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE safety_alerts 
                SET resolved = TRUE, resolution_time = ?
                WHERE alert_id = ?
            """, (time.time(), alert_id))
            conn.commit()
            conn.close()
            
            return True
        
        return False

# Global safety service instance
safety_service = SafetyService()