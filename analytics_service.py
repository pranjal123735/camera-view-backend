"""
Advanced Analytics Service for Car Vision System
Provides comprehensive performance metrics, AI insights, and safety analytics.
"""

import json
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path

@dataclass
class PerformanceMetrics:
    detection_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    average_confidence: float
    processing_latency_ms: float
    fps: float
    ai_enhancement_rate: float

@dataclass
class SafetyMetrics:
    total_detections: int
    near_misses: int
    danger_events: int
    caution_events: int
    safety_score: float
    risk_trend: str
    most_common_threats: List[str]

@dataclass
class AIEnhancementStats:
    rag_corrections: int
    knowledge_graph_validations: int
    ensemble_consensus_rate: float
    temporal_consistency_improvements: int
    misclassification_fixes: int
    confidence_adjustments: int

@dataclass
class TripAnalytics:
    trip_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_distance_km: float
    average_speed_kmh: float
    safety_events: List[Dict]
    route_safety_score: float
    weather_conditions: str
    lighting_conditions: str

class AnalyticsService:
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.performance_history = deque(maxlen=1000)
        self.detection_history = deque(maxlen=5000)
        self.ai_stats = defaultdict(int)
        self.safety_events = deque(maxlen=1000)
        self.trip_data = {}
        self.current_trip_id = None
        
        self._init_database()
    
    def _init_database(self):
        """Initialize analytics database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                detection_accuracy REAL,
                false_positive_rate REAL,
                false_negative_rate REAL,
                average_confidence REAL,
                processing_latency_ms REAL,
                fps REAL,
                ai_enhancement_rate REAL
            )
        """)
        
        # Detection events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                trip_id TEXT,
                object_type TEXT,
                confidence REAL,
                distance_m REAL,
                speed_kmh REAL,
                risk_percent REAL,
                ai_enhanced BOOLEAN,
                correction_type TEXT,
                scene_context TEXT
            )
        """)
        
        # Safety events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                trip_id TEXT,
                event_type TEXT,
                severity TEXT,
                object_type TEXT,
                distance_m REAL,
                speed_kmh REAL,
                risk_percent REAL,
                weather_conditions TEXT,
                lighting_conditions TEXT,
                location_lat REAL,
                location_lon REAL
            )
        """)
        
        # Trip analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trip_analytics (
                trip_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                total_distance_km REAL,
                average_speed_kmh REAL,
                safety_score REAL,
                weather_conditions TEXT,
                lighting_conditions TEXT,
                route_hash TEXT,
                user_id TEXT
            )
        """)
        
        # AI enhancement statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_enhancements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                enhancement_type TEXT,
                original_label TEXT,
                corrected_label TEXT,
                confidence_before REAL,
                confidence_after REAL,
                scene_context TEXT,
                success BOOLEAN
            )
        """)
        
        # User behavior patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency INTEGER,
                last_updated REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_trip(self, user_id: str = "default", location: Tuple[float, float] = None) -> str:
        """Start a new trip session"""
        trip_id = f"trip_{int(time.time())}_{user_id}"
        self.current_trip_id = trip_id
        
        trip_data = TripAnalytics(
            trip_id=trip_id,
            start_time=datetime.now(),
            end_time=None,
            total_distance_km=0.0,
            average_speed_kmh=0.0,
            safety_events=[],
            route_safety_score=100.0,
            weather_conditions="unknown",
            lighting_conditions="unknown"
        )
        
        self.trip_data[trip_id] = trip_data
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trip_analytics 
            (trip_id, start_time, weather_conditions, lighting_conditions)
            VALUES (?, ?, ?, ?)
        """, (trip_id, time.time(), "unknown", "unknown"))
        conn.commit()
        conn.close()
        
        return trip_id
    
    def end_trip(self, trip_id: str = None) -> TripAnalytics:
        """End current trip and return analytics"""
        if not trip_id:
            trip_id = self.current_trip_id
        
        if trip_id not in self.trip_data:
            return None
        
        trip = self.trip_data[trip_id]
        trip.end_time = datetime.now()
        
        # Calculate final metrics
        duration_hours = (trip.end_time - trip.start_time).total_seconds() / 3600
        if duration_hours > 0:
            trip.average_speed_kmh = trip.total_distance_km / duration_hours
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE trip_analytics 
            SET end_time = ?, total_distance_km = ?, average_speed_kmh = ?, safety_score = ?
            WHERE trip_id = ?
        """, (time.time(), trip.total_distance_km, trip.average_speed_kmh, 
              trip.route_safety_score, trip_id))
        conn.commit()
        conn.close()
        
        return trip
    
    def record_detection(self, detection: Dict, ai_enhanced: bool = False, 
                        correction_type: str = None, scene_context: str = "unknown"):
        """Record a detection event for analytics"""
        timestamp = time.time()
        
        # Add to memory
        self.detection_history.append({
            'timestamp': timestamp,
            'detection': detection,
            'ai_enhanced': ai_enhanced,
            'correction_type': correction_type,
            'scene_context': scene_context
        })
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detection_events 
            (timestamp, trip_id, object_type, confidence, distance_m, speed_kmh, 
             risk_percent, ai_enhanced, correction_type, scene_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, self.current_trip_id, detection.get('label', 'unknown'),
            detection.get('confidence', 0.0), detection.get('distance_m', 0.0),
            detection.get('speed_kmh', 0.0), detection.get('risk_percent', 0.0),
            ai_enhanced, correction_type, scene_context
        ))
        conn.commit()
        conn.close()
    
    def record_safety_event(self, event_type: str, severity: str, detection: Dict,
                           weather: str = "unknown", lighting: str = "unknown",
                           location: Tuple[float, float] = None):
        """Record a safety event (near miss, danger, etc.)"""
        timestamp = time.time()
        
        event_data = {
            'timestamp': timestamp,
            'trip_id': self.current_trip_id,
            'event_type': event_type,
            'severity': severity,
            'detection': detection,
            'weather': weather,
            'lighting': lighting,
            'location': location
        }
        
        self.safety_events.append(event_data)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO safety_events 
            (timestamp, trip_id, event_type, severity, object_type, distance_m, 
             speed_kmh, risk_percent, weather_conditions, lighting_conditions,
             location_lat, location_lon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, self.current_trip_id, event_type, severity,
            detection.get('label', 'unknown'), detection.get('distance_m', 0.0),
            detection.get('speed_kmh', 0.0), detection.get('risk_percent', 0.0),
            weather, lighting,
            location[0] if location else None,
            location[1] if location else None
        ))
        conn.commit()
        conn.close()
    
    def record_ai_enhancement(self, enhancement_type: str, original_label: str,
                             corrected_label: str, confidence_before: float,
                             confidence_after: float, scene_context: str = "unknown",
                             success: bool = True):
        """Record AI enhancement statistics"""
        timestamp = time.time()
        self.ai_stats[enhancement_type] += 1
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ai_enhancements 
            (timestamp, enhancement_type, original_label, corrected_label,
             confidence_before, confidence_after, scene_context, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, enhancement_type, original_label, corrected_label,
            confidence_before, confidence_after, scene_context, success
        ))
        conn.commit()
        conn.close()
    
    def get_performance_metrics(self, hours: int = 24) -> PerformanceMetrics:
        """Get performance metrics for the specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent performance data
        cursor.execute("""
            SELECT AVG(detection_accuracy), AVG(false_positive_rate), 
                   AVG(false_negative_rate), AVG(average_confidence),
                   AVG(processing_latency_ms), AVG(fps), AVG(ai_enhancement_rate)
            FROM performance_metrics 
            WHERE timestamp > ?
        """, (cutoff_time,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] is not None:
            return PerformanceMetrics(
                detection_accuracy=result[0] or 0.85,
                false_positive_rate=result[1] or 0.05,
                false_negative_rate=result[2] or 0.08,
                average_confidence=result[3] or 0.75,
                processing_latency_ms=result[4] or 150.0,
                fps=result[5] or 15.0,
                ai_enhancement_rate=result[6] or 0.25
            )
        
        # Default metrics if no data
        return PerformanceMetrics(
            detection_accuracy=0.85,
            false_positive_rate=0.05,
            false_negative_rate=0.08,
            average_confidence=0.75,
            processing_latency_ms=150.0,
            fps=15.0,
            ai_enhancement_rate=0.25
        )
    
    def get_safety_metrics(self, hours: int = 24) -> SafetyMetrics:
        """Get safety metrics for the specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count safety events
        cursor.execute("""
            SELECT COUNT(*) FROM safety_events WHERE timestamp > ?
        """, (cutoff_time,))
        total_events = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM safety_events 
            WHERE timestamp > ? AND event_type = 'near_miss'
        """, (cutoff_time,))
        near_misses = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM safety_events 
            WHERE timestamp > ? AND severity = 'DANGER'
        """, (cutoff_time,))
        danger_events = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM safety_events 
            WHERE timestamp > ? AND severity = 'CAUTION'
        """, (cutoff_time,))
        caution_events = cursor.fetchone()[0] or 0
        
        # Get most common threats
        cursor.execute("""
            SELECT object_type, COUNT(*) as count 
            FROM safety_events 
            WHERE timestamp > ? 
            GROUP BY object_type 
            ORDER BY count DESC 
            LIMIT 5
        """, (cutoff_time,))
        threats = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # Calculate safety score (0-100)
        safety_score = max(0, 100 - (danger_events * 10) - (caution_events * 5) - (near_misses * 2))
        
        # Determine risk trend
        risk_trend = "stable"
        if danger_events > 5:
            risk_trend = "increasing"
        elif danger_events == 0 and caution_events < 3:
            risk_trend = "decreasing"
        
        return SafetyMetrics(
            total_detections=total_events,
            near_misses=near_misses,
            danger_events=danger_events,
            caution_events=caution_events,
            safety_score=safety_score,
            risk_trend=risk_trend,
            most_common_threats=threats
        )
    
    def get_ai_enhancement_stats(self, hours: int = 24) -> AIEnhancementStats:
        """Get AI enhancement statistics"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count different types of enhancements
        cursor.execute("""
            SELECT enhancement_type, COUNT(*) 
            FROM ai_enhancements 
            WHERE timestamp > ? 
            GROUP BY enhancement_type
        """, (cutoff_time,))
        
        enhancement_counts = dict(cursor.fetchall())
        
        # Calculate success rate
        cursor.execute("""
            SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)
            FROM ai_enhancements 
            WHERE timestamp > ?
        """, (cutoff_time,))
        
        success_rate = cursor.fetchone()[0] or 0.85
        
        conn.close()
        
        return AIEnhancementStats(
            rag_corrections=enhancement_counts.get('rag_correction', 0),
            knowledge_graph_validations=enhancement_counts.get('kg_validation', 0),
            ensemble_consensus_rate=success_rate,
            temporal_consistency_improvements=enhancement_counts.get('temporal_fix', 0),
            misclassification_fixes=enhancement_counts.get('misclass_fix', 0),
            confidence_adjustments=enhancement_counts.get('confidence_adj', 0)
        )
    
    def get_trip_analytics(self, trip_id: str = None) -> Optional[TripAnalytics]:
        """Get analytics for a specific trip"""
        if not trip_id:
            trip_id = self.current_trip_id
        
        if trip_id in self.trip_data:
            return self.trip_data[trip_id]
        
        # Try to load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trip_analytics WHERE trip_id = ?
        """, (trip_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return TripAnalytics(
                trip_id=result[0],
                start_time=datetime.fromtimestamp(result[1]),
                end_time=datetime.fromtimestamp(result[2]) if result[2] else None,
                total_distance_km=result[3] or 0.0,
                average_speed_kmh=result[4] or 0.0,
                safety_events=[],
                route_safety_score=result[5] or 100.0,
                weather_conditions=result[6] or "unknown",
                lighting_conditions=result[7] or "unknown"
            )
        
        return None
    
    def get_historical_trends(self, days: int = 7) -> Dict[str, List]:
        """Get historical trends for charts and graphs"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Daily safety scores
        cursor.execute("""
            SELECT DATE(timestamp, 'unixepoch') as date, 
                   AVG(CASE WHEN severity = 'DANGER' THEN 0 
                           WHEN severity = 'CAUTION' THEN 50 
                           ELSE 100 END) as daily_score
            FROM safety_events 
            WHERE timestamp > ?
            GROUP BY date
            ORDER BY date
        """, (cutoff_time,))
        
        safety_trend = [{'date': row[0], 'score': row[1]} for row in cursor.fetchall()]
        
        # Detection accuracy over time
        cursor.execute("""
            SELECT DATE(timestamp, 'unixepoch') as date, 
                   AVG(detection_accuracy) as accuracy
            FROM performance_metrics 
            WHERE timestamp > ?
            GROUP BY date
            ORDER BY date
        """, (cutoff_time,))
        
        accuracy_trend = [{'date': row[0], 'accuracy': row[1]} for row in cursor.fetchall()]
        
        # Object type distribution
        cursor.execute("""
            SELECT object_type, COUNT(*) as count
            FROM detection_events 
            WHERE timestamp > ?
            GROUP BY object_type
            ORDER BY count DESC
        """, (cutoff_time,))
        
        object_distribution = [{'type': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'safety_trend': safety_trend,
            'accuracy_trend': accuracy_trend,
            'object_distribution': object_distribution
        }
    
    def generate_safety_recommendations(self, user_id: str = "default") -> List[Dict]:
        """Generate personalized safety recommendations"""
        recommendations = []
        
        # Get recent safety metrics
        safety_metrics = self.get_safety_metrics(hours=168)  # Last week
        
        if safety_metrics.danger_events > 3:
            recommendations.append({
                'type': 'warning',
                'title': 'High Risk Activity Detected',
                'message': f'You\'ve had {safety_metrics.danger_events} danger events this week. Consider adjusting your route or travel times.',
                'priority': 'high'
            })
        
        if safety_metrics.safety_score < 70:
            recommendations.append({
                'type': 'improvement',
                'title': 'Safety Score Below Average',
                'message': f'Your safety score is {safety_metrics.safety_score:.0f}%. Try maintaining greater following distances.',
                'priority': 'medium'
            })
        
        # Check for common threat patterns
        if 'car' in safety_metrics.most_common_threats[:2]:
            recommendations.append({
                'type': 'tip',
                'title': 'Vehicle Interaction Safety',
                'message': 'Cars are your most common threat. Increase visibility with reflective gear and maintain predictable movements.',
                'priority': 'medium'
            })
        
        if not recommendations:
            recommendations.append({
                'type': 'positive',
                'title': 'Great Safety Record!',
                'message': 'You\'re maintaining excellent safety practices. Keep up the good work!',
                'priority': 'low'
            })
        
        return recommendations

# Global analytics service instance
analytics_service = AnalyticsService()