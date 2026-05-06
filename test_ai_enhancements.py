#!/usr/bin/env python3
"""
Test script for AI-enhanced detection system
Demonstrates RAG, Knowledge Graph, and Ensemble capabilities
"""

import asyncio
import numpy as np
from PIL import Image
import json
import time

# Import our AI enhancement modules
from rag_enhanced_detection import rag_enhancer
from knowledge_graph import knowledge_graph
from ensemble_detection import ensemble_system

def create_test_detections():
    """Create test detection data for demonstration"""
    return [
        {
            "label": "truck",
            "confidence": 0.6,
            "bbox_xyxy": [400, 200, 800, 600],  # Large object in center
            "track_id": 1
        },
        {
            "label": "car", 
            "confidence": 0.4,
            "bbox_xyxy": [50, 50, 150, 100],    # Small object
            "track_id": 2
        },
        {
            "label": "laptop",
            "confidence": 0.8,
            "bbox_xyxy": [300, 400, 500, 500],  # Indoor object
            "track_id": 3
        }
    ]

def create_test_diagnostics():
    """Create test frame diagnostics"""
    return {
        "brightness_01": 0.3,
        "low_light": True,
        "low_contrast": True,
        "glare_risk": False,
        "quality_hint": "low_light"
    }

async def test_rag_enhancement():
    """Test RAG-enhanced detection"""
    print("\n🧠 Testing RAG Enhancement...")
    
    detections = create_test_detections()
    diagnostics = create_test_diagnostics()
    
    print(f"Original detections: {len(detections)}")
    for det in detections:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
    
    # Apply RAG enhancement
    enhanced = rag_enhancer.enhance_detections_with_rag(detections, diagnostics)
    
    print(f"\nRAG-enhanced detections: {len(enhanced)}")
    for det in enhanced:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
        if "rag_metadata" in det:
            print(f"    Context score: {det['rag_metadata']['context_score']:.2f}")
            if det['rag_metadata']['suggestions']:
                print(f"    Suggestions: {det['rag_metadata']['suggestions']}")
    
    return enhanced

def test_knowledge_graph():
    """Test knowledge graph analysis"""
    print("\n🕸️ Testing Knowledge Graph...")
    
    detections = create_test_detections()
    scene_context = "indoor"  # Indoor scene with laptop
    
    print(f"Scene context: {scene_context}")
    print(f"Detections to analyze: {[d['label'] for d in detections]}")
    
    # Analyze with knowledge graph
    analysis = knowledge_graph.analyze_detection_context(detections, scene_context)
    
    print(f"\nContext consistency scores:")
    for obj, score in analysis["context_consistency"].items():
        print(f"  - {obj}: {score:.2f}")
    
    print(f"\nConfidence adjustments:")
    for obj, adj in analysis["confidence_adjustments"].items():
        print(f"  - {obj}: {adj:.2f}x")
    
    if analysis["relationship_violations"]:
        print(f"\nRelationship violations:")
        for violation in analysis["relationship_violations"]:
            print(f"  - {violation['type']}: {violation['object1']} vs {violation['object2']}")
    
    if analysis["suggested_corrections"]:
        print(f"\nSuggested corrections:")
        for correction in analysis["suggested_corrections"]:
            print(f"  - {correction['action']}: {correction.get('reason', 'N/A')}")
    
    return analysis

async def test_ensemble_detection():
    """Test ensemble detection system"""
    print("\n🎯 Testing Ensemble Detection...")
    
    # Create a dummy frame
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    diagnostics = create_test_diagnostics()
    
    print("Running ensemble detection...")
    
    try:
        # Note: This would normally use a real YOLO model
        # For testing, we'll simulate the ensemble process
        
        # Test temporal consistency
        detections = create_test_detections()
        
        print("Testing temporal consistency...")
        for det in detections:
            track_id = det["track_id"]
            ensemble_system.temporal_tracker.update_object_history(
                track_id, det["label"], det["confidence"], det["bbox_xyxy"]
            )
        
        # Get temporal consensus
        for det in detections:
            track_id = det["track_id"]
            consensus_label, consensus_conf = ensemble_system.temporal_tracker.get_temporal_consensus(track_id)
            print(f"  Track {track_id}: {consensus_label} ({consensus_conf:.2f})")
        
        # Test adaptive learning
        print("\nTesting adaptive learning...")
        adaptive_results = ensemble_system.adaptive_learner.get_adaptive_corrections(detections)
        
        print(f"Adaptive corrections applied: {len(adaptive_results)}")
        for det in adaptive_results:
            if "correction_applied" in det:
                print(f"  - {det['label']}: {det['correction_applied']}")
        
        return adaptive_results
        
    except Exception as e:
        print(f"Ensemble test error (expected without real model): {e}")
        return detections

def test_integration():
    """Test full AI integration pipeline"""
    print("\n🚀 Testing Full AI Integration...")
    
    detections = create_test_detections()
    diagnostics = create_test_diagnostics()
    
    print("Step 1: Original YOLO detections")
    print(f"  Objects: {[d['label'] for d in detections]}")
    
    print("\nStep 2: RAG enhancement")
    rag_enhanced = rag_enhancer.enhance_detections_with_rag(detections, diagnostics)
    print(f"  Enhanced objects: {[d['label'] for d in rag_enhanced]}")
    
    print("\nStep 3: Knowledge graph analysis")
    scene_context = "indoor"
    kg_analysis = knowledge_graph.analyze_detection_context(rag_enhanced, scene_context)
    
    # Apply KG corrections
    kg_corrected = []
    for detection in rag_enhanced:
        corrected = detection.copy()
        obj_type = detection.get("label", "")
        
        # Apply confidence adjustments
        if obj_type in kg_analysis["confidence_adjustments"]:
            adj_factor = kg_analysis["confidence_adjustments"][obj_type]
            corrected["confidence"] = min(0.99, corrected["confidence"] * adj_factor)
        
        # Apply suggested corrections
        for suggestion in kg_analysis["suggested_corrections"]:
            if (suggestion["action"] == "relabel_detection" and 
                suggestion["current_label"] == obj_type and
                suggestion["confidence"] > 0.7):
                corrected["label"] = suggestion["suggested_label"]
                corrected["kg_correction"] = suggestion["reason"]
        
        kg_corrected.append(corrected)
    
    print(f"  KG-corrected objects: {[d['label'] for d in kg_corrected]}")
    
    print("\nStep 4: Final results")
    for det in kg_corrected:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
        if "kg_correction" in det:
            print(f"    Correction: {det['kg_correction']}")
    
    return kg_corrected

def test_performance():
    """Test performance of AI enhancements"""
    print("\n⚡ Testing Performance...")
    
    detections = create_test_detections()
    diagnostics = create_test_diagnostics()
    
    # Test RAG performance
    start_time = time.time()
    for _ in range(10):
        rag_enhancer.enhance_detections_with_rag(detections, diagnostics)
    rag_time = (time.time() - start_time) / 10
    
    # Test KG performance
    start_time = time.time()
    for _ in range(10):
        knowledge_graph.analyze_detection_context(detections, "indoor")
    kg_time = (time.time() - start_time) / 10
    
    print(f"RAG enhancement: {rag_time*1000:.2f}ms per frame")
    print(f"Knowledge graph: {kg_time*1000:.2f}ms per frame")
    print(f"Total AI overhead: {(rag_time + kg_time)*1000:.2f}ms per frame")
    
    return rag_time + kg_time

async def main():
    """Main test function"""
    print("🤖 AI-Enhanced Detection System Test")
    print("=" * 50)
    
    try:
        # Test individual components
        await test_rag_enhancement()
        test_knowledge_graph()
        await test_ensemble_detection()
        
        # Test integration
        test_integration()
        
        # Test performance
        overhead = test_performance()
        
        print("\n✅ All AI Enhancement Tests Completed!")
        print(f"Total processing overhead: {overhead*1000:.2f}ms")
        print("System ready for enhanced detection!")
        
        # Get system insights
        print("\n📊 System Insights:")
        insights = rag_enhancer.get_learning_insights()
        print(f"Learning patterns stored: {insights['total_patterns']}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())