"""
Universal AI Detection Examples
Demonstrates how to use the AI-enhanced detection system for various domains
"""

import asyncio
import numpy as np
from PIL import Image
import requests
import json
import time

from universal_ai_detection import (
    create_general_detection_system,
    create_security_detection_system,
    create_medical_detection_system,
    create_retail_detection_system,
    UniversalDetectionConfig,
    DetectionDomain
)

def create_sample_detections(scenario: str) -> list:
    """Create sample detection data for different scenarios"""
    
    scenarios = {
        "office": [
            {"label": "person", "confidence": 0.85, "bbox_xyxy": [300, 100, 500, 400]},
            {"label": "laptop", "confidence": 0.92, "bbox_xyxy": [200, 300, 400, 380]},
            {"label": "chair", "confidence": 0.78, "bbox_xyxy": [150, 200, 250, 350]},
            {"label": "cell_phone", "confidence": 0.65, "bbox_xyxy": [450, 320, 470, 340]}
        ],
        
        "security_checkpoint": [
            {"label": "person", "confidence": 0.88, "bbox_xyxy": [200, 50, 400, 450]},
            {"label": "backpack", "confidence": 0.82, "bbox_xyxy": [100, 200, 180, 300]},
            {"label": "laptop", "confidence": 0.75, "bbox_xyxy": [300, 350, 450, 420]},
            {"label": "bottle", "confidence": 0.70, "bbox_xyxy": [50, 300, 80, 380]}
        ],
        
        "medical_room": [
            {"label": "person", "confidence": 0.90, "bbox_xyxy": [250, 80, 450, 420]},
            {"label": "syringe", "confidence": 0.85, "bbox_xyxy": [100, 250, 130, 280]},
            {"label": "bottle", "confidence": 0.78, "bbox_xyxy": [500, 200, 530, 260]},
            {"label": "chair", "confidence": 0.72, "bbox_xyxy": [50, 300, 150, 450]}
        ],
        
        "retail_store": [
            {"label": "person", "confidence": 0.87, "bbox_xyxy": [300, 100, 450, 450]},
            {"label": "product", "confidence": 0.83, "bbox_xyxy": [100, 200, 150, 250]},
            {"label": "shopping_cart", "confidence": 0.79, "bbox_xyxy": [500, 250, 600, 400]},
            {"label": "bottle", "confidence": 0.74, "bbox_xyxy": [200, 300, 230, 360]}
        ],
        
        "outdoor_scene": [
            {"label": "person", "confidence": 0.82, "bbox_xyxy": [200, 150, 300, 400]},
            {"label": "car", "confidence": 0.89, "bbox_xyxy": [400, 200, 600, 350]},
            {"label": "dog", "confidence": 0.76, "bbox_xyxy": [150, 350, 220, 420]},
            {"label": "bicycle", "confidence": 0.71, "bbox_xyxy": [50, 180, 150, 320]}
        ],
        
        "wildlife_monitoring": [
            {"label": "deer", "confidence": 0.88, "bbox_xyxy": [300, 200, 450, 380]},
            {"label": "bird", "confidence": 0.72, "bbox_xyxy": [100, 50, 140, 90]},
            {"label": "tree", "confidence": 0.85, "bbox_xyxy": [500, 0, 640, 400]},
            {"label": "person", "confidence": 0.65, "bbox_xyxy": [50, 250, 120, 420]}
        ]
    }
    
    return scenarios.get(scenario, [])

def create_sample_diagnostics(lighting: str = "normal") -> dict:
    """Create sample frame diagnostics"""
    
    diagnostics_map = {
        "normal": {
            "brightness_01": 0.6,
            "contrast": 0.4,
            "low_light": False,
            "low_contrast": False,
            "overexposed": False,
            "quality_score": 0.8
        },
        "low_light": {
            "brightness_01": 0.2,
            "contrast": 0.3,
            "low_light": True,
            "low_contrast": True,
            "overexposed": False,
            "quality_score": 0.4
        },
        "bright": {
            "brightness_01": 0.9,
            "contrast": 0.5,
            "low_light": False,
            "low_contrast": False,
            "overexposed": True,
            "quality_score": 0.7
        }
    }
    
    return diagnostics_map.get(lighting, diagnostics_map["normal"])

async def example_general_detection():
    """Example: General object detection in office environment"""
    
    print("\n🏢 Example 1: General Office Detection")
    print("=" * 50)
    
    # Create general detection system
    system = create_general_detection_system(
        confidence_threshold=0.3,
        enable_rag=True,
        enable_knowledge_graph=True
    )
    
    # Sample office scene
    detections = create_sample_detections("office")
    diagnostics = create_sample_diagnostics("normal")
    
    print(f"Original detections: {len(detections)}")
    for det in detections:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
    
    # Process through AI pipeline
    result = await system.process_detections(detections, diagnostics)
    
    print(f"\nAI-enhanced detections: {len(result['detections'])}")
    for det in result['detections']:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
        if 'rag_enhancement' in det:
            print(f"    Enhancement: {det['rag_enhancement']['scene_context']}")
        if 'correction_applied' in det:
            print(f"    Correction: {det['correction_applied']}")
    
    print(f"\nProcessing time: {result['processing_time_ms']:.2f}ms")
    print(f"Pipeline steps: {', '.join(result['pipeline_steps'])}")
    
    return result

async def example_security_detection():
    """Example: Security checkpoint detection"""
    
    print("\n🔒 Example 2: Security Checkpoint Detection")
    print("=" * 50)
    
    # Create security detection system
    system = create_security_detection_system(
        confidence_threshold=0.2,  # Lower threshold for security
        enable_rag=True,
        enable_knowledge_graph=True
    )
    
    # Sample security checkpoint scene
    detections = create_sample_detections("security_checkpoint")
    diagnostics = create_sample_diagnostics("normal")
    
    print(f"Security scan detections: {len(detections)}")
    for det in detections:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
    
    # Process through security AI pipeline
    result = await system.process_detections(detections, diagnostics)
    
    print(f"\nSecurity-enhanced detections: {len(result['detections'])}")
    for det in result['detections']:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
        if 'exclusion_warning' in det:
            print(f"    ⚠️  Warning: {det['exclusion_warning']}")
        if 'potential_confusions' in det:
            print(f"    Potential confusions: {det['potential_confusions']}")
    
    print(f"\nSecurity processing time: {result['processing_time_ms']:.2f}ms")
    
    return result

async def example_medical_detection():
    """Example: Medical equipment detection"""
    
    print("\n🏥 Example 3: Medical Equipment Detection")
    print("=" * 50)
    
    # Create medical detection system
    system = create_medical_detection_system(
        confidence_threshold=0.4,  # Higher threshold for medical accuracy
        enable_rag=True,
        enable_knowledge_graph=True
    )
    
    # Sample medical room scene
    detections = create_sample_detections("medical_room")
    diagnostics = create_sample_diagnostics("bright")  # Well-lit medical environment
    
    print(f"Medical room detections: {len(detections)}")
    for det in detections:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
    
    # Process through medical AI pipeline
    result = await system.process_detections(detections, diagnostics)
    
    print(f"\nMedical-enhanced detections: {len(result['detections'])}")
    for det in result['detections']:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
        if 'rag_enhancement' in det:
            enhancement = det['rag_enhancement']
            print(f"    Medical context: {enhancement['scene_context']}")
            print(f"    Confidence modifier: {enhancement['confidence_modifier']:.2f}x")
    
    print(f"\nMedical processing time: {result['processing_time_ms']:.2f}ms")
    
    return result

async def example_retail_detection():
    """Example: Retail analytics detection"""
    
    print("\n🛒 Example 4: Retail Analytics Detection")
    print("=" * 50)
    
    # Create retail detection system
    system = create_retail_detection_system(
        confidence_threshold=0.25,
        enable_rag=True,
        enable_knowledge_graph=True
    )
    
    # Sample retail store scene
    detections = create_sample_detections("retail_store")
    diagnostics = create_sample_diagnostics("normal")
    
    print(f"Retail store detections: {len(detections)}")
    for det in detections:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
    
    # Process through retail AI pipeline
    result = await system.process_detections(detections, diagnostics)
    
    print(f"\nRetail-enhanced detections: {len(result['detections'])}")
    for det in result['detections']:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
        if 'rag_enhancement' in det:
            enhancement = det['rag_enhancement']
            print(f"    Retail context: {enhancement['scene_context']}")
    
    print(f"\nRetail processing time: {result['processing_time_ms']:.2f}ms")
    
    return result

async def example_comparison_study():
    """Example: Compare different domains on same detection data"""
    
    print("\n📊 Example 5: Cross-Domain Comparison")
    print("=" * 50)
    
    # Use office scene for all domains
    base_detections = create_sample_detections("office")
    diagnostics = create_sample_diagnostics("normal")
    
    print(f"Base detections: {[d['label'] for d in base_detections]}")
    
    # Test with different domain systems
    domains = {
        "General": create_general_detection_system(),
        "Security": create_security_detection_system(),
        "Medical": create_medical_detection_system(),
        "Retail": create_retail_detection_system()
    }
    
    results = {}
    
    for domain_name, system in domains.items():
        result = await system.process_detections(base_detections.copy(), diagnostics)
        results[domain_name] = result
        
        print(f"\n{domain_name} Domain Results:")
        print(f"  Processing time: {result['processing_time_ms']:.2f}ms")
        print(f"  Final detections: {len(result['detections'])}")
        
        for det in result['detections']:
            original_conf = next((d['confidence'] for d in base_detections 
                                if d['label'] == det['label']), 0)
            change = det['confidence'] - original_conf
            change_str = f"({change:+.2f})" if change != 0 else ""
            print(f"    - {det['label']}: {det['confidence']:.2f} {change_str}")
    
    return results

def example_api_usage():
    """Example: Using the universal API endpoints"""
    
    print("\n🌐 Example 6: Universal API Usage")
    print("=" * 50)
    
    # Example API calls (assuming server is running)
    base_url = "http://localhost:8000"
    
    print("Available API endpoints:")
    print(f"  GET  {base_url}/")
    print(f"  GET  {base_url}/health")
    print(f"  GET  {base_url}/domains")
    print(f"  GET  {base_url}/domains/{{domain}}")
    print(f"  POST {base_url}/detect")
    print(f"  POST {base_url}/detect/{{domain}}")
    print(f"  POST {base_url}/batch-detect/{{domain}}")
    print(f"  GET  {base_url}/stats")
    
    print("\nExample usage:")
    print("""
    # General detection
    curl -X POST "http://localhost:8000/detect" \\
         -F "file=@image.jpg" \\
         -F "confidence=0.3" \\
         -F "enable_rag=true"
    
    # Security detection
    curl -X POST "http://localhost:8000/detect/security" \\
         -F "file=@security_cam.jpg" \\
         -F "confidence=0.2"
    
    # Medical detection
    curl -X POST "http://localhost:8000/detect/medical" \\
         -F "file=@xray.jpg" \\
         -F "confidence=0.4"
    
    # Get system stats
    curl "http://localhost:8000/stats"
    """)

async def example_performance_benchmark():
    """Example: Performance benchmarking across domains"""
    
    print("\n⚡ Example 7: Performance Benchmark")
    print("=" * 50)
    
    # Create test scenarios
    scenarios = [
        ("office", "normal"),
        ("security_checkpoint", "low_light"),
        ("medical_room", "bright"),
        ("retail_store", "normal"),
        ("outdoor_scene", "normal")
    ]
    
    systems = {
        "general": create_general_detection_system(),
        "security": create_security_detection_system(),
        "medical": create_medical_detection_system(),
        "retail": create_retail_detection_system()
    }
    
    print("Benchmarking AI enhancement performance...")
    
    benchmark_results = {}
    
    for system_name, system in systems.items():
        times = []
        
        for scenario, lighting in scenarios:
            detections = create_sample_detections(scenario)
            diagnostics = create_sample_diagnostics(lighting)
            
            # Run multiple times for average
            for _ in range(5):
                start_time = time.time()
                await system.process_detections(detections, diagnostics)
                times.append((time.time() - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        benchmark_results[system_name] = {
            "avg_time_ms": avg_time,
            "std_time_ms": std_time,
            "min_time_ms": min(times),
            "max_time_ms": max(times)
        }
        
        print(f"{system_name.capitalize()} System:")
        print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"  Range: {min(times):.2f}ms - {max(times):.2f}ms")
    
    return benchmark_results

async def main():
    """Run all examples"""
    
    print("🤖 Universal AI-Enhanced Detection Examples")
    print("=" * 60)
    print("Demonstrating AI enhancements for multiple detection domains")
    
    try:
        # Run all examples
        await example_general_detection()
        await example_security_detection()
        await example_medical_detection()
        await example_retail_detection()
        await example_comparison_study()
        example_api_usage()
        await example_performance_benchmark()
        
        print("\n✅ All Universal Detection Examples Completed!")
        print("\n🎯 Key Takeaways:")
        print("  • AI enhancements work for ANY detection domain")
        print("  • RAG provides contextual knowledge for each domain")
        print("  • Knowledge graphs adapt to domain-specific relationships")
        print("  • Performance remains real-time across all domains")
        print("  • Universal API supports any YOLO model")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())