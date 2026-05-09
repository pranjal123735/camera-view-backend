#!/usr/bin/env python3
"""
Performance comparison between heavy AI system and ultra-fast system
"""

import time
import requests
import json
from pathlib import Path
import numpy as np
from PIL import Image
import io

def create_test_image():
    """Create a test image for performance testing"""
    # Create a simple test image with some shapes
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some rectangular shapes to simulate cars/objects
    img[100:200, 150:300] = [255, 0, 0]  # Red rectangle
    img[250:350, 400:550] = [0, 255, 0]  # Green rectangle
    
    pil_img = Image.fromarray(img)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_endpoint_performance(url, image_data, num_tests=5):
    """Test endpoint performance"""
    
    print(f"Testing {url}...")
    
    times = []
    successful_requests = 0
    
    for i in range(num_tests):
        try:
            start_time = time.time()
            
            files = {'file': ('test.jpg', image_data, 'image/jpeg')}
            response = requests.post(url, files=files, timeout=60)
            
            end_time = time.time()
            
            if response.status_code == 200:
                latency_ms = (end_time - start_time) * 1000
                times.append(latency_ms)
                successful_requests += 1
                print(f"  Test {i+1}: {latency_ms:.1f}ms")
            else:
                print(f"  Test {i+1}: Failed (HTTP {response.status_code})")
                
        except requests.exceptions.Timeout:
            print(f"  Test {i+1}: Timeout (>60s)")
        except Exception as e:
            print(f"  Test {i+1}: Error - {e}")
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        return {
            "avg_ms": avg_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "successful_requests": successful_requests,
            "total_requests": num_tests,
            "success_rate": successful_requests / num_tests * 100
        }
    else:
        return {
            "avg_ms": None,
            "min_ms": None,
            "max_ms": None,
            "successful_requests": 0,
            "total_requests": num_tests,
            "success_rate": 0
        }

def main():
    print("🏁 Car Vision Performance Comparison")
    print("=" * 60)
    
    # Create test image
    print("Creating test image...")
    test_image = create_test_image()
    print(f"✅ Test image created ({len(test_image)} bytes)")
    
    # Test configurations
    endpoints = [
        {
            "name": "Current Heavy AI System",
            "url": "http://localhost:8000/analyze-image",
            "port": 8000
        },
        {
            "name": "Ultra-Fast Optimized System", 
            "url": "http://localhost:8001/analyze-image",
            "port": 8001
        }
    ]
    
    results = {}
    
    for endpoint in endpoints:
        print(f"\n🧪 Testing: {endpoint['name']}")
        print("-" * 40)
        
        # Check if server is running
        try:
            health_response = requests.get(f"http://localhost:{endpoint['port']}/health", timeout=5)
            if health_response.status_code != 200:
                print(f"❌ Server not running on port {endpoint['port']}")
                continue
        except:
            print(f"❌ Server not running on port {endpoint['port']}")
            print(f"   Start with: python main.py (port 8000) or python main_optimized.py (port 8001)")
            continue
        
        # Run performance test
        result = test_endpoint_performance(endpoint['url'], test_image)
        results[endpoint['name']] = result
    
    # Display comparison
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if len(results) >= 2:
        heavy_system = list(results.values())[0]
        fast_system = list(results.values())[1]
        
        print(f"{'Metric':<25} {'Heavy AI':<15} {'Ultra-Fast':<15} {'Improvement':<15}")
        print("-" * 70)
        
        if heavy_system['avg_ms'] and fast_system['avg_ms']:
            improvement = heavy_system['avg_ms'] / fast_system['avg_ms']
            print(f"{'Average Latency':<25} {heavy_system['avg_ms']:.1f}ms{'':<6} {fast_system['avg_ms']:.1f}ms{'':<6} {improvement:.1f}x faster")
            
            print(f"{'Min Latency':<25} {heavy_system['min_ms']:.1f}ms{'':<6} {fast_system['min_ms']:.1f}ms{'':<6}")
            print(f"{'Max Latency':<25} {heavy_system['max_ms']:.1f}ms{'':<6} {fast_system['max_ms']:.1f}ms{'':<6}")
        
        print(f"{'Success Rate':<25} {heavy_system['success_rate']:.1f}%{'':<8} {fast_system['success_rate']:.1f}%{'':<8}")
        
        print("\n🎯 RECOMMENDATIONS:")
        if heavy_system['avg_ms'] and heavy_system['avg_ms'] > 1000:
            print("❌ Heavy AI system has >1000ms latency - NOT suitable for real-time use")
        if fast_system['avg_ms'] and fast_system['avg_ms'] < 500:
            print("✅ Ultra-fast system has <500ms latency - EXCELLENT for real-time use")
            
    else:
        print("⚠️  Could not compare - make sure both servers are running")
        print("\nTo start servers:")
        print("  Terminal 1: cd car-vision-backend && python main.py")
        print("  Terminal 2: cd car-vision-backend && python main_optimized.py")
    
    # Show detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for name, result in results.items():
        print(f"\n{name}:")
        if result['avg_ms']:
            print(f"  Average: {result['avg_ms']:.1f}ms")
            print(f"  Range: {result['min_ms']:.1f}ms - {result['max_ms']:.1f}ms")
            print(f"  Success: {result['successful_requests']}/{result['total_requests']} ({result['success_rate']:.1f}%)")
        else:
            print(f"  ❌ All requests failed")

if __name__ == "__main__":
    main()