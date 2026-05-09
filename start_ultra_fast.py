#!/usr/bin/env python3
"""
Ultra-fast startup script for car vision backend
This bypasses all heavy AI processing for real-time performance
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "main_optimized.py",
        "yolov8n.pt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        
        if "yolov8n.pt" in missing_files:
            print("\n📥 Downloading yolov8n.pt model...")
            try:
                from ultralytics import YOLO
                model = YOLO("yolov8n.pt")  # This will download it
                print("✅ Downloaded yolov8n.pt")
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                return False
        
        return len(missing_files) == 1 and "yolov8n.pt" in missing_files
    
    return True

def set_ultra_fast_env():
    """Set environment variables for ultra-fast performance"""
    
    env_vars = {
        "CAR_VISION_YOLO_MODEL": "yolov8n.pt",
        "CAR_VISION_YOLO_CONF": "0.4",
        "CAR_VISION_YOLO_IOU": "0.6", 
        "CAR_VISION_YOLO_MAX_DET": "20",
        "CAR_VISION_YOLO_IMGSZ": "416",
        "CAR_VISION_YOLO_TTA": "false",
        "PYTHONUNBUFFERED": "1"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ Ultra-fast environment configured")

def start_server():
    """Start the ultra-fast server"""
    
    print("🚀 Starting Ultra-Fast Car Vision Backend...")
    print("=" * 50)
    print("Configuration:")
    print("  - Model: yolov8n.pt (nano - fastest)")
    print("  - Image size: 416px")
    print("  - Max detections: 20")
    print("  - AI enhancements: DISABLED")
    print("  - Expected latency: 50-200ms")
    print("=" * 50)
    
    try:
        # Start the optimized server
        subprocess.run([
            sys.executable, "main_optimized.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    print("🏎️  Ultra-Fast Car Vision Startup")
    print("This will start the car vision backend with minimal latency")
    print()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed")
        sys.exit(1)
    
    # Set environment
    set_ultra_fast_env()
    
    # Start server
    if not start_server():
        sys.exit(1)

if __name__ == "__main__":
    main()