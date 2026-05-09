#!/usr/bin/env python3
"""
Start Car Vision Backend with Performance Toggle Support
This version includes the performance toggle endpoints
"""

import os
import sys
import uvicorn
from pathlib import Path

def setup_environment():
    """Setup environment for performance toggle support"""
    
    # Set default to ultra-fast mode for safety
    env_vars = {
        "CAR_VISION_ULTRA_FAST": "1",
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
    
    print("✅ Environment configured for performance toggle")

def check_files():
    """Check if required files exist"""
    required_files = [
        "main.py",
        "performance_controller.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def main():
    print("🚀 Car Vision Backend with Performance Toggle")
    print("=" * 50)
    
    # Check files
    if not check_files():
        print("\n❌ Cannot start - missing required files")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    print("\n🎛️  Performance Toggle Features:")
    print("  - Ultra-Fast Mode: 50-200ms latency")
    print("  - Heavy AI Mode: 10-50 seconds latency")
    print("  - Real-time switching via UI")
    print("  - Automatic optimization")
    
    print("\n📡 API Endpoints:")
    print("  - GET  /performance/status")
    print("  - POST /performance/set-ultra-fast")
    print("  - POST /performance/set-heavy-ai")
    print("  - POST /analyze-image")
    print("  - GET  /health")
    
    print("\n🌐 Starting server on http://localhost:8000")
    print("   Use the UI Performance Toggle to switch modes!")
    print("=" * 50)
    
    try:
        # Import here to ensure environment is set
        from main import app
        
        # Start server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()