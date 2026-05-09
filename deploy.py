#!/usr/bin/env python3
"""
Deployment Helper for Car Vision with Performance Toggle
Prepares the project for deployment on various platforms
"""

import os
import json
import shutil
from pathlib import Path

def create_deployment_configs():
    """Create deployment configuration files"""
    
    print("📦 Creating deployment configurations...")
    
    # Railway configuration
    railway_config = {
        "build": {
            "builder": "DOCKERFILE",
            "dockerfilePath": "Dockerfile.performance"
        },
        "deploy": {
            "startCommand": "python start_with_performance_toggle.py",
            "healthcheckPath": "/health",
            "healthcheckTimeout": 300
        }
    }
    
    with open("railway.json", "w") as f:
        json.dump(railway_config, f, indent=2)
    
    print("✅ Created railway.json")
    
    # Vercel configuration for frontend
    vercel_config = {
        "builds": [
            {
                "src": "package.json",
                "use": "@vercel/static-build",
                "config": {
                    "distDir": "web-dist"
                }
            }
        ],
        "routes": [
            {
                "src": "/(.*)",
                "dest": "/index.html"
            }
        ]
    }
    
    frontend_path = Path("../car-vision-frontend")
    if frontend_path.exists():
        with open(frontend_path / "vercel.json", "w") as f:
            json.dump(vercel_config, f, indent=2)
        print("✅ Created vercel.json for frontend")

def setup_environment_files():
    """Create environment files for different deployment scenarios"""
    
    print("🔧 Setting up environment files...")
    
    # Production environment (ultra-fast by default)
    prod_env = """# Production Environment - Ultra-Fast Mode
CAR_VISION_ULTRA_FAST=1
CAR_VISION_YOLO_MODEL=yolov8n.pt
CAR_VISION_YOLO_CONF=0.4
CAR_VISION_YOLO_IOU=0.6
CAR_VISION_YOLO_MAX_DET=20
CAR_VISION_YOLO_IMGSZ=416
CAR_VISION_YOLO_TTA=false
DEPLOYMENT_ENV=production
PYTHONUNBUFFERED=1
"""
    
    with open(".env.production", "w") as f:
        f.write(prod_env)
    
    # Development environment
    dev_env = """# Development Environment
CAR_VISION_ULTRA_FAST=1
CAR_VISION_YOLO_MODEL=yolov8n.pt
CAR_VISION_YOLO_CONF=0.4
CAR_VISION_YOLO_MAX_DET=20
CAR_VISION_YOLO_IMGSZ=416
DEPLOYMENT_ENV=development
PYTHONUNBUFFERED=1
"""
    
    with open(".env.development", "w") as f:
        f.write(dev_env)
    
    print("✅ Created .env.production and .env.development")

def create_docker_compose():
    """Create docker-compose for local deployment testing"""
    
    print("🐳 Creating Docker Compose configuration...")
    
    docker_compose = """version: '3.8'

services:
  car-vision-backend:
    build:
      context: .
      dockerfile: Dockerfile.performance
    ports:
      - "8000:8000"
    environment:
      - CAR_VISION_ULTRA_FAST=1
      - CAR_VISION_YOLO_MODEL=yolov8n.pt
      - CAR_VISION_YOLO_CONF=0.4
      - CAR_VISION_YOLO_MAX_DET=20
      - CAR_VISION_YOLO_IMGSZ=416
      - DEPLOYMENT_ENV=docker
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  car-vision-frontend:
    build:
      context: ../car-vision-frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_BACKEND_URL=http://localhost:8000
    depends_on:
      - car-vision-backend
    restart: unless-stopped
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("✅ Created docker-compose.yml")

def create_deployment_scripts():
    """Create deployment scripts for different platforms"""
    
    print("📜 Creating deployment scripts...")
    
    # Railway deployment script
    railway_deploy = """#!/bin/bash
# Deploy to Railway

echo "🚀 Deploying Car Vision Backend to Railway..."

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Login to Railway (if not already logged in)
railway login

# Deploy
railway up --detach

echo "✅ Deployment initiated. Check Railway dashboard for status."
echo "🔗 Dashboard: https://railway.app/dashboard"
"""
    
    with open("deploy-railway.sh", "w") as f:
        f.write(railway_deploy)
    
    os.chmod("deploy-railway.sh", 0o755)
    
    # Render deployment script
    render_deploy = """#!/bin/bash
# Deploy to Render

echo "🚀 Deploying Car Vision Backend to Render..."

echo "📋 Steps to deploy on Render:"
echo "1. Go to https://render.com"
echo "2. Connect your GitHub repository"
echo "3. Create a new Web Service"
echo "4. Use these settings:"
echo "   - Build Command: pip install -r requirements.txt"
echo "5. Start Command: python start_with_performance_toggle.py"
echo "6. Use render.performance.yaml for configuration"
echo ""
echo "✅ Configuration files are ready in this directory"
"""
    
    with open("deploy-render.sh", "w") as f:
        f.write(render_deploy)
    
    os.chmod("deploy-render.sh", 0o755)
    
    print("✅ Created deployment scripts")

def validate_deployment_readiness():
    """Validate that the project is ready for deployment"""
    
    print("🔍 Validating deployment readiness...")
    
    required_files = [
        "main.py",
        "performance_controller.py",
        "start_with_performance_toggle.py",
        "requirements.txt",
        "Dockerfile.performance"
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
    
    # Check if performance toggle is properly integrated
    try:
        with open("main.py", "r") as f:
            content = f.read()
            if "performance_controller" not in content:
                print("❌ Performance controller not integrated in main.py")
                return False
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False
    
    print("✅ All required files present")
    print("✅ Performance controller integrated")
    return True

def main():
    print("🚀 Car Vision Deployment Preparation")
    print("=" * 50)
    
    # Validate readiness
    if not validate_deployment_readiness():
        print("\n❌ Project not ready for deployment")
        return
    
    # Create configurations
    create_deployment_configs()
    setup_environment_files()
    create_docker_compose()
    create_deployment_scripts()
    
    print("\n✅ Deployment preparation complete!")
    print("\n📋 Next steps:")
    print("1. Choose your deployment platform:")
    print("   - Railway: ./deploy-railway.sh")
    print("   - Render: ./deploy-render.sh")
    print("   - Docker: docker-compose up")
    print("")
    print("2. Deploy frontend separately:")
    print("   - Vercel/Netlify for static hosting")
    print("   - Set REACT_APP_BACKEND_URL to your backend URL")
    print("")
    print("3. Test the performance toggle:")
    print("   - Visit your deployed frontend")
    print("   - Look for '🚀 Performance Mode' section")
    print("   - Test both ultra-fast and heavy AI modes")
    print("")
    print("🎯 Your system will start in ultra-fast mode (50-200ms latency)")
    print("🔧 Users can switch to heavy AI mode via the UI toggle")

if __name__ == "__main__":
    main()