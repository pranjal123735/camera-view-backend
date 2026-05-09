# 🚀 Deployment Guide - Car Vision with Performance Toggle

## Overview
This guide shows how to deploy the car vision system with performance toggle support on various platforms.

## 🏗️ Deployment Options

### 1. Railway (Recommended)

#### Backend Deployment
```bash
# 1. Push to GitHub
git add .
git commit -m "Add performance toggle"
git push

# 2. Connect to Railway
# - Go to railway.app
# - Connect your GitHub repo
# - Select car-vision-backend folder
# - Use Dockerfile.performance

# 3. Set Environment Variables (Railway Dashboard)
CAR_VISION_ULTRA_FAST=1
CAR_VISION_YOLO_MODEL=yolov8n.pt
CAR_VISION_YOLO_CONF=0.4
CAR_VISION_YOLO_MAX_DET=20
CAR_VISION_YOLO_IMGSZ=416
DEPLOYMENT_ENV=railway
```

#### Frontend Deployment
```bash
# 1. Deploy frontend to Railway/Vercel/Netlify
# 2. Set backend URL in environment
REACT_APP_BACKEND_URL=https://your-backend.railway.app
```

### 2. Render

#### Backend (render.performance.yaml)
```bash
# 1. Connect GitHub repo to Render
# 2. Use render.performance.yaml config
# 3. Environment variables are pre-configured
```

#### Frontend
```bash
# Deploy as static site
# Set REACT_APP_BACKEND_URL to your Render backend URL
```

### 3. Docker Deployment

#### Build and Run
```bash
# Build with performance toggle support
docker build -f Dockerfile.performance -t car-vision-performance .

# Run with ultra-fast mode (default)
docker run -p 8000:8000 \
  -e CAR_VISION_ULTRA_FAST=1 \
  -e CAR_VISION_YOLO_MODEL=yolov8n.pt \
  car-vision-performance

# Run with heavy AI mode
docker run -p 8000:8000 \
  -e CAR_VISION_ULTRA_FAST=0 \
  -e CAR_VISION_YOLO_MODEL=yolov8s.pt \
  car-vision-performance
```

### 4. Local Development

#### Start Backend
```bash
cd car-vision-backend
python start_with_performance_toggle.py
```

#### Start Frontend
```bash
cd car-vision-frontend
npm run web
```

## 🔧 Environment Variables

### Required for All Deployments
```bash
# Performance Mode (1=ultra-fast, 0=heavy-ai)
CAR_VISION_ULTRA_FAST=1

# YOLO Configuration
CAR_VISION_YOLO_MODEL=yolov8n.pt
CAR_VISION_YOLO_CONF=0.4
CAR_VISION_YOLO_IOU=0.6
CAR_VISION_YOLO_MAX_DET=20
CAR_VISION_YOLO_IMGSZ=416
CAR_VISION_YOLO_TTA=false

# System
PYTHONUNBUFFERED=1
```

### Optional
```bash
# Deployment tracking
DEPLOYMENT_ENV=production
RAILWAY_ENVIRONMENT=production
RENDER=true

# Frontend backend URL
REACT_APP_BACKEND_URL=https://your-backend-url.com
```

## 🎛️ Performance Toggle in Production

### How It Works
1. **Default Mode**: Starts in ultra-fast mode (safe for production)
2. **Runtime Switching**: Users can toggle modes via UI
3. **No Restart Required**: Changes apply immediately
4. **Persistent**: Settings persist until next restart

### API Endpoints
```bash
# Check current mode
GET /performance/status

# Switch to ultra-fast (50-200ms)
POST /performance/set-ultra-fast

# Switch to heavy AI (10-50 seconds)
POST /performance/set-heavy-ai

# Health check
GET /health
```

## 🔍 Troubleshooting

### Backend Issues

#### Performance Toggle Not Working
```bash
# Check if endpoints exist
curl https://your-backend.com/performance/status

# Check logs for errors
# Railway: View logs in dashboard
# Render: Check logs in dashboard
# Docker: docker logs container-name
```

#### High Latency in Ultra-Fast Mode
```bash
# Verify environment variables
curl https://your-backend.com/performance/status

# Should show:
# "mode": "ultra_fast"
# "yolo_config.model": "yolov8n.pt"
```

#### Model Download Issues
```bash
# Models download automatically on first run
# Check logs for download progress
# Ensure sufficient disk space (500MB+)
```

### Frontend Issues

#### Toggle UI Not Appearing
```bash
# Check if PerformanceToggle component is imported
# Verify backend URL is correct
# Check browser console for errors
```

#### Backend Connection Failed
```bash
# Verify backend URL in environment
# Check CORS settings
# Test backend health endpoint directly
```

## 📊 Monitoring

### Performance Metrics
- **Ultra-Fast Mode**: 50-200ms expected
- **Heavy AI Mode**: 10-50 seconds expected
- **Health Check**: Should respond < 5 seconds

### Logging
```python
# Backend logs show:
# - Performance mode switches
# - Latency measurements
# - Error details
# - Model loading status
```

### Alerts
```bash
# Set up monitoring for:
# - Response time > 1 second (ultra-fast mode)
# - Response time > 60 seconds (heavy AI mode)
# - Health check failures
# - Memory usage > 80%
```

## 🚀 Production Best Practices

### 1. Default Configuration
- Always start in ultra-fast mode
- Use yolov8n.pt model by default
- Set reasonable resource limits

### 2. Resource Management
```bash
# Memory: 2GB minimum, 4GB recommended
# CPU: 2 cores minimum
# Storage: 2GB for models and cache
```

### 3. Security
```bash
# Enable HTTPS in production
# Set proper CORS origins
# Use environment variables for secrets
```

### 4. Scaling
```bash
# Ultra-fast mode: Can handle 10-50 requests/second
# Heavy AI mode: 1 request per 10-50 seconds
# Use load balancer for multiple instances
```

## 🎯 Deployment Checklist

### Pre-Deployment
- [ ] Test performance toggle locally
- [ ] Verify environment variables
- [ ] Check model downloads
- [ ] Test both modes

### Deployment
- [ ] Deploy backend with performance toggle
- [ ] Deploy frontend with toggle UI
- [ ] Set environment variables
- [ ] Test health endpoints

### Post-Deployment
- [ ] Verify ultra-fast mode is default
- [ ] Test mode switching via UI
- [ ] Monitor latency metrics
- [ ] Check error logs

### Production Validation
- [ ] Test with real images
- [ ] Verify 50-200ms latency in ultra-fast mode
- [ ] Confirm toggle works in deployed environment
- [ ] Test error handling and recovery

## 📞 Support

### Common Issues
1. **High latency**: Check if in heavy AI mode, switch to ultra-fast
2. **Toggle not working**: Verify API endpoints and CORS
3. **Model errors**: Check disk space and network connectivity
4. **Memory issues**: Increase container memory limits

### Debug Commands
```bash
# Check backend status
curl https://your-backend.com/health

# Check performance mode
curl https://your-backend.com/performance/status

# Test latency
time curl -X POST https://your-backend.com/analyze-image \
  -F "file=@test-image.jpg"
```

This deployment setup ensures your performance toggle works reliably in production environments!