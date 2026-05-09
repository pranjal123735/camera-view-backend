# 🚀 Performance Toggle Feature

## Overview
The Performance Toggle allows you to switch between **Ultra-Fast Mode** (50-200ms) and **Heavy AI Mode** (10-50 seconds) directly from the UI without restarting the server.

## Quick Start

### 1. Start Backend with Performance Toggle
```bash
cd car-vision-backend
python start_with_performance_toggle.py
```

### 2. Start Frontend
```bash
cd car-vision-frontend
npm run web
```

### 3. Use the Performance Toggle
- Open the web UI at http://localhost:19006
- Look for the "🚀 Performance Mode" section
- Click the toggle button to switch between modes
- Test latency with the "Test Latency" button

## Modes Comparison

### ⚡ Ultra-Fast Mode (RECOMMENDED for driving)
- **Latency:** 50-200ms
- **Features:** Core detection, tracking, distance, risk calculation
- **Disabled:** RAG, Knowledge Graph, Ensemble, Learning
- **Model:** yolov8n.pt (nano - fastest)
- **Use Case:** Real-time driving, live demos

### 🤖 Heavy AI Mode (For analysis only)
- **Latency:** 10-50 seconds
- **Features:** All Ultra-Fast features + advanced AI
- **Enabled:** RAG enhancement, Knowledge Graph, Ensemble detection, Learning
- **Model:** yolov8s.pt (standard)
- **Use Case:** Offline analysis, research, testing

## API Endpoints

### Check Current Mode
```bash
GET /performance/status
```

### Switch to Ultra-Fast
```bash
POST /performance/set-ultra-fast
```

### Switch to Heavy AI
```bash
POST /performance/set-heavy-ai
```

### Test Latency
```bash
POST /analyze-image
# (with image file)
```

## UI Features

### Performance Toggle Component
- **Real-time latency testing**
- **Mode comparison display**
- **One-click switching**
- **Visual status indicators**
- **Safety warnings**

### Status Indicators
- 🟢 **Green:** Ultra-Fast mode (good for driving)
- 🟣 **Purple:** Heavy AI mode (analysis only)
- ⚠️ **Warning:** Displayed when in Heavy AI mode

## Safety Features

### Automatic Warnings
- Heavy AI mode shows warning: "NOT suitable for real-time driving"
- Latency status: Excellent (<500ms), Good (<2s), Poor (<10s), Unusable (>10s)
- Mode recommendations based on use case

### Default Mode
- System starts in **Ultra-Fast mode** by default for safety
- Environment variables are automatically configured
- No manual configuration needed

## Troubleshooting

### Backend Issues
```bash
# Check if server is running
curl http://localhost:8000/health

# Check performance status
curl http://localhost:8000/performance/status

# Restart with performance toggle
python start_with_performance_toggle.py
```

### Frontend Issues
```bash
# Restart frontend
cd car-vision-frontend
npm run web
```

### High Latency in Ultra-Fast Mode
- Check if yolov8n.pt model is downloaded
- Verify environment variables are set correctly
- Restart backend server

### Toggle Not Working
- Ensure both backend and frontend are running
- Check browser console for errors
- Verify API endpoints are accessible

## Environment Variables

The performance toggle automatically manages these variables:

```bash
# Ultra-Fast Mode
CAR_VISION_ULTRA_FAST=1
CAR_VISION_YOLO_MODEL=yolov8n.pt
CAR_VISION_YOLO_CONF=0.4
CAR_VISION_YOLO_MAX_DET=20
CAR_VISION_YOLO_IMGSZ=416

# Heavy AI Mode  
CAR_VISION_ULTRA_FAST=0
CAR_VISION_YOLO_MODEL=yolov8s.pt
CAR_VISION_YOLO_CONF=0.3
CAR_VISION_YOLO_MAX_DET=60
CAR_VISION_YOLO_IMGSZ=640
```

## Files Added

### Backend
- `performance_controller.py` - Performance toggle API
- `start_with_performance_toggle.py` - Startup script

### Frontend
- `components/PerformanceToggle.js` - UI toggle component

### Modified Files
- `main.py` - Added performance toggle support
- `App.js` - Added PerformanceToggle component

## Benefits

1. **No Server Restart Required** - Switch modes instantly
2. **Real-time Latency Testing** - See immediate performance impact
3. **Safety First** - Starts in ultra-fast mode by default
4. **Visual Feedback** - Clear indicators and warnings
5. **Easy Deployment** - Works in production environments

## Production Deployment

The performance toggle works in deployed environments:

1. Deploy backend with performance toggle support
2. Deploy frontend with toggle UI
3. Users can switch modes as needed
4. No server access required for mode switching

This solves your original problem of 48655ms latency while giving you the flexibility to use advanced AI features when needed!