#!/usr/bin/env python3
"""
Script to disable heavy AI features causing latency in car vision system
Run this to modify main.py and remove the heavy processing layers
"""

import os
import shutil
from pathlib import Path

def create_fast_main_py():
    """Create a fast version of main.py by removing heavy AI processing"""
    
    # Read the current main.py
    main_py_path = Path("main.py")
    if not main_py_path.exists():
        print("❌ main.py not found!")
        return False
    
    # Backup original
    backup_path = Path("main_original_with_ai.py")
    shutil.copy2(main_py_path, backup_path)
    print(f"✅ Backed up original to {backup_path}")
    
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Remove heavy imports
    content = content.replace("from rag_enhanced_detection import rag_enhancer", "# from rag_enhanced_detection import rag_enhancer")
    content = content.replace("from ensemble_detection import ensemble_system", "# from ensemble_detection import ensemble_system")
    content = content.replace("from knowledge_graph import knowledge_graph", "# from knowledge_graph import knowledge_graph")
    
    # Find and replace the heavy processing section in analyze_image
    heavy_processing_start = "# === ADVANCED RAG & AI ENHANCEMENT PIPELINE ==="
    heavy_processing_end = "# Convert back to the original pipeline format"
    
    if heavy_processing_start in content and heavy_processing_end in content:
        start_idx = content.find(heavy_processing_start)
        end_idx = content.find(heavy_processing_end)
        
        if start_idx != -1 and end_idx != -1:
            # Replace heavy processing with simple conversion
            replacement = """    # === FAST PROCESSING - NO AI ENHANCEMENTS ===
    
    # Convert to detection dictionaries for simple processing
    final_enhanced_detections = []
    for label, conf, bbox in raw_rows:
        final_enhanced_detections.append({
            "label": label,
            "confidence": conf,
            "bbox_xyxy": list(bbox)
        })

    """
            
            content = content[:start_idx] + replacement + content[end_idx:]
            print("✅ Removed heavy AI processing pipeline")
        else:
            print("⚠️  Could not find heavy processing section boundaries")
    else:
        print("⚠️  Could not find heavy processing section")
    
    # Remove AI enhancement metadata
    ai_metadata_section = '''        # Add AI enhancement metadata
        if hasattr(detection_out, '__dict__'):
            detection_out.__dict__.update({
                "ai_enhanced": True,
                "rag_metadata": detection.get("rag_metadata", {}),
                "kg_analysis": kg_analysis.get("context_consistency", {}).get(smooth_label, 0.5),
                "ensemble_data": {
                    "consensus": detection.get("ensemble_consensus", 1.0),
                    "agreement": detection.get("model_agreement", 1.0)
                } if "ensemble_consensus" in detection else None
            })'''
    
    if ai_metadata_section in content:
        content = content.replace(ai_metadata_section, "        # AI enhancements disabled for performance")
        print("✅ Removed AI enhancement metadata")
    
    # Write the fast version
    with open(main_py_path, 'w') as f:
        f.write(content)
    
    print("✅ Created fast version of main.py")
    return True

def set_fast_environment():
    """Set environment variables for fastest performance"""
    
    env_vars = {
        "CAR_VISION_YOLO_MODEL": "yolov8n.pt",  # Fastest model
        "CAR_VISION_YOLO_CONF": "0.4",          # Higher confidence for fewer false positives
        "CAR_VISION_YOLO_IOU": "0.6",           # Higher IoU for better separation
        "CAR_VISION_YOLO_MAX_DET": "20",        # Limit detections for speed
        "CAR_VISION_YOLO_IMGSZ": "416",         # Smaller image size for speed
        "CAR_VISION_YOLO_TTA": "false",         # Disable test time augmentation
    }
    
    # Create .env file
    env_content = "# Ultra-fast car vision configuration\n"
    for key, value in env_vars.items():
        env_content += f"{key}={value}\n"
        os.environ[key] = value
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Set fast environment variables in .env file")
    print("Environment variables set:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")

def main():
    print("🚀 Car Vision Latency Fix")
    print("=" * 50)
    
    print("\n1. Creating fast version of main.py...")
    if create_fast_main_py():
        print("\n2. Setting fast environment variables...")
        set_fast_environment()
        
        print("\n✅ LATENCY FIX COMPLETE!")
        print("\nNext steps:")
        print("1. Restart your car vision backend:")
        print("   python main.py")
        print("\n2. Expected latency reduction:")
        print("   From: 48655ms+ → To: 50-200ms")
        print("\n3. Your original file is backed up as: main_original_with_ai.py")
        print("\n4. To restore AI features later:")
        print("   mv main_original_with_ai.py main.py")
        
    else:
        print("\n❌ Failed to create fast version")
        print("Alternative: Use main_optimized.py instead:")
        print("   python main_optimized.py")

if __name__ == "__main__":
    main()