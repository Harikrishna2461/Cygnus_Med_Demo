#!/usr/bin/env python3
"""
Example usage of the Vision Vein Detection Pipeline

This script demonstrates various ways to use the ultrasound vein detection system.
"""

import os
import json
from pathlib import Path

# Example 1: Basic video processing
def example_basic_video_processing():
    """Process ultrasound video with default settings"""
    from vision.vision_main import process_ultrasound_video
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Video Processing")
    print("="*70)
    
    # Replace with your actual video path
    video_path = "sample_ultrasound.mp4"
    
    if not Path(video_path).exists():
        print(f"⚠️ Video file not found: {video_path}")
        print("Please provide a valid ultrasound video file")
        return
    
    result = process_ultrasound_video(
        video_path=video_path,
        enable_llm=False,  # Don't use LLM for quick testing
        max_frames=10
    )
    
    print(f"\n✓ Processing completed")
    print(f"  Frames processed: {result['total_frames_processed']}")
    print(f"  Total veins: {result['summary']['total_veins_detected']}")
    print(f"  Vein types: {result['summary']['vein_type_summary']}")
    print(f"  GSV found: {len(result['summary']['gsv_frames']) > 0}")


# Example 2: Video processing with LLM
def example_video_with_llm():
    """Process video with Vision LLM confirmation"""
    from vision.vision_main import process_ultrasound_video
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Video Processing with Vision LLM")
    print("="*70)
    
    video_path = "sample_ultrasound.mp4"
    llm_api_key = os.getenv("OPENAI_API_KEY")
    
    if not llm_api_key:
        print("⚠️ OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        return
    
    result = process_ultrasound_video(
        video_path=video_path,
        enable_llm=True,
        llm_provider="openai",
        llm_api_key=llm_api_key,
        output_dir="./vision_results",
        max_frames=5  # Fewer frames for faster processing
    )
    
    print(f"\n✓ LLM analysis completed")
    
    # Show LLM results
    for frame_result in result['frame_results']:
        for vein in frame_result['veins']:
            if 'llm_analysis' in vein:
                llm = vein['llm_analysis']
                print(f"\nVein {vein['vein_id']}:")
                print(f"  LLM confirmed type: {llm['llm_confirmed_type']}")
                print(f"  LLM confidence: {llm['llm_confidence']:.2f}")
                if llm['llm_is_gsv']:
                    print(f"  🏆 This is likely the GSV!")


# Example 3: Single frame analysis
def example_single_frame():
    """Analyze a single ultrasound frame"""
    from vision.vision_main import VeinDetectionPipeline
    import cv2
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Single Frame Analysis")
    print("="*70)
    
    frame_path = "ultrasound_frame.jpg"
    
    if not Path(frame_path).exists():
        print(f"⚠️ Frame file not found: {frame_path}")
        return
    
    # Read frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error reading frame: {frame_path}")
        return
    
    # Create pipeline
    pipeline = VeinDetectionPipeline(
        enable_llm=False,
        pixels_per_mm=1.0
    )
    
    # Process frame
    result = pipeline.process_frame(frame, frame_idx=0)
    
    print(f"\n✓ Frame analysis completed")
    print(f"  Veins detected: {len(result['veins'])}")
    
    # Show vein details
    for vein in result['veins']:
        print(f"\n  Vein {vein['vein_id']}:")
        classification = vein.get('classification', {})
        print(f"    Type: {classification.get('primary_classification')}")
        print(f"    N-level: {classification.get('n_level')}")
        print(f"    Confidence: {classification.get('confidence'):.2f}")
        
        spatial = vein.get('spatial_analysis', {})
        print(f"    Distance to fascia: {spatial.get('distance_to_fascia_mm'):.1f}mm")
        print(f"    Relative position: {spatial.get('relative_position')}")


# Example 4: Custom pipeline configuration
def example_custom_configuration():
    """Use custom pipeline configuration"""
    from vision.vision_main import VeinDetectionPipeline
    import cv2
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Configuration")
    print("="*70)
    
    frame_path = "ultrasound_frame.jpg"
    
    if not Path(frame_path).exists():
        print(f"⚠️ Frame file not found: {frame_path}")
        return
    
    pipeline = VeinDetectionPipeline(
        enable_llm=True,
        llm_provider="openai",
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        pixels_per_mm=1.5,  # Custom calibration
        target_fps=3,       # Lower FPS for video
        resize_shape=(512, 512)  # Custom frame size
    )
    
    frame = cv2.imread(frame_path)
    result = pipeline.process_frame(frame)
    
    print(f"\n✓ Analysis with custom config completed")
    print(f"  Total veins: {len(result['veins'])}")


# Example 5: Detailed analysis output
def example_detailed_analysis():
    """Show detailed analysis output format"""
    from vision.vision_main import VeinDetectionPipeline
    import cv2
    import json
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Detailed Analysis Output")
    print("="*70)
    
    frame_path = "ultrasound_frame.jpg"
    
    if not Path(frame_path).exists():
        print(f"⚠️ Frame file not found: {frame_path}")
        return
    
    pipeline = VeinDetectionPipeline()
    frame = cv2.imread(frame_path)
    result = pipeline.process_frame(frame)
    
    # Show full result structure (excluding numpy arrays and images)
    serializable_result = {
        "frame_index": result['frame_index'],
        "timestamp": result['timestamp'],
        "segmentation": result['segmentation'],
        "num_veins": len(result['veins']),
        "summary_statistics": result['summary_statistics']
    }
    
    print("\nResult structure:")
    print(json.dumps(serializable_result, indent=2))


# Example 6: Using individual modules
def example_individual_modules():
    """Demonstrate using individual modules directly"""
    from vision.video.frame_extractor import FrameExtractor
    from vision.segmentation.sam_wrapper import SAMSegmenter
    from vision.geometry.spatial_analysis import SpatialAnalyzer
    from vision.classification.rules import VeinClassifier
    import cv2
    
    print("\n" + "="*70)
    print("EXAMPLE 6: Using Individual Modules")
    print("="*70)
    
    frame_path = "ultrasound_frame.jpg"
    
    if not Path(frame_path).exists():
        print(f"⚠️ Frame file not found: {frame_path}")
        return
    
    frame = cv2.imread(frame_path)
    
    # Step 1: Segmentation
    print("\nStep 1: Segmenting fascia and veins...")
    segmenter = SAMSegmenter(model_type="vit_b", device="cpu")
    fascia_mask = segmenter.segment_fascia(frame)
    vein_masks = segmenter.segment_veins(frame, fascia_mask=fascia_mask, num_masks=5)
    print(f"  Found fascia and {len(vein_masks)} veins")
    
    # Step 2: Spatial analysis
    print("\nStep 2: Analyzing spatial relationships...")
    analyzer = SpatialAnalyzer(pixels_per_mm=1.0)
    veins_analyzed = analyzer.batch_analyze_veins(vein_masks, fascia_mask)
    print(f"  Analyzed {len(veins_analyzed)} veins")
    
    # Step 3: Classification
    print("\nStep 3: Classifying veins...")
    classifier = VeinClassifier()
    veins_classified = classifier.classify_batch(veins_analyzed)
    
    # Show results
    print(f"\n  Classification summary:")
    for vein in veins_classified:
        classification = vein.get('classification', {})
        print(f"    {vein.get('vein_id')}: {classification.get('primary_classification')} (confidence: {classification.get('confidence'):.2f})")


# Example 7: REST API usage via requests
def example_rest_api_usage():
    """Demonstrate using the vision endpoints via HTTP requests"""
    import requests
    
    print("\n" + "="*70)
    print("EXAMPLE 7: REST API Usage")
    print("="*70)
    
    # Assuming backend is running on localhost:5002
    backend_url = "http://localhost:5002"
    
    # Check vision health
    print("\nChecking vision module health...")
    response = requests.get(f"{backend_url}/api/vision/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"✓ Vision module is {health['status']}")
        print(f"  Capabilities: {', '.join(health['capabilities'])}")
    else:
        print(f"✗ Vision module unavailable (status: {response.status_code})")
        print("Make sure backend is running: python app.py")
        return
    
    # Upload video for analysis
    video_path = "ultrasound.mp4"
    if Path(video_path).exists():
        print(f"\nProcessing video: {video_path}")
        
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {
                'enable_llm': 'false',
                'max_frames': '10'
            }
            
            response = requests.post(
                f"{backend_url}/api/vision/detect-veins",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Analysis completed")
            print(f"  Frames processed: {result['total_frames_processed']}")
            print(f"  Total veins: {result['summary']['total_veins_detected']}")
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.json())
    else:
        print(f"⚠️ Video file not found: {video_path}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Vision Vein Detection - Example Usage")
    print("="*70)
    
    print("\nAvailable examples:")
    print("  1. Basic video processing")
    print("  2. Video with Vision LLM")
    print("  3. Single frame analysis")
    print("  4. Custom configuration")
    print("  5. Detailed analysis output")
    print("  6. Using individual modules")
    print("  7. REST API usage")
    
    choice = input("\nSelect example (1-7) or 'all': ").strip()
    
    if choice == "1":
        example_basic_video_processing()
    elif choice == "2":
        example_video_with_llm()
    elif choice == "3":
        example_single_frame()
    elif choice == "4":
        example_custom_configuration()
    elif choice == "5":
        example_detailed_analysis()
    elif choice == "6":
        example_individual_modules()
    elif choice == "7":
        example_rest_api_usage()
    elif choice.lower() == "all":
        example_basic_video_processing()
        example_video_with_llm()
        example_single_frame()
        example_custom_configuration()
        example_detailed_analysis()
        example_individual_modules()
        example_rest_api_usage()
    else:
        print("Invalid choice")
