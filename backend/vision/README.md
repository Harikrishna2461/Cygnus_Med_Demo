# Vision Module: Ultrasound Vein Detection & Classification

A production-ready computer vision pipeline for detecting, segmenting, and classifying different types of veins from ultrasound video and images.

## 🎯 Overview

This module provides a complete solution for analyzing ultrasound images to:

1. **Detect and segment** fascia and multiple vein structures
2. **Compute spatial relationships** between veins and anatomical landmarks
3. **Classify veins** into clinically relevant categories:
   - **Deep Veins** (below fascia)
   - **Superficial Veins** (above fascia)
   - **Perforator Veins** (crossing fascia)
   - **Great Saphenous Vein (GSV)** (specific superficial vein)
4. **Assign depth levels** (N1/N2/N3) based on distance from skin surface
5. **Optionally use Vision LLM** for ambiguous classifications and GSV identification

## 📦 Architecture

```
vision/
├── video/
│   └── frame_extractor.py          # Video frame extraction
├── segmentation/
│   └── sam_wrapper.py               # Segment Anything Model wrapper
├── geometry/
│   └── spatial_analysis.py          # Spatial relationship computation
├── classification/
│   ├── rules.py                     # Rule-based classification
│   └── llm_interface.py             # Vision LLM integration
├── utils/
│   └── visualization.py             # Annotation and visualization
├── vision_main.py                   # Main orchestrator pipeline
└── config.py                        # Configuration settings
```

## 🚀 Quick Start

### Installation

1. **Install dependencies** (part of backend requirements):
```bash
pip install -r requirements.txt
```

2. **Install SAM weights** (first time only):
```bash
# Note: SAM weights are downloaded automatically on first segmentation attempt
# Or manually download from: https://dl.fbaipublicfiles.com/segment_anything/
```

### Basic Usage

#### Video Processing

```python
from vision.vision_main import process_ultrasound_video

result = process_ultrasound_video(
    video_path="ultrasound.mp4",
    enable_llm=True,
    llm_provider="openai",
    llm_api_key="sk-...",
    output_dir="./results",
    max_frames=30
)

print(f"Processed {result['total_frames_processed']} frames")
print(f"Total veins detected: {result['summary']['total_veins_detected']}")
print(f"GSV found: {result['summary']['gsv_frames']}")
```

#### Single Frame Analysis

```python
from vision.vision_main import VeinDetectionPipeline
import cv2

pipeline = VeinDetectionPipeline(enable_llm=True)
frame = cv2.imread("ultrasound_frame.jpg")

result = pipeline.process_frame(frame, frame_idx=0)

for vein in result['veins']:
    print(f"Vein {vein['vein_id']}:")
    print(f"  Type: {vein['classification']['primary_classification']}")
    print(f"  Depth (N-level): {vein['classification']['n_level']}")
    print(f"  Confidence: {vein['classification']['confidence']:.2f}")
```

## 🔌 API Endpoints

### POST /api/vision/detect-veins

**Process ultrasound video file**

Request (multipart/form-data):
```json
{
  "file": <binary video file>,
  "enable_llm": true,                    // optional
  "llm_provider": "openai",              // optional
  "llm_api_key": "sk-...",               // optional
  "max_frames": 30                       // optional
}
```

Response:
```json
{
  "status": "success",
  "video_file": "ultrasound.mp4",
  "total_frames_processed": 30,
  "summary": {
    "total_veins_detected": 15,
    "by_type": {
      "deep_vein": 4,
      "superficial_vein": 8,
      "perforator_vein": 2,
      "gsv": 1
    },
    "by_nlevel": {
      "N1": 4,
      "N2": 8,
      "N3": 3
    },
    "gsv_frames": [5, 12, 18]
  },
  "frame_results": [
    {
      "frame_index": 0,
      "timestamp": "2024-01-15T10:30:45.123Z",
      "veins": [
        {
          "vein_id": "V0",
          "classification": {
            "primary_classification": "superficial_vein",
            "n_level": "N3",
            "confidence": 0.92,
            "reasoning": "Classified as SUPERFICIAL VEIN: Located above fascia..."
          },
          "spatial_analysis": {
            "vein_centroid": [256.5, 128.3],
            "distance_to_fascia_mm": 3.2,
            "intersects_fascia": false,
            "relative_position": "above",
            "depth_info": {
              "distance_from_skin_mm": 2.1,
              "distance_from_fascia_mm": 3.2
            }
          }
        }
      ],
      "summary_statistics": {
        "total_veins": 2,
        "by_type": {
          "superficial_vein": 1,
          "deep_vein": 1
        },
        "by_nlevel": {
          "N1": 1,
          "N3": 1
        },
        "gsv_present": false
      }
    }
  ]
}
```

### POST /api/vision/analyze-frame

**Analyze single ultrasound image**

Request (multipart/form-data):
```json
{
  "file": <binary image file>,
  "enable_llm": true               // optional
}
```

Response:
```json
{
  "status": "success",
  "image_file": "frame.jpg",
  "veins": [...],
  "summary_statistics": {...},
  "visualization": {
    "format": "base64",
    "data": "iVBORw0KGgoAAAAN..."
  }
}
```

### GET /api/vision/health

**Check vision module health**

Response:
```json
{
  "status": "healthy",
  "module": "vision_vein_detection",
  "capabilities": [
    "video_processing",
    "frame_segmentation",
    "vein_classification",
    "spatial_analysis",
    "optional_llm_integration"
  ]
}
```

## 🧠 Classification System

### Primary Classification Rules

| Rule | Condition | Result |
|------|-----------|--------|
| **Perforator** | Vein crosses/intersects fascia | `perforator_vein` |
| **Deep Vein** | Located below fascia | `deep_vein` |
| **Superficial** | Located above fascia | `superficial_vein` |
| **Unknown** | Ambiguous spatial relationship | Requires LLM |

### N-Level (Depth) Classification

| Level | Distance from Skin | Typical Location |
|-------|-------------------|-----------------|
| **N1** | > 10mm | Deep veins, below muscle layer |
| **N2** | 5-10mm | Mid-depth (GSVs, some perforators) |
| **N3** | < 5mm | Superficial, close to skin |

### Confidence Scoring

Each classification includes a confidence score (0-1):
- **≥ 0.8**: High confidence, likely accurate
- **0.6-0.8**: Moderate confidence, reasonable classification
- **< 0.6**: Low confidence, LLM confirmation recommended

## 🤖 Vision LLM Integration

When `enable_llm=true`, the system uses OpenAI's GPT-4 Vision or Anthropic's Claude Vision to:

1. **Confirm ambiguous classifications**
2. **Identify Great Saphenous Vein (GSV)** specifically
3. **Provide clinical confidence scoring**
4. **Flag anatomical variants**

### Example LLM Response

```json
{
  "vein_classifications": [
    {
      "vein_id": "V1",
      "confirmed_type": "gsv",
      "confidence": 0.95,
      "is_gsv": true,
      "notes": "Large tortuous vein with characteristic medial location, likely GSV"
    }
  ],
  "gsv_summary": {
    "identified": true,
    "vein_id": "V1",
    "confidence": 0.95
  }
}
```

## 📊 Output Data Structures

### Vein Object

```json
{
  "vein_id": "V0",
  "mask": <numpy array>,
  "confidence": 0.87,
  "properties": {
    "area": 2450,
    "centroid": [256.5, 128.3],
    "perimeter": 180.5
  },
  "spatial_analysis": {
    "vein_centroid": [256.5, 128.3],
    "distance_to_fascia_mm": 3.2,
    "distance_to_fascia_px": 3.2,
    "intersects_fascia": false,
    "intersection_length_px": 0,
    "relative_position": "above",
    "depth_info": {
      "distance_from_skin_mm": 2.1,
      "distance_from_skin_px": 2.1,
      "distance_from_fascia_mm": 3.2,
      "distance_from_fascia_px": 3.2
    }
  },
  "classification": {
    "primary_classification": "superficial_vein",
    "n_level": "N3",
    "confidence": 0.92,
    "reasoning": "Classified as SUPERFICIAL VEIN: Located above fascia (3.2mm) | Depth level N3: 2.1mm from skin surface",
    "requires_llm_confirmation": false
  },
  "llm_analysis": {
    "llm_confirmed_type": "superficial_vein",
    "llm_confidence": 0.95,
    "llm_is_gsv": false,
    "llm_notes": "Confirmed as superficial branch, not main GSV trunk"
  }
}
```

## ⚙️ Configuration

Edit `vision/config.py` to customize:

- **Segmentation quality**: SAM model type, confidence thresholds
- **Depth calibration**: Pixels per millimeter (adjust for ultrasound probe/settings)
- **Classification thresholds**: Distance thresholds for all vein types
- **LLM settings**: Model selection, timeout, token limits
- **Performance**: Batch size, caching, visualization settings

## 🔍 Technical Details

### Segmentation Module (SAM)

- Uses **Meta's Segment Anything Model (SAM)** for mask generation
- Supports automatic detection hints for fascia location
- Filters vein masks based on shape (elongation ratio > 1.5)
- Computes mask centroids and properties

### Geometry Module

- Computes **minimum distance** from vein to fascia using distance transform
- Determines **relative position** (above/below/crossing) via spatial analysis
- Calculates **depth metrics** from skin surface
- Handles edge cases (veins at image boundaries)

### Classification Module

- **Rule-based approach** for deterministic, explainable decisions
- **Confidence scoring** based on segmentation quality and spatial clarity
- **LLM fallback** for ambiguous cases
- **GSV detection** using anatomical heuristics + LLM confirmation

### Visualization Module

- **Color-coded overlays**: Different colors for each vein type
- **Segmentation masks**: Semi-transparent overlays of detected structures
- **Spatial measurements**: Distance lines, depth annotations
- **Classification labels**: Vein ID, type, confidence, N-level
- **Comparison grids**: 2x2 grid showing original, segmentation, classification, detailed analysis

## 🧪 Testing

### Test with Sample Video

```python
from vision.vision_main import process_ultrasound_video

# Process test video
result = process_ultrasound_video(
    video_path="sample_ultrasound.mp4",
    enable_llm=False,  # Disable LLM for testing
    output_dir="./test_output",
    max_frames=5
)

print(f"Found {result['summary']['total_veins_detected']} veins")
```

### Test LLM Integration

```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'

result = process_ultrasound_video(
    video_path="sample_ultrasound.mp4",
    enable_llm=True,
    llm_provider="openai",
    output_dir="./test_output",
    max_frames=1
)

# Check LLM results
for frame_result in result['frame_results']:
    for vein in frame_result['veins']:
        if 'llm_analysis' in vein:
            print(f"LLM confirmed: {vein['llm_analysis']['llm_confirmed_type']}")
```

## 🐛 Troubleshooting

### SAM Model Weights Not Found

```
Error: ./weights/sam_vit_b.pth not found
```

Solution: Download SAM weights manually:
```bash
mkdir -p weights
wget -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b.pth
```

### LLM API Errors

**"Vision module not available"**: Install OpenCV:
```bash
pip install opencv-python opencv-contrib-python
```

**"API key invalid"**: Check OPENAI_API_KEY environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Out of Memory Errors

For large videos or high-resolution frames:
- Reduce `max_frames` parameter
- Enable frame resizing: `resize_shape=(480, 360)`
- Reduce SAM model size: `model_type="mobile_sam"`
- Disable visualizations: `save_visualizations=False`

### Slow Performance

- Use GPU for SAM: `device="cuda"`
- Reduce target FPS: `target_fps=2`
- Skip LLM analysis: `enable_llm=False`
- Enable batch processing: `USE_BATCH_PROCESSING=True` in config

## 📝 Key Functions

### `VeinDetectionPipeline`

Main orchestrator class

```python
pipeline = VeinDetectionPipeline(
    enable_llm=False,
    llm_provider="openai",
    llm_api_key=None,
    pixels_per_mm=1.0,
    target_fps=5,
    resize_shape=None
)

# Process entire video
result = pipeline.process_video(
    video_path="ultrasound.mp4",
    max_frames=30,
    save_visualizations=True,
    output_dir="./results"
)

# Process single frame
result = pipeline.process_frame(frame, frame_idx=0)
```

### `FrameExtractor`

Video frame sampling

```python
extractor = FrameExtractor(target_fps=5, resize_shape=(480, 360))
frames = extractor.extract_frames("ultrasound.mp4", max_frames=30)
```

### `SAMSegmenter`

Segmentation

```python
segmenter = SAMSegmenter(model_type="vit_b", device="cpu")
fascia_mask = segmenter.segment_fascia(frame)
vein_masks = segmenter.segment_veins(frame, num_masks=5)
```

### `SpatialAnalyzer`

Geometry computation

```python
analyzer = SpatialAnalyzer(pixels_per_mm=1.0)
analysis = analyzer.analyze_vein_position(vein_mask, fascia_mask)
# Returns: centroid, distance_to_fascia, intersects, relative_position, depth_info
```

### `VeinClassifier`

Rule-based classification

```python
classifier = VeinClassifier()
classification = classifier.classify_vein(vein_data)
# Returns: primary_classification, n_level, confidence, reasoning
```

### `VisionLLMInterface`

Vision LLM integration

```python
llm = VisionLLMInterface(model_provider="openai", api_key="sk-...")
veins = llm.classify_veins_with_llm(frame, veins, overlay)
gsv_result = llm.identify_gsv(frame, superficial_veins, overlay)
```

## 📚 References

- **Segment Anything**: https://segment-anything.com/
- **Vision Transformers (ViT)**: https://arxiv.org/abs/2010.11929
- **Ultrasound Anatomy**: https://www.sonosite.com/ (Educational resources)
- **Venous Classification**: https://en.wikipedia.org/wiki/Vein#Classification

## 📄 License

Part of the Cygnus Medical Decision Support System. See main LICENSE file.

## 👥 Contributors

- Computer Vision Engineering Team
- Ultrasound Imaging Specialists
- Clinical Domain Experts

## 🤝 Support

For issues, questions, or contributions:
- Create an issue in the main repository
- Check existing documentation in this module
- Review API endpoint examples above
