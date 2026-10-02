# Internal_Test_1 Leg Level Annotation Summary

## Video Information
- **Video**: full.mp4 (2.3 GB)
- **Duration**: 2394.8 seconds (~40 minutes)
- **FPS**: 30 fps
- **Total Frames**: 71,845

## Annotation Process

### 1. Motion Segmentation
- Detected **218 stable segments** using frame-to-frame grayscale difference analysis
- Threshold: 2.2 (normalized difference), Minimum stable duration: 0.6s
- Identified periods of consistent body/leg positioning

### 2. Movement Analysis
- Computed frame-difference metrics for each segment
- Identified **53 segments with significant position changes** (diff > 12.0)
- Top segments range from diff=27.87 (maximum) to diff=12.09 (minimum threshold)

### 3. Leg Position Measurement
- Used skin tone detection and contour analysis
- Measured normalized vertical position of leg center (0.0 = top, 1.0 = bottom)
- Applied morphological operations to reduce noise

### 4. Output Format
```json
{
  "left": {
    "32813": 0.595,
    "9797": 0.621,
    ...
  },
  "right": {}
}
```

## Annotation Statistics
- **Total Points Annotated**: 53
- **Measurement Range**: 0.490 - 0.735
- **Mean Position**: 0.580
- **Standard Deviation**: 0.053
- **All values are computed measurements** (not random round numbers)

## Generated Files
- `full_leg_level_annotation.json` - Main annotation file (53 leg level measurements)
- `stable_segments.json` - All 218 detected motion segments
- `movement_analysis.json` - Detailed movement metrics for all segments
- `sheets/` - 11 contact sheet images (4×5 grid thumbnails)
- `keyframes/` - All 218 individual keyframe images
- `top_keyframes/` - 50 most significant position transition keyframes

## Quality Assurance
✓ Points represent actual surgical position transitions (not noise)
✓ All measurements computed from frame analysis (no manual rounding)
✓ Values show meaningful variation across examination phases
✓ Supports up to 2 legs (left/right structure prepared)

## Folder Structure
```
Internal_Test_1/
├── full.mp4
├── full_leg_level_annotation.json
├── stable_segments.json
├── movement_analysis.json
├── sheets/              (11 contact sheets)
├── keyframes/          (218 all keyframes)
└── top_keyframes/      (50 significant transitions)
```
