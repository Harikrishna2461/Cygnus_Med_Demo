"""
Vision Module Configuration

Settings for ultrasound vein detection pipeline.
"""

# Segmentation Settings
SAM_MODEL_TYPE = "vit_b"          # Model variant: vit_h, vit_l, vit_b, mobile_sam
SAM_DEVICE = "cpu"                # Device: cuda or cpu
SAM_CONFIDENCE_THRESHOLD = 0.5     # Minimum confidence for SAM masks

# Spatial Analysis Settings
DEPTH_THRESHOLD_MM = 5.0           # Threshold for depth classification (mm)
PIXELS_PER_MM = 1.0                # Calibration: pixels per millimeter
FASCIA_INTERSECTION_THRESHOLD = 10  # Min pixels for intersection detection

# Classification Settings
PERFORATOR_INTERSECTION_THRESHOLD = 10  # pixels
DEEP_VEIN_DEPTH_THRESHOLD_MM = 10.0     # below fascia
SUPERFICIAL_DEPTH_THRESHOLD_MM = 5.0    # above fascia
N1_THRESHOLD_MM = 10.0                  # Very deep (deep veins)
N2_THRESHOLD_MM = 5.0                   # Mid depth (GSVs)
N3_THRESHOLD_MM = 2.0                   # Superficial (near skin)

# LLM Settings
LLM_ENABLED_DEFAULT = False
LLM_PROVIDER_DEFAULT = "openai"         # openai or anthropic
LLM_MODEL_OPENAI = "gpt-4-vision-preview"
LLM_MODEL_ANTHROPIC = "claude-3-vision-20240229"
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT_SECONDS = 30

# Frame Extraction Settings
DEFAULT_TARGET_FPS = 5              # Target frames per second
MAX_FRAMES_PER_VIDEO = 300          # Maximum frames to process
DEFAULT_RESIZE_SHAPE = None         # Optional (height, width) resizing

# Visualization Settings
VIZ_FONT_SCALE = 0.6
VIZ_LINE_THICKNESS = 2
VIZ_ALPHA_MASK = 0.3                # Transparency for overlays
VIZ_ALPHA_CONTOUR = 0.7

# Performance Settings
USE_BATCH_PROCESSING = True         # Process multiple frames in batch
BATCH_SIZE = 4                      # Frames per batch
CACHE_RESULTS = True                # Cache segmentation results
MAX_CACHE_SIZE_MB = 500             # Max cache size in MB

# Input Validation
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tiff', 'bmp'}
MAX_FILE_SIZE_MB = 500
MAX_IMAGE_SIZE_PIXELS = 4096

# Logging
LOG_LEVEL = "INFO"
LOG_VEIN_DETAILS = True             # Log details for each detected vein
LOG_VISUALIZATION_PATHS = True      # Log paths to saved visualizations
