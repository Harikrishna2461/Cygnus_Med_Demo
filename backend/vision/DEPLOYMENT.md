# Vision Module Deployment Guide

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Navigate to backend directory
cd backend

# Install all required packages
pip install -r requirements.txt

# Install SAM (Segment Anything Model)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install additional optional dependencies
pip install openai anthropic  # For Vision LLM support
```

### Step 2: Download SAM Model Weights

The SAM model requires pre-trained weights. Choose your model size:

```bash
# Create weights directory
mkdir -p weights

# Download SAM weights (choose one)

# Option 1: ViT-B (recommended, good balance)
wget -O weights/sam_vit_b.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b.pth

# Option 2: ViT-L (better accuracy, slower)
wget -O weights/sam_vit_l.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l.pth

# Option 3: ViT-H (best accuracy, requires GPU)
wget -O weights/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h.pth

# Option 4: Mobile SAM (fastest, lower accuracy)
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

Alternatively, weights are automatically downloaded on first use (slower first run).

### Step 3: Configure Vision Settings

Edit `vision/config.py`:

```python
# Choose appropriate model and device
SAM_MODEL_TYPE = "vit_b"          # Use "vit_l" or "vit_h" for better accuracy
SAM_DEVICE = "cpu"                # Use "cuda" if GPU available

# Enable LLM if API key available
LLM_ENABLED_DEFAULT = False       # Set to True to enable by default
LLM_PROVIDER_DEFAULT = "openai"   # openai or anthropic

# Adjust calibration for your ultrasound system
PIXELS_PER_MM = 1.0               # Adjust based on actual ultrasound probe/settings
```

### Step 4: Set Environment Variables

For LLM support:

```bash
# OpenAI API key
export OPENAI_API_KEY="sk-..."

# Or Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Flask secret key
export FLASK_SECRET_KEY="your-secret-key"
```

### Step 5: Start the Backend

```bash
# Run the backend
python app.py

# Server starts at http://localhost:5002
# Vision endpoints available at http://localhost:5002/api/vision/
```

## 🔧 Advanced Configuration

### GPU Support

To use NVIDIA GPU acceleration:

```bash
# Install GPU-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Update config
python -c "from vision import config; config.SAM_DEVICE='cuda'"
```

Check GPU availability:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Performance Tuning

For different deployment scenarios:

**High-Speed Processing (Lower Accuracy)**
```python
# vision/config.py
SAM_MODEL_TYPE = "mobile_sam"
MAX_FRAMES_PER_VIDEO = 100
BATCH_SIZE = 8
DEFAULT_TARGET_FPS = 2
```

**High-Accuracy Processing (Slower)**
```python
SAM_MODEL_TYPE = "vit_h"
SAM_DEVICE = "cuda"
DEFAULT_TARGET_FPS = 5
BATCH_SIZE = 2
```

**Balanced (Recommended)**
```python
SAM_MODEL_TYPE = "vit_b"
SAM_DEVICE = "cuda"  # or "cpu"
DEFAULT_TARGET_FPS = 3
BATCH_SIZE = 4
```

### Memory Optimization

For limited memory environments:

```python
# vision/config.py
MAX_CACHE_SIZE_MB = 100           # Reduce cache
DEFAULT_RESIZE_SHAPE = (480, 360) # Resize frames
MAX_FRAMES_PER_VIDEO = 50         # Process fewer frames
```

## 🧪 Testing Deployment

### Health Check

```bash
# Check vision module health
curl http://localhost:5002/api/vision/health

# Expected response:
# {
#   "status": "healthy",
#   "module": "vision_vein_detection",
#   "capabilities": [...]
# }
```

### Test Frame Analysis

```bash
# Test with sample image
curl -X POST http://localhost:5002/api/vision/analyze-frame \
  -F "file=@ultrasound_sample.jpg" \
  -F "enable_llm=false"
```

### Test Video Processing

```bash
# Test with sample video
curl -X POST http://localhost:5002/api/vision/detect-veins \
  -F "file=@ultrasound_sample.mp4" \
  -F "max_frames=5" \
  -F "enable_llm=false"
```

## 📦 Docker Deployment

### Dockerfile Addition (Optional)

Add to backend/Dockerfile:

```dockerfile
# Install vision dependencies
RUN pip install opencv-python torch torchvision scipy

# Download SAM weights
RUN mkdir -p /app/weights && \
    wget -O /app/weights/sam_vit_b.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b.pth

# Copy vision module
COPY vision/ /app/vision/
```

### Docker Compose

Update docker-compose.yml:

```yaml
backend:
  build:
    context: ./backend
  environment:
    FLASK_ENV: production
    SAM_DEVICE: cpu
    OPENAI_API_KEY: ${OPENAI_API_KEY}
  volumes:
    - ./backend/weights:/app/weights
    - ./backend/vision:/app/vision
  ports:
    - "5002:5002"
  deploy:
    resources:
      limits:
        memory: 4G
        cpus: '2'
```

Run with Docker:

```bash
docker-compose up -d backend
```

## 🔍 Monitoring & Logging

### Enable Detailed Logging

```python
# vision/config.py
LOG_LEVEL = "DEBUG"
LOG_VEIN_DETAILS = True
LOG_VISUALIZATION_PATHS = True
```

Check logs:

```bash
# Tail backend logs
tail -f backend.log

# Filter for vision operations
grep "vision\|vein\|fascia" backend.log
```

### Performance Metrics

The system automatically logs:
- Processing time per frame
- Number of veins detected
- Segmentation confidence scores
- Classification accuracy
- LLM API calls and responses

Access via MLOps endpoint:

```bash
curl http://localhost:5002/api/mlops/runs/Vein Detection
```

## 🚨 Troubleshooting

### ImportError: SAM not found

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### CUDA out of memory

```python
# vision/config.py
SAM_DEVICE = "cpu"  # Fall back to CPU
DEFAULT_RESIZE_SHAPE = (480, 360)  # Reduce frame size
```

### LLM API errors

Check API key:
```bash
echo $OPENAI_API_KEY
```

Test API connection:
```python
from openai import OpenAI
client = OpenAI(api_key="sk-...")
print(client.models.list())
```

### Slow processing

- Reduce `max_frames`
- Disable LLM (`enable_llm=false`)
- Use GPU if available
- Reduce frame resolution

### Missing SAM weights

Weights are auto-downloaded, but if stuck:

```bash
# Clear cache
rm -rf ~/.cache/torch/

# Re-download SAM weights
python -c "from segment_anything import sam_model_registry; sam_model_registry['vit_b']"
```

## 📊 Performance Benchmarks

Typical processing times (per frame):

| Operation | ViT-B (CPU) | ViT-B (GPU) | Mobile SAM |
|-----------|-----------|-----------|-----------|
| Segmentation | 2-3s | 0.3-0.5s | 0.1-0.2s |
| Spatial Analysis | 0.1s | 0.1s | 0.1s |
| Classification | 0.05s | 0.05s | 0.05s |
| LLM Analysis | 5-15s | 5-15s | 5-15s |
| Visualization | 0.2s | 0.2s | 0.2s |

**Total times (per frame):**
- CPU without LLM: ~2.4 seconds
- GPU without LLM: ~0.8 seconds
- With Vision LLM: +10 seconds

## 🔐 Security Checklist

- [ ] API keys stored in environment variables
- [ ] File upload size limits enforced (500MB default)
- [ ] File type validation (mp4, avi, jpg, etc.)
- [ ] Temporary files cleaned up after processing
- [ ] Rate limiting implemented (adjust as needed)
- [ ] CORS configured properly
- [ ] HTTPS enabled in production
- [ ] Input sanitization applied

## 📈 Scaling Considerations

### Horizontal Scaling

Deploy multiple backend instances:

```yaml
# docker-compose.yml
backend-1:
  image: backend:latest
  ports:
    - "5001:5002"

backend-2:
  image: backend:latest
  ports:
    - "5003:5002"

nginx:
  image: nginx:latest
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
```

### Load Balancing

Use nginx to distribute requests:

```nginx
upstream backend {
  server backend-1:5002;
  server backend-2:5002;
}

server {
  listen 80;
  location /api/vision/ {
    proxy_pass http://backend;
  }
}
```

### Async Processing

For large videos, consider async processing:

```python
# Use Celery or similar
from celery import Celery

celery_app = Celery('vision')

@celery_app.task
def process_video_async(video_path, enable_llm=False):
    from vision.vision_main import process_ultrasound_video
    return process_ultrasound_video(video_path, enable_llm=enable_llm)
```

## 📝 Maintenance

### Regular Updates

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Update SAM model
pip install --upgrade git+https://github.com/facebookresearch/segment-anything.git
```

### Cleanup

```bash
# Remove cached models
rm -rf ~/.cache/torch/
rm -rf weights/*.pth

# Clear temporary files
rm -rf /tmp/vision_*

# Clean old logs
find logs/ -mtime +30 -delete
```

### Backup

```bash
# Backup weights
tar -czvf sam_weights_backup.tar.gz weights/

# Backup configuration
cp vision/config.py vision/config.py.backup
```

## 📞 Support

For issues:

1. Check logs: `grep "ERROR\|WARNING" backend.log`
2. Review config: `cat vision/config.py`
3. Test health: `curl /api/vision/health`
4. Check examples: `python vision/examples.py`
5. Read docs: [Vision README](./README.md)

---

For more details, refer to:
- [Vision README](./README.md)
- [Frontend Integration Guide](./FRONTEND_INTEGRATION.md)
- [Configuration Reference](./config.py)
- [Examples](./examples.py)
