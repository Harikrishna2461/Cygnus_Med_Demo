# CHIVA Clinical Assistant (Task 1) — Docker image
#
# Packages the Flask backend + frontend + Python dependencies into a single
# runnable image. Requires a separate Ollama instance for embeddings
# (see docker-compose.yml or DOCKER_GUIDE for the host-Ollama option) and
# a Groq API key supplied at runtime via GROQ_API_KEY.
#
# Build (from Task_1_Handover/):
#   docker build -f Docker/Dockerfile -t chiva-task1:latest .
#
# Run:
#   docker run -p 7860:7860 --env-file Docker/.env chiva-task1:latest

FROM python:3.11-slim

WORKDIR /app

# System deps needed by some Python packages (PyPDF2/openpyxl build chain, sqlite3 CLI for DB ops)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY Code/backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application code
COPY Code/backend ./backend
COPY Code/frontend ./frontend

# Runtime config — HOST must be 0.0.0.0 so the port is reachable from outside the container
ENV CMED_HOST=0.0.0.0
ENV CMED_PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

WORKDIR /app/backend
CMD ["python", "app.py"]
