#!/bin/bash
# Quick Start Script for Clinical Medical Decision Support Demo

set -e

echo "🏥 Clinical Medical Decision Support System"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found$(python3 --version)${NC}"

# Check Node
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js not found. Please install Node.js 16+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found: $(node --version)${NC}"

# Check Ollama
echo "🤖 Checking Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${YELLOW}⚠ Ollama not detected on localhost:11434${NC}"
    echo "  Start Ollama: ollama serve"
    echo "  Pull model: ollama pull mistral"
fi

echo ""
echo "📦 Setting up backend..."

# Backend setup
cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Installing Python dependencies..."
pip install -q -r requirements.txt

# Check for FAISS index
if [ ! -f "faiss_index/medical_index.faiss" ]; then
    echo ""
    echo -e "${YELLOW}📚 FAISS index not found. Ingesting medical data...${NC}"
    python ingest.py
    echo -e "${GREEN}✓ Medical data ingested${NC}"
else
    echo -e "${GREEN}✓ FAISS index found${NC}"
fi

cd ..

echo ""
echo -e "${GREEN}✓ Backend ready!${NC}"
echo ""
echo "📦 Setting up frontend..."

# Frontend setup
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies (this may take 2-3 minutes)..."
    npm install -q
fi

cd ..

echo -e "${GREEN}✓ Frontend ready!${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "📖 Next steps:"
echo ""
echo "1. Start Backend (Terminal 1):"
echo -e "   ${YELLOW}cd backend && source venv/bin/activate && python app.py${NC}"
echo ""
echo "2. Start Frontend (Terminal 2):"
echo -e "   ${YELLOW}cd frontend && npm start${NC}"
echo ""
echo "3. Open browser:"
echo -e "   ${YELLOW}http://localhost:3000${NC}"
echo ""
echo "=========================================="
