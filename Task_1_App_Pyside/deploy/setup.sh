#!/bin/bash
# First-time server setup for CHIVA Clinical Assistant on Tencent CVM (Ubuntu 22.04)
# Run as root: sudo bash deploy/setup.sh
set -e

APP_DIR=/var/www/chiva
LOG_DIR=/var/log/chiva

echo "=== Updating system ==="
apt-get update && apt-get upgrade -y
apt-get install -y python3-venv python3-pip nginx git curl

echo "=== Installing Ollama (for local embeddings) ==="
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama
sleep 5
ollama pull nomic-embed-text
echo "Ollama ready."

echo "=== Setting up app directory ==="
mkdir -p $APP_DIR $LOG_DIR
chown ubuntu:ubuntu $APP_DIR $LOG_DIR

echo "=== Cloning repository ==="
# Replace with your actual GitHub repo URL
# git clone https://github.com/YOUR_ORG/YOUR_REPO.git $APP_DIR
# If already cloned/copied:
chown -R ubuntu:ubuntu $APP_DIR

echo "=== Creating Python virtual environment ==="
cd $APP_DIR/backend
sudo -u ubuntu python3 -m venv venv
sudo -u ubuntu venv/bin/pip install --upgrade pip
sudo -u ubuntu venv/bin/pip install -r requirements.txt

echo "=== Creating .env from template ==="
if [ ! -f $APP_DIR/backend/.env ]; then
    cp $APP_DIR/.env.example $APP_DIR/backend/.env
    chown ubuntu:ubuntu $APP_DIR/backend/.env
    chmod 600 $APP_DIR/backend/.env
    echo ""
    echo ">>> ACTION REQUIRED: Edit $APP_DIR/backend/.env and add your GROQ_API_KEY <<<"
    echo ""
fi

echo "=== Installing systemd service ==="
cp $APP_DIR/deploy/chiva.service /etc/systemd/system/chiva.service
systemctl daemon-reload
systemctl enable chiva
systemctl start chiva

echo "=== Configuring nginx ==="
cp $APP_DIR/deploy/nginx.conf /etc/nginx/sites-available/chiva
ln -sf /etc/nginx/sites-available/chiva /etc/nginx/sites-enabled/chiva
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

echo ""
echo "=== Setup complete ==="
echo "App URL: http://$(curl -s ifconfig.me)"
echo "Check status: systemctl status chiva"
echo "View logs:    journalctl -u chiva -f"
