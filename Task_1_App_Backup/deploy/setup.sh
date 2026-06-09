#!/bin/bash
set -e

APP_DIR=/var/www/shunt_classification_and_ligation
LOG_DIR=/var/log/shunt_classification_and_ligation

echo "=== Updating system ==="
apt-get update && apt-get upgrade -y
apt-get install -y python3-venv python3-pip nginx git curl

echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama
sleep 5
ollama pull nomic-embed-text

echo "=== Setting up directories ==="
mkdir -p $APP_DIR $LOG_DIR

echo "=== Creating Python virtual environment ==="
cd $APP_DIR/backend
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo "=== Creating .env ==="
if [ ! -f $APP_DIR/backend/.env ]; then
    cp $APP_DIR/.env.example $APP_DIR/backend/.env
    chmod 600 $APP_DIR/backend/.env
    echo ">>> ACTION REQUIRED: Edit $APP_DIR/backend/.env and add your GROQ_API_KEY <<<"
fi

echo "=== Installing systemd service ==="
cp $APP_DIR/deploy/shunt.service /etc/systemd/system/shunt.service
systemctl daemon-reload
systemctl enable shunt
systemctl start shunt

echo "=== Configuring nginx ==="
cp $APP_DIR/deploy/nginx.conf /etc/nginx/sites-available/shunt
ln -sf /etc/nginx/sites-available/shunt /etc/nginx/sites-enabled/shunt
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

echo "=== Setup complete ==="
echo "App URL: http://$(curl -s ifconfig.me)"
