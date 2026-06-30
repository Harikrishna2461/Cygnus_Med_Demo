#!/bin/bash
set -e

REPO_DIR=/var/www/Cygnus_Med_Demo
APP_DIR=$REPO_DIR/Task_2_App/backend
SOCKET_DIR=/var/www/task2
LOG_DIR=/var/log/task2_probe_guidance

echo "=== Updating system ==="
apt-get update && apt-get upgrade -y
apt-get install -y python3-venv python3-pip nginx git curl

echo "=== Setting up directories ==="
mkdir -p $SOCKET_DIR $LOG_DIR

echo "=== Creating Python virtual environment ==="
cd $APP_DIR
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo "=== Creating .env ==="
if [ ! -f $APP_DIR/../.env ]; then
    cp $APP_DIR/../.env.example $APP_DIR/../.env
    chmod 600 $APP_DIR/../.env
    echo ">>> ACTION REQUIRED: Edit $APP_DIR/../.env and add your GROQ_API_KEY and STREAM_VIDEO_PATH <<<"
fi

echo "=== Installing systemd service ==="
cp $REPO_DIR/Task_2_App/deploy/task2.service /etc/systemd/system/task2.service
systemctl daemon-reload
systemctl enable task2
systemctl start task2

echo "=== Configuring nginx ==="
cp $REPO_DIR/Task_2_App/deploy/nginx.conf /etc/nginx/sites-available/task2
ln -sf /etc/nginx/sites-available/task2 /etc/nginx/sites-enabled/task2
nginx -t
systemctl restart nginx

echo "=== Setup complete ==="
echo "App URL: http://$(curl -s ifconfig.me):8080"
echo ""
echo "IMPORTANT: Upload your stream video to the server and set STREAM_VIDEO_PATH in $APP_DIR/../.env"
