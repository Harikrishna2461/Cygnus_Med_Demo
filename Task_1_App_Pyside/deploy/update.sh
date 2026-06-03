#!/bin/bash
# Pull latest code and restart service (run as ubuntu user)
set -e

APP_DIR=/var/www/chiva

cd $APP_DIR
git pull origin main

source backend/venv/bin/activate
pip install -r backend/requirements.txt -q

sudo systemctl restart chiva
echo "Deployed: $(date)"
