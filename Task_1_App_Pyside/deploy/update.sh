#!/bin/bash
set -e

APP_DIR=/var/www/shunt_classification_and_ligation

cd $APP_DIR
git pull origin main

source backend/venv/bin/activate
pip install -r backend/requirements.txt -q

sudo systemctl restart shunt
echo "Deployed: $(date)"
