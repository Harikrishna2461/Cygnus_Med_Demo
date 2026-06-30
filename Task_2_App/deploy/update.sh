#!/bin/bash
set -e

REPO_DIR=/var/www/Cygnus_Med_Demo
APP_DIR=$REPO_DIR/Task_2_App/backend

cd $REPO_DIR

git pull origin main

$APP_DIR/venv/bin/pip install --no-cache-dir -r $APP_DIR/requirements.txt -q

systemctl restart task2
echo "Deployed: $(date)"
