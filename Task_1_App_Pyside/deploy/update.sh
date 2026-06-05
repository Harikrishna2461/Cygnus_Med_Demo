#!/bin/bash
set -e

REPO_DIR=/var/www/Cygnus_Med_Demo
APP_DIR=$REPO_DIR/Task_1_App_Pyside/backend

cd $REPO_DIR
git pull origin main

$APP_DIR/venv/bin/pip install -r $APP_DIR/requirements.txt -q

systemctl restart shunt
echo "Deployed: $(date)"