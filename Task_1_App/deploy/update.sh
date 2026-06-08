#!/bin/bash
set -e

REPO_DIR=/var/www/Cygnus_Med_Demo
APP_DIR=$REPO_DIR/Task_1_App/backend
DB_PATH=$APP_DIR/chat_history.db

cd $REPO_DIR

# Preserve db before git pull so it's never deleted by a git operation
[ -f "$DB_PATH" ] && cp "$DB_PATH" /tmp/chat_history_backup.db

git pull origin main

# Restore db: prefer the preserved copy (has live data).
# Fall back to extracting from git history if no copy exists at all.
if [ -f /tmp/chat_history_backup.db ]; then
    cp /tmp/chat_history_backup.db "$DB_PATH"
elif [ ! -f "$DB_PATH" ]; then
    git show 8d8fb2ea:Task_1_App/backend/chat_history.db > "$DB_PATH"
    echo "Restored db from git history (8d8fb2ea)"
fi

$APP_DIR/venv/bin/pip install --no-cache-dir -r $APP_DIR/requirements.txt -q

systemctl restart shunt
echo "Deployed: $(date)"
