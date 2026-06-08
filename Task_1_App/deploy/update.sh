#!/bin/bash
set -e

REPO_DIR=/var/www/Cygnus_Med_Demo
APP_DIR=$REPO_DIR/Task_1_App/backend
DB_PATH=$APP_DIR/chat_history.db
RECOVERED_COMMIT="8d8fb2ea"
RECOVERED_GIT_PATH="Task_1_App/backend/chat_history.db"

cd $REPO_DIR

# Preserve db before git pull (in case git deletes it)
[ -f "$DB_PATH" ] && cp "$DB_PATH" /tmp/chat_history_pre_pull.db

git pull origin main

# Count sessions in current db (0 if missing, empty, or sqlite3 not available)
SESSION_COUNT=0
if [ -f "$DB_PATH" ]; then
    SESSION_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM sessions" 2>/dev/null || echo "0")
fi

# If db has no chat history, restore June 4 backup from git object store
if [ "$SESSION_COUNT" = "0" ] && git cat-file -e "$RECOVERED_COMMIT:$RECOVERED_GIT_PATH" 2>/dev/null; then
    git show "$RECOVERED_COMMIT:$RECOVERED_GIT_PATH" > "$DB_PATH"
    echo "Restored db from git commit $RECOVERED_COMMIT (June 4 backup, 280 sessions)"
fi

$APP_DIR/venv/bin/pip install --no-cache-dir -r $APP_DIR/requirements.txt -q

systemctl restart shunt
echo "Deployed: $(date)"
