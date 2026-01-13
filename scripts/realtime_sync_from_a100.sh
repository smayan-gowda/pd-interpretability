#!/bin/bash
# real-time automatic sync from a100 using fswatch (macos)
# syncs files immediately when created/modified on remote

REMOTE="cc@192.5.86.144"
REMOTE_DIR="~/projects/pd-interpretability/results/"
LOCAL_DIR="/Volumes/usb drive/pd-interpretability/results/"
SSH_KEY="$HOME/.ssh/neuroscope-key"
SYNC_LOG="/tmp/a100_realtime_sync.log"

echo "starting real-time sync from a100..." | tee -a "$SYNC_LOG"
echo "remote: $REMOTE:$REMOTE_DIR" | tee -a "$SYNC_LOG"
echo "local: $LOCAL_DIR" | tee -a "$SYNC_LOG"
echo "" | tee -a "$SYNC_LOG"

# initial full sync
echo "[$(date '+%Y-%m-%d %H:%M:%S')] performing initial sync..." | tee -a "$SYNC_LOG"
rsync -az --update -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "$REMOTE:$REMOTE_DIR" "$LOCAL_DIR" 2>&1 | tee -a "$SYNC_LOG"

# continuous monitoring - poll every 5 seconds for changes
while true; do
    rsync -az --update -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$REMOTE:$REMOTE_DIR" "$LOCAL_DIR" 2>&1 | grep -v "^$" | while read line; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] $line" | tee -a "$SYNC_LOG"
    done

    sleep 5
done
