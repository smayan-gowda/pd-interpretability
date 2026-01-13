#!/bin/bash
# automatic continuous sync from chameleon cloud to local workspace
# runs in background and syncs results directory every 30 seconds

REMOTE="cc@192.5.86.214"
REMOTE_DIR="~/projects/pd-interpretability/results/"
LOCAL_DIR="/Volumes/usb drive/pd-interpretability/results/"
SSH_KEY="$HOME/.ssh/neuroscope-key"
SYNC_INTERVAL=30

echo "starting automatic sync from chameleon cloud..."
echo "remote: $REMOTE:$REMOTE_DIR"
echo "local: $LOCAL_DIR"
echo "interval: ${SYNC_INTERVAL}s"
echo ""

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # sync results directory
    rsync -az --update -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$REMOTE:$REMOTE_DIR" "$LOCAL_DIR" 2>&1 | grep -v "^$" | while read line; do
        echo "[$timestamp] $line"
    done

    sleep $SYNC_INTERVAL
done
