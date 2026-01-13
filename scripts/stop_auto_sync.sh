#!/bin/bash
# stop the automatic sync process

PID=$(pgrep -f "auto_sync_from_chameleon.sh")

if [ -z "$PID" ]; then
    echo "no auto sync process found"
    exit 1
fi

kill $PID
echo "stopped auto sync process (pid: $PID)"
