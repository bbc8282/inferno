#!/bin/bash

# 프로세스 이름을 정의합니다.
PROCESS_NAME="src.api_server.worker"

# 해당 프로세스를 찾아 종료합니다. SIGTERM을 사용하여 안전하게 종료합니다.
pkill -f $PROCESS_NAME

# 프로세스가 완전히 종료될 때까지 대기합니다.
while pgrep -f $PROCESS_NAME > /dev/null; do
    echo "Waiting for the process to terminate..."
    sleep 1
done

echo "Process terminated. Restarting now..."

# 프로세스를 재시작합니다.
nohup python3 -m $PROCESS_NAME &

echo "Process restarted successfully."

