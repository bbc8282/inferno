#!/bin/bash

# 프로세스 이름을 정의합니다.
WORKER_PROCESS="src.api_server.worker"
APP_PROCESS="src.api_server.app"

# 함수 정의: 프로세스 종료 및 재시작
stop_and_restart() {
    local PROCESS=$1
    echo "Stopping $PROCESS..."
    pkill -f $PROCESS

    # 프로세스가 완전히 종료될 때까지 대기합니다.
    while pgrep -f $PROCESS > /dev/null; do
        echo "Waiting for $PROCESS to terminate..."
        sleep 1
    done

    echo "$PROCESS terminated. Restarting now..."

    # 프로세스를 재시작합니다.
    nohup python3 -m $PROCESS &

    echo "$PROCESS restarted successfully."
}

# worker 프로세스 종료 및 재시작
stop_and_restart $WORKER_PROCESS

# app 프로세스 종료 및 재시작
stop_and_restart $APP_PROCESS

echo "All processes have been restarted."