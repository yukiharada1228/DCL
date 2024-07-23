#!/bin/bash

# ユーザー情報の取得
nb_user=${USER}
nb_uid=$(id -u)
nb_gid=$(id -g)

# イメージ名の設定
image_name=${USER}/dcl:1.2

# GPUデバイス番号の設定（デフォルトは0）
gpu_device=${GPU_DEVICE:-0}

# 再起動の制限回数とインターバルの設定
MAX_RESTARTS=3
RESTART_INTERVAL=60  # 秒

# 実行するコマンドの取得
if [ -z "$1" ]; then
  echo "Usage: $0 <command> [--second-instance]"
  echo "You can set GPU_DEVICE environment variable to specify GPU (default: 0)"
  exit 1
fi

# 2番目のインスタンスかどうかをチェック
if [[ "$*" == *"--second-instance"* ]]; then
  second_instance=true
  # --second-instanceを除去
  cmd=$(echo "$@" | sed 's/--second-instance//')
else
  second_instance=false
  cmd="$@"
fi

# コンテナ名の設定（GPUデバイス番号を含む）
if [ "$second_instance" = true ]; then
  container_name=dcl_${nb_user}_gpu${gpu_device}_2
else
  container_name=dcl_${nb_user}_gpu${gpu_device}
fi

# コンテナを起動し、終了したら再起動する関数
run_container() {
  restart_count=0
  last_restart_time=0

  while true; do
    current_time=$(date +%s)
    
    # 前回の再起動から一定時間以上経過していれば、再起動カウントをリセット
    if [ $((current_time - last_restart_time)) -ge $RESTART_INTERVAL ]; then
      restart_count=0
    fi

    echo "Starting container: ${container_name}"
    docker run \
      --rm \
      --ipc=host \
      --net=host \
      -v /raid/yukiharada/:/workspace \
      --user ${nb_uid}:${nb_gid} \
      --gpus "device=${gpu_device}" \
      --name=${container_name} \
      ${image_name} bash -c "cd /workspace/DCL && $cmd"
    
    exit_code=$?
    echo "Container ${container_name} exited with code ${exit_code}"
    
    if [ $exit_code -eq 0 ]; then
      echo "Container exited successfully. Not restarting."
      break
    else
      restart_count=$((restart_count + 1))
      last_restart_time=$current_time
      
      if [ $restart_count -ge $MAX_RESTARTS ]; then
        echo "Container has restarted $restart_count times in the last $RESTART_INTERVAL seconds. Stopping further restarts."
        break
      else
        echo "Container exited with an error. Restarting in 10 seconds... (Restart count: $restart_count)"
        sleep 10
      fi
    fi
  done
}

# コンテナを起動し、必要に応じて再起動する
run_container