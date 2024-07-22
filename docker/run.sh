#!/bin/bash

# ユーザー情報の取得
nb_user=${USER}
nb_uid=$(id -u)
nb_gid=$(id -g)

# イメージ名の設定
image_name=${USER}/dcl:1.2

# GPUデバイス番号の設定（デフォルトは0）
gpu_device=${GPU_DEVICE:-0}

# コンテナ名の設定（GPUデバイス番号を含む）
container_name=dcl_${nb_user}_gpu${gpu_device}

# 実行するコマンドの取得
if [ -z "$1" ]; then
  echo "Usage: $0 <command>"
  echo "You can set GPU_DEVICE environment variable to specify GPU (default: 0)"
  exit 1
fi
cmd="$@"

# Dockerコンテナの起動（バックグラウンド実行）
docker run \
    -d \
    --rm \
    --ipc=host \
    --net=host \
    -v /raid/yukiharada/:/workspace \
    --user ${nb_uid}:${nb_gid} \
    --gpus "device=${gpu_device}" \
    --name=${container_name} \
    ${image_name} bash -c "cd /workspace/DCL && $cmd"