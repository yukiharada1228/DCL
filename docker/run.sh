#!/bin/bash

# ユーザー情報の取得
nb_user=${USER}
nb_uid=$(id -u)
nb_gid=$(id -g)

# イメージ名とコンテナ名の設定
image_name=${USER}/dcl:1.2
container_name=dcl_${nb_user}

# 実行するコマンドの取得
if [ -z "$1" ]; then
  echo "Usage: $0 <command>"
  exit 1
fi
cmd="$@"

# Dockerコンテナの起動（バックグラウンド実行）
docker run \
    -d \
    --rm \
    --runtime=nvidia \
    --ipc=host \
    --net=host \
    -v ${HOME}/SCOPE:/workspace/SCOPE \
    --user ${nb_uid}:${nb_gid} \
    --name=${container_name} \
    ${image_name} bash -c "cd /workspace/SCOPE/DCL && $cmd"
