#!/bin/bash

# 実行するコマンドの設定
BASE_CMD="./docker/run.sh ipython ./train.py -- --num_nodes=3 --target_model=ResNet32 --dataset=CIFAR100 --num_trial=1500 --optuna_dir=./result/"

# 0から7までのGPUデバイスでループ
for gpu_id in {0..7}
do
    echo "Starting execution on GPU $gpu_id"
    
    # 1番目のインスタンス
    GPU_DEVICE=$gpu_id bash -c "GPU_DEVICE=$gpu_id $BASE_CMD --gpu_id=0" &
    
    # 2番目のインスタンス
    GPU_DEVICE=$gpu_id bash -c "GPU_DEVICE=$gpu_id $BASE_CMD --gpu_id=0 --second-instance" &
    
    # 各実行の間に少し待機時間を入れる（オプション）
    sleep 5
done

# バックグラウンドジョブが終了するのを待たない
echo "All GPU runs started. They will automatically restart if they crash."