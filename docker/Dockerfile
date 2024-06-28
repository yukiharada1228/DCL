FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

# パッケージの更新と必要なツールのインストール
RUN apt-get update && apt-get install -y \
    sudo \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

CMD ["bash"]
