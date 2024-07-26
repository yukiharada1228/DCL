FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# パッケージの更新と必要なツールのインストール
RUN apt-get update && apt-get install -y \
    sudo \
    vim \
    tmux \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

CMD ["bash"]
