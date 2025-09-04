FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04


RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    wget \
    procps \
    git-lfs \
    zip \
    unzip \
    htop \
    vim \
    nano \
    bzip2 \
    libx11-6 \
    build-essential \
    libsndfile-dev \
    software-properties-common \
 && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:flexiondotorg/nvtop && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends nvtop

RUN curl -sL https://deb.nodesource.com/setup_21.x  | bash - && \
    apt-get install -y nodejs && \
    npm install -g configurable-http-proxy

RUN pip install vllm

WORKDIR /app

COPY ./src/serve_model.py /app

EXPOSE 8000

CMD ["python", "serve_model.py"]
