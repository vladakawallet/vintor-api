FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONBUFFERED=1
ENV PYTHONDONTWRITEBYCODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    sudo add-apt-repository ppa:deadsnakes/ppa \
    sudo apt install python3.12-full \
    git \ 
    wget \ 
    curl \ 
    redis-server \ 
    supervisor \ 
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.12 /usr/bin/python

RUN python -m pip install --upgrade pip

RUN git clone https://github.com/ultralytics/yolov5.git yolov5
RUN cd yolov5 && \
    pip install -r requirements.txt && \
    cd ..

RUN python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ --timeout 60


COPY requirements.txt . 
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . . 

RUN mkdir -p models /app/logs /var/log/supervisor

COPY models/ models/

COPY supervisord.conf /etc/supervisor/conf.d/supervisor.conf

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]

