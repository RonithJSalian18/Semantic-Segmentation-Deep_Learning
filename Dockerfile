FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# install system packages required by OpenCV (libxcb and related)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libx11-6 libxcb1 libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "train.py"]