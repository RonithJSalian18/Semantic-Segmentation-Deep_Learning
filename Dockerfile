FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]