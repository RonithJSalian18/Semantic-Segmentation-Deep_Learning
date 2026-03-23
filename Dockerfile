# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (important for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    tensorflow \
    scikit-learn \
    tqdm \
    albumentations

# Create output directory
RUN mkdir -p segmentation_results

# Run script
CMD ["python", "main.py"]
