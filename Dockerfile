FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install PyTorch (CPU for smaller image, GPU handled by HF Spaces runtime)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Copy the startup script
COPY deploy_start.py .

# Expose port (Hugging Face Spaces expects port 7860)
EXPOSE 7860

# Start the combined server
CMD ["python", "deploy_start.py"]
