FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libxss1 \
    libgconf-2-4 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]