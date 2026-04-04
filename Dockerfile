# Use a more complete base image that often includes common libraries
FROM python:3.10-slim

# Set environment variables to skip interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# We try the install again, but with a 'fix-missing' flag 
# which helps with Exit Code 100
RUN apt-get update --fix-missing && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your pothole detection code
COPY . .

# Hugging Face Spaces listens on 7860
EXPOSE 7860

# Start the app
CMD ["python", "app.py"]

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Start the app
CMD ["python", "app.py"]