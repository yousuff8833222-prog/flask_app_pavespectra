# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables
# Hugging Face Spaces uses port 7860 by default. 
# Setting PORT=7860 ensures your app.py picks it up via os.environ.get("PORT").
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV
# libglib2.0-0 is required for gthread support even with headless OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app (respecting your .dockerignore)
COPY . .

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Command to run your Flask app
CMD ["python", "app.py"]
