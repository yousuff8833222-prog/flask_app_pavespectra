FROM python:3.11-slim

WORKDIR /app

# We no longer need the complex apt-get install for libgl1
# because we are using opencv-python-headless!

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces port
EXPOSE 7860

CMD ["python", "app.py"]