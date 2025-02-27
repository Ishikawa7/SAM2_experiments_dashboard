# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the source code into the container.
COPY . .

RUN    apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN    python3 -m pip install --upgrade pip
RUN    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN    python3 -m pip install -r requirements.txt

# Expose the port that the application listens on.
EXPOSE 8080

# Run the application.
CMD ["python3", "app.py"]
