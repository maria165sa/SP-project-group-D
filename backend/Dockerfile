# Backend Dockerfile - VERSIÓN CORREGIDA
FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code SOLAMENTE (los models se montan con volume)
COPY api /app/api

# Copy el main.py si está en backend/
COPY main.py /app/main.py

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]