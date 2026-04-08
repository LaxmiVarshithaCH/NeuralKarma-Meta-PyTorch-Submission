FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create models directory if it doesn't exist
RUN mkdir -p /app/ml/models

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini

# Run the application
CMD ["python", "run.py", "--serve"]
