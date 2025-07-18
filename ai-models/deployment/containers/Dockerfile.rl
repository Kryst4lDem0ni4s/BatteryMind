# BatteryMind Reinforcement Learning Container
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including GPU support
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash batterymind
USER batterymind

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_TYPE=reinforcement_learning
ENV MODEL_VERSION=v1.0
ENV OMP_NUM_THREADS=4

# Copy requirements
COPY requirements.txt .
COPY requirements-rl.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-rl.txt

# Copy application code
COPY --chown=batterymind:batterymind . .

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data /app/rl_agents

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application
CMD ["python", "-m", "reinforcement_learning.agents.charging_agent", "--mode", "inference"]
