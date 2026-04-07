# Dockerfile for HuggingFace Spaces deployment
# This Dockerfile is optimized for HF Spaces which run as root by default

FROM python:3.11-slim

# Install Rust (minimal installation)
RUN apt-get update && apt-get install -y curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Run the server
# HF Spaces automatically starts containers with: python app.py
CMD ["python", "-m", "uvicorn", "algo_reasoning_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
