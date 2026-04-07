# Dockerfile for HuggingFace Spaces deployment
# Optimized for HF Spaces which run as root by default

FROM python:3.11-slim

# Install Rust (minimal installation with rustc only)
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

# Install the algo_reasoning_env package in editable mode
COPY algo_reasoning_env/ /app/algo_reasoning_env/
RUN pip install --no-cache-dir -e /app/algo_reasoning_env

# Copy dataset files to /data/
COPY complexity_reasoning_data/ /data/

# Copy entry point
COPY app.py .

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Run the server
# HF Spaces automatically starts containers with: python app.py
CMD ["python", "app.py"]
