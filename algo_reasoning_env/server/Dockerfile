FROM python:3.11-slim

# Install Rust (minimal installation - rustc only)
RUN apt-get update && apt-get install -y curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy environment code (includes README.md for pyproject.toml readme field)
COPY algo_reasoning_env/ /app/algo_reasoning_env/
COPY complexity_reasoning_data/ /data/

# Install Python dependencies via pyproject.toml
RUN pip install --no-cache-dir -e /app/algo_reasoning_env

# Expose port
EXPOSE 7860

# Run the server
CMD ["uvicorn", "algo_reasoning_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
