FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y libopenblas-dev && rm -rf /var/lib/apt/lists/*
RUN pip install jax==0.4.23 jaxlib==0.4.23 numpy==1.26.4

# Set up working directory
WORKDIR /app
COPY . /app

# Create non-root user
RUN useradd -m sandbox && chown -R sandbox:sandbox /app
USER sandbox

# Directly execute Python code from stdin
CMD ["python", "-c", "import sys; exec(sys.stdin.read())"]