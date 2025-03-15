# JAX-specific sandbox
FROM python:3.11-slim

# Security and performance settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPYTHONDONTWRITEBYTECODE=1 \
    JAX_PLATFORMS=cpu

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (JAX + minimal runtime)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    jax==0.4.23 \
    jaxlib==0.4.23 \
    numpy==1.26.4

# Sandbox setup
WORKDIR /sandbox
RUN useradd -r -s /bin/false sandbox && \
    chown -R sandbox:sandbox /sandbox

USER sandbox

# Entrypoint for code execution
CMD ["python", "/sandbox/run_jax.py"]