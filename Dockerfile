FROM python:3.12-slim

# System deps (only what pip wheels need; slim already has most)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching: rebuild only when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run as a non-root user
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

CMD ["python", "-m", "src.cli", "chat", "--role", "executive", "--name", "dana"]