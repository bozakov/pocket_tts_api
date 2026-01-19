FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Disable development dependencies
ENV UV_NO_DEV=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg

# Install Python dependencies first (for better caching)
RUN pip install --no-cache-dir --upgrade pip

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Sync the project into a new environment, asserting the lockfile is up to date


WORKDIR /app
# Copy the project into the image
COPY ./pyproject.toml /app/
RUN uv sync

COPY ./pocketapi.py /app/

# Set default command to start API server
CMD ["uv", "run", "pocketapi.py", "--host", "0.0.0.0", "--port", "8000"]
# 
# Expose any ports if needed (for future web interface)
EXPOSE 8000

