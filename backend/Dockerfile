# Base image
FROM python:3.9-slim-bookworm

WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .

RUN uv lock

# Install dependencies
RUN uv sync --no-install-project

# new virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application
COPY main.py .

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]