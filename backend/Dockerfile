# **************************************
# Stage 1: The Builder
FROM python:3.9-slim-bookworm AS builder

WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .

# Install dependencies
RUN uv sync --no-install-project


# **************************************
# Stage 2: The Final Image
FROM python:3.9-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application
COPY main.py .

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]