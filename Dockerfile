# Use a lightweight Python base image
FROM python:3.12-slim-trixie

# Copy uv (package manager) binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /ask-your-files

# Copy only dependency files first for caching
COPY uv.lock pyproject.toml ./

# Install dependencies using uv (respects lock file)
RUN uv sync --frozen --no-cache

# Now copy the rest of your source code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Default command
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]


