#!/bin/bash

# Extract port from YAML
PORT=$(awk -F: '/fastapi_port/{gsub(/ /,"",$2); print $2}' config/settings.yaml)

# Build Docker image
docker build -t ask_your_files .

# Run container using the extracted port
docker run -p ${PORT}:8000 --env-file .env ask_your_files
