#!/bin/bash

# Create cache directory if it doesn't exist
mkdir -p /tmp/docker-cache

# Build and run with BuildKit enabled
DOCKER_BUILDKIT=1 docker-compose -f docker-compose.dev.yml up --build