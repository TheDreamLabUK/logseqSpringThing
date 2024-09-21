#!/bin/bash

# Stop and remove existing container
docker stop logseqXR || true
docker rm logseqXR || true

# Build the Docker image
docker build -t logseq-xr-image .

# Run the Docker container with the correct environment variables
docker run -d --name logseqXR \
  -e TOPICS="Artificial Intelligence,Machine Learning,Rust Programming,Web Development,WebXR,Three.js,GPU Computing,Graph Visualization,Markdown Processing" \
  logseq-xr-image
