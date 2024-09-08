# Use an official Rust image as the base image
FROM rust:1.67 as builder

# Install NVIDIA GPU support
RUN apt-get update && apt-get install -y \
    libcuda1-384 \
    nvidia-cuda-toolkit

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./

# Copy the source code
COPY src ./src

# Build the application
RUN cargo build --release

# Use a smaller base image for the final image
FROM debian:buster-slim

# Install NVIDIA GPU support
RUN apt-get update && apt-get install -y \
    libcuda1-384 \
    nvidia-cuda-toolkit

# Copy the built executable from the builder stage
COPY --from=builder /usr/src/app/target/release/webxr-graph /usr/local/bin/webxr-graph

# Set the command to run the executable
CMD ["webxr-graph"]