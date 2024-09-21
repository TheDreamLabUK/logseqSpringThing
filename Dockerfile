# Use NVIDIA CUDA runtime base image with Ubuntu 22.04 and necessary libraries
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as builder

# Install necessary dependencies for building Rust applications
RUN apt-get update && apt-get install -y \
    build-essential \
    gnupg2 \
    curl \
    libssl-dev \
    pkg-config \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the default Rust toolchain to stable
RUN /bin/bash -c "source $HOME/.cargo/env && rustup default stable"

# Verify Rust installation
RUN /bin/bash -c "source $HOME/.cargo/env && rustc --version && cargo --version"

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./

# Copy the source code
COPY src ./src

# Copy settings.toml
COPY settings.toml ./

# Build the Rust application
RUN /bin/bash -c "source $HOME/.cargo/env && cargo build --release"

# Use the NVIDIA CUDA runtime in the final image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install necessary runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the built executable from the builder stage
COPY --from=builder /usr/src/app/target/release/webxr-graph /app/webxr-graph

# Copy Cargo from the builder stage
COPY --from=builder /root/.cargo /root/.cargo
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy settings.toml from the builder stage
COPY --from=builder /usr/src/app/settings.toml /app/settings.toml

# Expose port 8081
EXPOSE 8081

# Set the command to run the executable
CMD ["/app/webxr-graph"]
