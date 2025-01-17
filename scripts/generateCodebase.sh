#!/bin/bash

# Path to the export repository relative to this script
EXPORT_REPO="/mnt/mldata/githubs/export-repository-to-prompt-for-llm"

# Activate virtual environment and ensure we deactivate it even if script fails
activate_venv() {
    source "$EXPORT_REPO/venv/bin/activate"
}

# Export both directories and combine them
export_and_combine() {
    # Export server (src) code
    #python "$EXPORT_REPO/export-repository-to-file.py" "../src"
    #mv output.txt server.txt

    # Export client code
    python "$EXPORT_REPO/export-repository-to-file.py" "../client"
    mv output.txt client.txt

    # Combine files and cleanup
    cat client.txt >> server.txt
    mv server.txt codebase.txt
    rm client.txt
}

# Export Docker and deployment configuration
export_docker_config() {
    echo -e "\n\n=== Docker Configuration ===\n" >> codebase.txt
    
    echo -e "\n--- docker-compose.yml ---\n" >> codebase.txt
    cat ../docker-compose.yml >> codebase.txt
    
    echo -e "\n--- Dockerfile ---\n" >> codebase.txt
    cat ../Dockerfile >> codebase.txt
    
    echo -e "\n--- nginx.conf ---\n" >> codebase.txt
    cat ../nginx.conf >> codebase.txt
    
    echo -e "\n--- scripts/launch-docker.sh ---\n" >> codebase.txt
    cat ../scripts/launch-docker.sh >> codebase.txt
    
    echo -e "\n--- scripts/start.sh ---\n" >> codebase.txt
    cat ../scripts/start.sh >> codebase.txt
}

# Export Docker network information
export_network_info() {
    echo -e "\n\n=== Docker Network Configuration ===\n" >> codebase.txt
    echo -e "\n--- docker network inspect docker_ragflow ---\n" >> codebase.txt
    docker network inspect docker_ragflow >> codebase.txt 2>/dev/null || echo "Unable to fetch network info - docker daemon not running or network doesn't exist" >> codebase.txt
}

# Main execution
if [ ! -d "$EXPORT_REPO" ]; then
    echo "Error: Export repository not found at $EXPORT_REPO"
    exit 1
fi

if [ ! -d "$EXPORT_REPO/venv" ]; then
    echo "Error: Virtual environment not found at $EXPORT_REPO/venv"
    exit 1
fi

# Execute the export process
activate_venv
export_and_combine
deactivate

# Add Docker configuration and network info
export_docker_config
export_network_info

echo "Successfully generated codebase.txt with Docker configuration and network info"
