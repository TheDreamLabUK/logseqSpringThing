#!/bin/bash

# Path to the export repository relative to this script
EXPORT_REPO="/mnt/mldata/githubs/export-repository-to-prompt-for-llm"

# Activate virtual environment and ensure we deactivate it even if script fails
activate_venv() {
    source "$EXPORT_REPO/venv/bin/activate"
}


# Export Docker and deployment configuration
export_docker_config() {
    # Add project metadata
    echo -e "# Project Codebase\n" >> codebase.txt
    echo -e "Generated: $(date)\n" >> codebase.txt
    echo -e "## Project Structure\n" >> codebase.txt
    echo -e "- Server: Rust (src directory)\n- Client: TypeScript (client directory)\n" >> codebase.txt

    echo -e "\n## README.md\n" >> codebase.txt
    cat ../README.md >> codebase.txt

    echo -e "\n\n## Docker Configuration\n" >> codebase.txt
    
    # Add each file with proper headers
    for file in "../docker-compose.yml" "../docker-compose.dev.yml" "../docker-compose.production.yml" \
                "../Dockerfile" "../Dockerfile.dev" "../Dockerfile.production" \
                "../nginx.conf" "../nginx.dev.conf" \
                "../data/settings.yaml" "../.dockerignore"; do
        if [ -f "$file" ]; then
            echo -e "\n### $(basename $file)\n" >> codebase.txt
            cat "$file" >> codebase.txt
        else
            echo -e "\n### $(basename $file) - MISSING\n" >> codebase.txt
        fi
    done

    # Add package management and config files
    echo -e "\n\n## Configuration Files\n" >> codebase.txt
    for file in "../Cargo.toml" "../client/package.json" "../client/vite.config.ts" "../.env.template" \
                "../tsconfig.json" "../.eslintrc" "../.gitignore" \
                "../client/tsconfig.json"; do
        if [ -f "$file" ]; then
            echo -e "\n### $(basename $file)\n" >> codebase.txt
            cat "$file" >> codebase.txt
        else
            echo -e "\n### $(basename $file) - MISSING\n" >> codebase.txt
        fi
    done

    # Export docker directory if it exists
    if [ -d "../docker" ]; then
        echo -e "\n\n## Docker Directory Contents\n" >> codebase.txt
        for file in ../docker/*; do
            if [ -f "$file" ]; then
                echo -e "\n### docker/$(basename $file)\n" >> codebase.txt
                cat "$file" >> codebase.txt
            fi
        done
    fi

    echo -e "\n\n## Important Scripts\n" >> codebase.txt
    for script_file in "../scripts/dev-entrypoint.sh" "../scripts/dev.sh" \
                       "../scripts/launch-production.sh" "../scripts/start.sh" \
                       "../scripts/compile_ptx.sh" "../scripts/diagnostics.sh" \
                       "../scripts/check-rust-logs.sh" "../scripts/check-vite.sh" \
                       "../scripts/test.sh" "../scripts/generateCodebase.sh"; do
        if [ -f "$script_file" ]; then
            echo -e "\n### scripts/$(basename $script_file)\n" >> codebase.txt
            cat "$script_file" >> codebase.txt
        else
            echo -e "\n### scripts/$(basename $script_file) - MISSING\n" >> codebase.txt
        fi
    done

    # Vite environment files (example, adjust if actual files differ)
    echo -e "\n\n## Vite Environment Configuration\n" >> codebase.txt
    # Add client .env files if they exist and are relevant (usually gitignored)
    # For now, we'll just note that vite.config.ts and .env.template are captured.
    # If specific .env files like .env.development or .env.production exist at client root, add them.
    echo -e "Vite configuration is primarily in 'client/vite.config.ts' (captured above)." >> codebase.txt
    echo -e "Environment variable templates are in '.env.template' (captured above)." >> codebase.txt
    # Example for specific vite env files if they were used:
    # for vite_env_file in "../client/.env.development" "../client/.env.production"; do
    #     if [ -f "$vite_env_file" ]; then
    #         echo -e "\n### client/$(basename $vite_env_file)\n" >> codebase.txt
    #         cat "$vite_env_file" >> codebase.txt
    #     else
    #         echo -e "\n### client/$(basename $vite_env_file) - MISSING (or not applicable)\n" >> codebase.txt
    #     fi
    # done

}

# Export Docker network information
export_network_info() {
    echo -e "\n\n=== Docker Network Configuration ===\n" >> codebase.txt
    echo -e "\n--- docker network inspect docker_ragflow ---\n" >> codebase.txt
    docker network inspect docker_ragflow >> codebase.txt 2>/dev/null || echo "Unable to fetch network info - docker daemon not running or network doesn't exist" >> codebase.txt
}

# Export both directories and combine them
export_and_combine() {
    # Export server (src) code
    python "$EXPORT_REPO/export-repository-to-file.py" "../src"
    mv output.txt server.txt

    # Export client code
    python "$EXPORT_REPO/export-repository-to-file.py" "../client"
    mv output.txt client.txt

    # Export docs code
    python "$EXPORT_REPO/export-repository-to-file.py" "../docs"
    mv output.txt docs.txt

    # Combine files with clear separation
    echo -e "\n\n## Server Code (Rust)\n" >> codebase.txt
    cat server.txt >> codebase.txt
    
    echo -e "\n\n## Client Code (TypeScript)\n" >> codebase.txt
    cat client.txt >> codebase.txt
    
    echo -e "\n\n## Documentation\n" >> codebase.txt
    cat docs.txt >> codebase.txt
    
    rm server.txt
    rm client.txt
    rm docs.txt
}

# Add project structure information
export_project_structure() {
    echo -e "\n## Project Structure Tree\n" >> codebase.txt
    echo -e "\`\`\`" >> codebase.txt
    # Show root level files
    echo "Root files:" >> codebase.txt
    ls -p ../ | grep -v / >> codebase.txt
    echo -e "\nDirectories:" >> codebase.txt
    # Show client, src, and docs directories structure
    tree -I 'node_modules|target|dist|.git|venv' ../client ../src ../docs >> codebase.txt
    echo -e "\`\`\`\n" >> codebase.txt
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

# Add Docker configuration and network info
export_project_structure
export_docker_config
export_network_info

export_and_combine
deactivate



echo "Successfully generated codebase.txt with Docker configuration and network info"
