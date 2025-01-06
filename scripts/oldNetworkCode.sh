#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <git-commit-hash>"
    exit 1
fi

COMMIT_HASH=$1
# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/scripts/oldCode"
TEMP_FILE="$PROJECT_ROOT/scripts/network-files.txt"

# Create a list of files to extract
cat << EOF > "$TEMP_FILE"
src/handlers/visualization_handler.rs
src/handlers/socket_flow_handler.rs
src/main.rs
src/utils/socket_flow_messages.rs
src/utils/socket_flow_constants.rs
src/types/mod.rs
client/state/settings.ts
client/state/graphData.ts
client/index.ts
client/core/constants.ts
client/state/defaultSettings.ts
client/websocket/websocketService.ts
client/xr/xrInteraction.ts
vite.config.ts
nginx.conf
docker-compose.yml
scripts/launch-docker.sh
EOF

# Clean up any existing output directory
rm -rf "$OUTPUT_DIR"

# Create the output directory structure
mkdir -p "$OUTPUT_DIR/src/handlers"
mkdir -p "$OUTPUT_DIR/src/utils"
mkdir -p "$OUTPUT_DIR/src/types"
mkdir -p "$OUTPUT_DIR/client/state"
mkdir -p "$OUTPUT_DIR/client/core"
mkdir -p "$OUTPUT_DIR/client/websocket"
mkdir -p "$OUTPUT_DIR/client/xr"
mkdir -p "$OUTPUT_DIR/scripts"

# Change to project root for git operations
cd "$PROJECT_ROOT"

# Extract each file individually to maintain proper directory structure
while IFS= read -r file; do
    # Create the output path
    output_path="$OUTPUT_DIR/$file"
    # Extract the file content from git and save it to the proper location
    git show "$COMMIT_HASH:$file" > "$output_path" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Extracted: $file"
    else
        echo "Warning: Could not extract $file"
    fi
done < "$TEMP_FILE"

# Special handling for root-level files
git show "$COMMIT_HASH:vite.config.ts" > "$OUTPUT_DIR/vite.config.ts"
git show "$COMMIT_HASH:nginx.conf" > "$OUTPUT_DIR/nginx.conf"
git show "$COMMIT_HASH:docker-compose.yml" > "$OUTPUT_DIR/docker-compose.yml"

# Cleanup temporary files
rm -f "$TEMP_FILE"

echo "Network code from commit $COMMIT_HASH has been extracted to scripts/oldCode/"
echo "To remove the extracted code, run: rm -rf scripts/oldCode"
