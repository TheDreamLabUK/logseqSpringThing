#!/bin/bash

# Create necessary directories
mkdir -p server/src/{controllers,services,models,utils}
mkdir -p public/js/{components,services,threeJS,xr}
mkdir -p tests/{server,client}

# Move server-side files
mv src/controllers/* server/src/controllers/
mv src/services/* server/src/services/
mv src/models/* server/src/models/
mv src/utilities/* server/src/utils/
mv server/src/app.js server/src/
mv server/src/index.js server/src/
mv server/src/server.js server/src/

# Move client-side files
mv public/js/components/* public/js/components/
mv public/js/services/* public/js/services/
mv public/js/threeJS/* public/js/threeJS/
mv public/js/xr/* public/js/xr/
mv public/js/app.js public/js/
mv public/js/index.js public/js/
mv public/js/gpuUtils.js public/js/

# Move test files
mv tests/server/* tests/server/
mv tests/client/* tests/client/

# Remove empty directories
rm -rf src client

# Rename files to match README.md if necessary
mv server/src/utils/websocketUtils.js server/src/utils/websocketUtils.js
mv public/js/components/webXRVisualization.js public/js/components/webXRVisualization.js
mv public/js/components/graphSimulation.js public/js/components/graphSimulation.js
mv public/js/components/chatManager.js public/js/components/chatManager.js

# Print the new structure
echo "New project structure:"
tree -L 3 .

echo "Script completed. Please review the changes and update README.md accordingly."
