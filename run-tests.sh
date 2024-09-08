#!/bin/bash

# Run Jest and capture the output
OUTPUT=$(npm test -- --verbose 2>&1)

# Print the captured output
echo "$OUTPUT"

# Check if the tests passed
if echo "$OUTPUT" | grep -q "PASS"; then
  echo "Tests passed successfully!"
  exit 0
else
  echo "Tests failed!"
  exit 1
fi