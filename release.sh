#!/bin/bash

# Ensure script stops on first error
set -e

# Check if version type is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <major|minor|patch>"
    exit 1
fi

VERSION_TYPE=$1

# Validate version type
if [[ ! $VERSION_TYPE =~ ^(major|minor|patch)$ ]]; then
    echo "Error: Version type must be 'major', 'minor', or 'patch'"
    exit 1
fi

echo "Starting release process..."

# Ensure we're on the master branch
git checkout master
git pull

# Run tests
echo "Running tests..."
pytest

# Run type checking
echo "Running type checking..."
mypy STanalysis

# Update version
echo "Bumping $VERSION_TYPE version..."
bumpver update --$VERSION_TYPE

# Make sure git is up to date with version changes
git add .
git commit --amend --no-edit

# Install the updated package
echo "Installing updated package..."
uv pip install -e ".[dev]"

# Run tests again to verify version match
echo "Verifying version update..."
pytest

# Build package
echo "Building package..."
python -m build

# Upload to PyPI
echo "Uploading to PyPI..."
if [ -z "$TWINE_USERNAME" ] || [ -z "$TWINE_PASSWORD" ]; then
    echo "Error: TWINE_USERNAME and TWINE_PASSWORD environment variables must be set"
    exit 1
fi

python -m twine upload --non-interactive dist/*

echo "Release process completed successfully!"
