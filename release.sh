#!/bin/bash

# Simple release script for NetCDFKit
# Usage: ./release.sh [major|minor|patch] (default: patch)

set -e

BUMP_TYPE=${1:-patch}

# Bump version
uv run bumpver update --$BUMP_TYPE

# Build package
uv build

# Upload to PyPI
uv publish
