#!/bin/bash
set -e
set -u
set -o pipefail

echo "Cleanup data..."
BUILD_DIR=$1
cd $BUILD_DIR && find . -type f ! -name 'config.cmake' -delete && cd ..
