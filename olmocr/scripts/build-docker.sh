#!/bin/bash

set -e

VERSION=$(python -c 'import olmocr.version; print(olmocr.version.VERSION)')
echo "$VERSION"

docker build --platform linux/amd64 -f ./scripts/beaker/Dockerfile-inference  -t olmocr-inference-$VERSION .
beaker image create --workspace ai2/oe-data-pdf --name olmocr-inference-$VERSION olmocr-inference-$VERSION