#!/bin/bash
# Build the WASM module and copy to web/pkg/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Building hyburn WASM module..."
wasm-pack build \
    --target web \
    --no-default-features \
    --features wasm \
    --out-dir web/pkg \
    --release

echo ""
echo "✓ Built successfully!"
echo "  Output: web/pkg/"
echo ""
echo "To serve locally:"
echo "  cd web && python3 -m http.server 8080"
echo "  Then open http://localhost:8080"
