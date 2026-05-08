#!/bin/bash
# Build the WASM module and copy to web/pkg/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

# Step 1: Regenerate presets from example TOML/NPY files
echo "Generating presets from examples/..."
python3 "$PROJECT_DIR/scripts/gen_presets.py"

# Step 2: Build WASM module
echo "Building hyburn WASM module..."
cd "$PROJECT_DIR"
wasm-pack build \
    --target web \
    --no-default-features \
    --features wasm

# Step 3: Copy output to web/pkg/
mkdir -p web/pkg
cp pkg/* web/pkg/

echo ""
echo "✓ Built successfully!"
echo "  WASM:    $(ls -lh web/pkg/hyburn_bg.wasm | awk '{print $5}')"
echo "  Presets: $(grep -c 'pub const PRESET_' src/presets.rs) embedded"
echo ""
echo "To serve locally:"
echo "  cd web && python3 -m http.server 8080"
echo "  Then open http://localhost:8080"
