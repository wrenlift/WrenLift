#!/usr/bin/env bash
# Build both the baseline and simd128 wasm-pack artifacts into
# `wasm/pkg/` so the playground always has a feature-detectable
# pair to serve. Mirrors the README recipe but in a single
# invocation — and cross-builds to a temporary pkg-simd/ dir so
# the simd binary ends up importing the same `./wlift_wasm_bg.js`
# shim as the baseline. That lets both wasm flavours share the
# single `wlift_wasm.js` loader createWlift imports.
#
# Usage (from `wasm/`):
#
#   ./build-pkg.sh
#
# Honours the env workarounds we need on macOS where Apple clang
# can't target wasm32 and the user shell sometimes ships a
# malformed CFLAGS.
set -euo pipefail

cd "$(dirname "$0")"

unset CFLAGS CPPFLAGS LDFLAGS
export CC_wasm32_unknown_unknown=/opt/homebrew/opt/llvm/bin/clang
export AR_wasm32_unknown_unknown=/opt/homebrew/opt/llvm/bin/llvm-ar
export CFLAGS_wasm32_unknown_unknown=""
export CPPFLAGS_wasm32_unknown_unknown=""

echo "==> baseline build (no SIMD)"
wasm-pack build . --target web --release --no-typescript

echo "==> simd128 build → pkg-simd/"
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build . --target web --release --no-typescript --out-dir pkg-simd

echo "==> moving simd binary into pkg/"
mv pkg-simd/wlift_wasm_bg.wasm pkg/wlift_wasm_bg.simd128.wasm
rm -rf pkg-simd

echo
echo "pkg/ contents:"
ls -la pkg/ | grep wlift_wasm
echo
echo "Both wasm files now import ./wlift_wasm_bg.js (the shared loader shim)."
