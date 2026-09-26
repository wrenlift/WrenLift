#!/usr/bin/env bash
# Build the prelinked runtime object a wasm AOT program links against.
#
# wlift_runtime.o is wren_lift built for wasm32-wasip1 with the portable
# `aot_runtime` feature (the AOT entry points and every runtime helper, no
# JIT), joined once with wasi-libc into one relocatable object. A program is
# then linked against exactly that object, so building one needs no wasi-sdk.
#
#   tools/build_wasm_runtime.sh                   # release, sysroot found
#   tools/build_wasm_runtime.sh --profile debug
#   WASI_SYSROOT=~/wasi-sysroot tools/build_wasm_runtime.sh
#
# The object lands at target/<profile>/wasm32-wasip1/wlift_runtime.o, beside
# the wlift binary of the same profile; tests/wasm_runtime_fresh.rs checks it.
#
# `--no-whole-archive` before libc keeps crt1 and the long-double printf out:
# whole-archived, they import `__main_argc_argv` and `__multc3`, which nothing
# provides.
set -euo pipefail

TRIPLE=wasm32-wasip1
PROFILE=release
while [ $# -gt 0 ]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --sysroot) WASI_SYSROOT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"

find_sysroot() {
    local c
    for c in "${WASI_SYSROOT:-}" \
        "$(brew --prefix wasi-libc 2>/dev/null || true)/share/wasi-sysroot" \
        /opt/homebrew/opt/wasi-libc/share/wasi-sysroot \
        /usr/local/opt/wasi-libc/share/wasi-sysroot \
        /opt/wasi-sdk/share/wasi-sysroot \
        /usr/local/wasi-sdk/share/wasi-sysroot \
        /usr/share/wasi-sysroot; do
        if [ -n "$c" ] && [ -f "$c/lib/$TRIPLE/libc.a" ]; then
            echo "$c"
            return
        fi
    done
    echo "no WASI sysroot with lib/$TRIPLE/libc.a: pass --sysroot, set WASI_SYSROOT," >&2
    echo "install wasi-libc (brew) or unpack a wasi-sdk release at /opt/wasi-sdk" >&2
    exit 1
}

# rust-lld ships with every toolchain, under the host's rustlib.
find_lld() {
    local host sysroot
    host="$(rustc -vV | sed -n 's/^host: //p')"
    sysroot="$(rustc --print sysroot)"
    if [ -x "$sysroot/lib/rustlib/$host/bin/rust-lld" ]; then
        echo "$sysroot/lib/rustlib/$host/bin/rust-lld"
    elif command -v rust-lld >/dev/null; then
        command -v rust-lld
    elif command -v wasm-ld >/dev/null; then
        command -v wasm-ld
    else
        echo "no rust-lld under $sysroot and none on PATH" >&2
        exit 1
    fi
}

SYSROOT="$(find_sysroot)"
LLD="$(find_lld)"
echo "sysroot: $SYSROOT"
echo "linker:  $LLD"

CARGO_FLAGS=(--lib --target "$TRIPLE" --no-default-features --features aot_runtime
    --crate-type staticlib)
if [ "$PROFILE" = release ]; then
    CARGO_FLAGS+=(--release)
fi
(cd "$REPO" && cargo rustc "${CARGO_FLAGS[@]}")

ARCHIVE="$REPO/target/$TRIPLE/$PROFILE/libwren_lift.a"
OUT="$REPO/target/$PROFILE/$TRIPLE/wlift_runtime.o"
mkdir -p "$(dirname "$OUT")"

LLD_FLAVOR=()
case "$(basename "$LLD")" in
    rust-lld*) LLD_FLAVOR=(-flavor wasm) ;;
esac
"$LLD" "${LLD_FLAVOR[@]}" -r -o "$OUT" \
    --whole-archive "$ARCHIVE" --no-whole-archive \
    -L"$SYSROOT/lib/$TRIPLE" -lc
echo "wrote $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
