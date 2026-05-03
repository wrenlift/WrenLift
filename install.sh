#!/usr/bin/env bash
# shellcheck shell=bash

# WrenLift installer.
#
#   curl -fsSL https://raw.githubusercontent.com/wrenlift/WrenLift/main/install.sh | bash
#
# Fetches the `wlift` + `hatch` + `wlift-lsp` binaries from the
# latest GitHub Release (or the tag in WLIFT_VERSION), verifies
# the SHA256, and drops them in INSTALL_DIR (default
# `$HOME/.local/bin`).
#
# Environment knobs
#   WLIFT_VERSION  — pin a tag instead of picking latest. `v0.1.0`.
#   INSTALL_DIR    — where to drop the binaries. Default ~/.local/bin.
#                    Use /usr/local/bin for a system-wide install
#                    (needs sudo writable, the script won't elevate).
#   WLIFT_REPO     — override the GitHub slug. Default wrenlift/WrenLift.
#                    Useful for testing against a fork.
#
# Supported platforms
#   macOS arm64 / x86_64, Linux x86_64 / aarch64.
#   Windows users should grab the binaries manually from Releases.

set -euo pipefail

# -- Styled output ----------------------------------------------

if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi

say()  { printf "%s==>%s %s\n" "$BOLD" "$RESET" "$1"; }
warn() { printf "%s==>%s %s\n" "$YELLOW" "$RESET" "$1" >&2; }
die()  { printf "%serror:%s %s\n" "$RED" "$RESET" "$1" >&2; exit 1; }

# -- Config -----------------------------------------------------

REPO="${WLIFT_REPO:-wrenlift/WrenLift}"
VERSION="${WLIFT_VERSION:-}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

# -- Tool checks ------------------------------------------------

need() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not in PATH"
}

need curl
need tar
need uname
# Either shasum (macOS) or sha256sum (Linux) — we probe below.

# -- Platform detection -----------------------------------------

detect_platform() {
  local os arch
  os=$(uname -s | tr '[:upper:]' '[:lower:]')
  arch=$(uname -m)

  case "$os" in
    darwin) ;;
    linux)  ;;
    *)      die "unsupported OS '$os' — grab binaries manually from https://github.com/$REPO/releases" ;;
  esac

  case "$arch" in
    x86_64|amd64) arch="x86_64" ;;
    arm64|aarch64) arch="aarch64" ;;
    *) die "unsupported architecture '$arch' — grab binaries manually from https://github.com/$REPO/releases" ;;
  esac

  # Map to the rust target triple that matches the release artifact
  # filenames (`wlift-<tag>-<triple>.tar.gz`).
  case "$os-$arch" in
    darwin-aarch64) echo "aarch64-apple-darwin" ;;
    darwin-x86_64)  echo "x86_64-apple-darwin" ;;
    linux-x86_64)   echo "x86_64-unknown-linux-gnu" ;;
    linux-aarch64)  echo "aarch64-unknown-linux-gnu" ;;
    *) die "unsupported combo '$os-$arch'" ;;
  esac
}

# -- Version resolution -----------------------------------------

# Resolve the empty VERSION → latest runtime-release tag.
# Filters to tags matching `v<digit>...` so non-runtime tags
# co-located in the same repo (e.g. extension releases tagged
# `vscode-v0.1.0`, plugin tags `publish/...`) don't get
# picked. No `gh` / `jq` dependency — grep over the JSON.
resolve_latest() {
  local body
  body=$(curl -fsSL "https://api.github.com/repos/$REPO/releases" 2>/dev/null) \
    || die "couldn't reach https://api.github.com/repos/$REPO/releases"
  printf '%s' "$body" \
    | grep -oE '"tag_name":[[:space:]]*"v[0-9][^"]*"' \
    | head -1 \
    | sed -E 's/.*"(v[0-9][^"]*)".*/\1/'
}

# -- SHA256 verification ----------------------------------------

verify_sha256() {
  local file="$1" expected_line="$2"
  local actual expected
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$file" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 256 "$file" | awk '{print $1}')
  else
    die "neither sha256sum nor shasum found — can't verify the download"
  fi
  expected=$(awk '{print $1}' <<<"$expected_line")
  [ "$actual" = "$expected" ] \
    || die "checksum mismatch (got $actual, expected $expected)"
}

# -- Install ----------------------------------------------------

main() {
  local triple tag archive_name archive_url sum_url tmpdir

  triple=$(detect_platform)

  if [ -z "$VERSION" ]; then
    say "Resolving latest release…"
    tag=$(resolve_latest)
    [ -n "$tag" ] || die "couldn't parse latest tag"
  else
    tag="$VERSION"
  fi
  say "Target:   $BOLD$tag$RESET ($triple)"
  say "Dest:     $BOLD$INSTALL_DIR$RESET"

  archive_name="wlift-${tag}-${triple}.tar.gz"
  archive_url="https://github.com/$REPO/releases/download/${tag}/${archive_name}"
  sum_url="${archive_url}.sha256"

  tmpdir=$(mktemp -d)
  # Clean the tmpdir when the script exits, success or not. The
  # `${tmpdir:-}` expansion guards against `set -u`: by the time
  # the EXIT trap fires, `tmpdir` is a function-local that's
  # already gone out of scope, so a bare `$tmpdir` would error
  # ("unbound variable") and propagate non-zero from a successful
  # install — exactly the bug we hit on Fly's debian:bookworm-slim
  # builder where `set -u` is in force.
  trap 'rm -rf "${tmpdir:-}"' EXIT

  say "Downloading ${archive_name}…"
  curl -fsSL -o "$tmpdir/$archive_name" "$archive_url" \
    || die "download failed: $archive_url

  - Is the tag '$tag' a real release? See https://github.com/$REPO/releases
  - Is your network / proxy blocking github.com?"

  curl -fsSL -o "$tmpdir/$archive_name.sha256" "$sum_url" \
    || die "checksum download failed: $sum_url"

  say "Verifying SHA256…"
  verify_sha256 "$tmpdir/$archive_name" "$(cat "$tmpdir/$archive_name.sha256")"

  say "Extracting…"
  tar -xzf "$tmpdir/$archive_name" -C "$tmpdir"

  # Extracted layout: `wlift-<tag>-<triple>/{wlift,hatch,wlift-lsp,README.txt}`
  local staged="$tmpdir/wlift-${tag}-${triple}"
  [ -f "$staged/wlift" ] || die "archive didn't contain wlift — did the release format change?"
  [ -f "$staged/hatch" ] || die "archive didn't contain hatch"
  # `wlift-lsp` ships in releases v0.1.13+. Older tarballs pre-date
  # the LSP binary; treat absence as a soft warning so an explicit
  # downgrade to an older tag still installs the runtime + CLI.
  has_lsp=0
  if [ -f "$staged/wlift-lsp" ]; then
    has_lsp=1
  fi

  mkdir -p "$INSTALL_DIR"
  # Install with mv (fast, atomic-ish) then chmod so we're not
  # subject to umask weirdness on CI-cached artifacts.
  mv "$staged/wlift" "$INSTALL_DIR/wlift"
  mv "$staged/hatch" "$INSTALL_DIR/hatch"
  chmod +x "$INSTALL_DIR/wlift" "$INSTALL_DIR/hatch"
  if [ "$has_lsp" = "1" ]; then
    mv "$staged/wlift-lsp" "$INSTALL_DIR/wlift-lsp"
    chmod +x "$INSTALL_DIR/wlift-lsp"
  fi

  printf "\n"
  printf "%sInstalled %s%s to %s%s%s\n" "$GREEN" "$BOLD" "$tag" "$BOLD" "$INSTALL_DIR" "$RESET"

  # PATH hint — only print when INSTALL_DIR isn't already on PATH.
  # Uses a literal substring match so trailing slashes / duplicate
  # entries don't fool us.
  case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
      printf "\n"
      printf "%s%s is not on your PATH. Add this to your shell rc:%s\n" "$YELLOW" "$INSTALL_DIR" "$RESET"
      printf "  %sexport PATH=\"%s:\$PATH\"%s\n" "$DIM" "$INSTALL_DIR" "$RESET"
      ;;
  esac

  printf "\n"
  printf "%sVerify:%s\n" "$DIM" "$RESET"
  printf "  wlift --version\n"
  printf "  hatch --help\n"
  if [ "$has_lsp" = "1" ]; then
    printf "\n"
    printf "%swlift-lsp%s is the language server. Editors invoke it; you don't run it directly.\n" "$DIM" "$RESET"
    printf "  VS Code:  install %sWrenLift%s from the marketplace.\n" "$BOLD" "$RESET"
    printf "  Other:    %shttps://github.com/wrenlift/WrenLift/tree/main/editors%s\n" "$DIM" "$RESET"
  fi
}

main "$@"
