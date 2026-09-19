#!/usr/bin/env bash
# Stage the release binaries for one target and make the tarball.
#
# The binary links LLVM statically. On macOS a few Homebrew dylibs it
# still references (z3, and whatever they pull in) live at paths only
# a machine with Homebrew has, so they are copied beside the binary
# with their load commands rewritten to @executable_path, and every
# Mach-O is ad-hoc re-signed: rewriting load commands invalidates a
# signature, and stripping does too. On Linux the equivalents are
# distro packages present on any desktop install and stay dynamic.
#
# Usage: tools/package_release.sh <tag> <triple> <bin_dir>
set -euo pipefail

TAG="${1:?usage: package_release.sh <tag> <triple> <bin_dir>}"
TRIPLE="${2:?usage: package_release.sh <tag> <triple> <bin_dir>}"
BIN_DIR="${3:?usage: package_release.sh <tag> <triple> <bin_dir>}"

STAGE="wlift-${TAG}-${TRIPLE}"
DIST="dist/${STAGE}"
rm -rf "$DIST"
mkdir -p "$DIST"
for bin in wlift hatch wlift-lsp; do
  test -x "${BIN_DIR}/${bin}" || { echo "error: ${BIN_DIR}/${bin} not built" >&2; exit 1; }
  cp "${BIN_DIR}/${bin}" "$DIST/"
done
cat > "$DIST/README.txt" <<TXT
wlift ${TAG}     — Wren runtime (JIT + interpreter)
hatch ${TAG}     — Package + build tool for Wren
wlift-lsp ${TAG} — Language server (LSP) for editors

Add this dir to your PATH to use the binaries directly.
Docs: https://github.com/wrenlift/WrenLift
TXT

if [[ "$(uname -s)" == "Darwin" ]]; then
  for bin in wlift hatch wlift-lsp; do
    macho="$DIST/$bin"
    strip -x "$macho"
    otool -L "$macho" | awk 'NR>1 {print $1}' | while read -r dep; do
      case "$dep" in
        /usr/lib/*|/System/*|@*) continue ;;
      esac
      name="$(basename "$dep")"
      [[ -f "$DIST/$name" ]] || { cp "$dep" "$DIST/$name"; chmod u+w "$DIST/$name"; }
      install_name_tool -change "$dep" "@executable_path/$name" "$macho"
      install_name_tool -id "@executable_path/$name" "$DIST/$name"
      # A bundled dylib can itself reference other Homebrew dylibs.
      otool -L "$DIST/$name" | awk 'NR>1 {print $1}' | while read -r sub; do
        case "$sub" in
          /usr/lib/*|/System/*|@*) continue ;;
        esac
        subname="$(basename "$sub")"
        [[ -f "$DIST/$subname" ]] || { cp "$sub" "$DIST/$subname"; chmod u+w "$DIST/$subname"; }
        install_name_tool -change "$sub" "@executable_path/$subname" "$DIST/$name"
      done
      codesign --force -s - "$DIST/$name"
    done
    codesign --force -s - "$macho"
  done
  echo "bundled:"
  ls -la "$DIST"
  otool -L "$DIST/wlift" | sed -n '2,20p'
else
  for bin in wlift hatch wlift-lsp; do
    strip "$DIST/$bin"
  done
  echo "dynamic dependencies (expected to come from distro packages):"
  ldd "$DIST/wlift" | grep -v "linux-vdso\|ld-linux\|libc\.\|libm\.\|libgcc\|libpthread\|libdl" || true
fi

tar -C dist -czf "dist/${STAGE}.tar.gz" "${STAGE}"
(cd dist && shasum -a 256 "${STAGE}.tar.gz" > "${STAGE}.tar.gz.sha256")
ls -la dist/
