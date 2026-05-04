#!/usr/bin/env bash
# Benchmark runner: compares AOT-compiled WrenLift vs standard Wren.
#
# Each .wren benchmark prints `elapsed: X.XXX` from inside the
# program (driven by System.clock) — that's what gets timed, not the
# bash wall-clock. AOT compilation is one-shot per benchmark, run
# once before the measurement loop, and surfaced as a separate
# `compile:` column for transparency.
#
# Usage:
#   ./bench/run-aot.sh                 # all benchmarks
#   ./bench/run-aot.sh fib             # single benchmark
#
# Env vars:
#   BENCH_RUNS         best-of-N (default 5)
#   BENCH_TIMEOUT      per-run timeout in seconds (default 60)
#   WREN_CLI           standard Wren binary (default `wren_cli`)
#   CC                 linker driver for the AOT link step (default `cc`)
#   WLIFT_STATICLIB    explicit path to libwren_lift.a (default
#                      `target/release/libwren_lift.a`)

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=${BENCH_RUNS:-5}
WLIFT="./target/release/wlift"
WREN="${WREN_CLI:-wren_cli}"
TIMEOUT_SEC=${BENCH_TIMEOUT:-60}

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

# AOT requires the `aot` feature + the runtime staticlib. Build
# release with the feature on; that also produces libwren_lift.a
# at target/release/.
printf "${DIM}Building release (--features aot)...${RESET}\n"
cargo build --release --features aot --quiet 2>/dev/null

if [[ ! -f "$WLIFT" ]]; then
  printf "${RED}error: $WLIFT not found after build${RESET}\n"
  exit 1
fi

# Standard Wren is optional — if absent, we still print AOT numbers.
HAS_WREN=false
if command -v "$WREN" &>/dev/null; then
  HAS_WREN=true
  WREN_VERSION=$("$WREN" --version 2>&1 | head -1 || echo "unknown")
else
  printf "${YELLOW}Warning: '$WREN' not found. Install wren-cli or set WREN_CLI env var.${RESET}\n"
  printf "${YELLOW}Showing WrenLift AOT results only.${RESET}\n\n"
fi

BENCHMARKS=(fib method_call binary_trees delta_blue)
if [[ $# -gt 0 ]]; then
  BENCHMARKS=("$1")
fi

# Workspace for AOT-compiled binaries. Keeps them around so a
# second invocation of this script (e.g. quick re-runs after a
# code change) can cd here and run them by hand.
AOT_OUT="bench/.aot-out"
mkdir -p "$AOT_OUT"

# Strip ANSI from elapsed-line capture; standard Wren may colour
# diagnostics on stderr but the elapsed line goes to stdout.
extract_time() {
  grep -oE 'elapsed: [0-9]+\.?[0-9]*' | tail -1 | awk '{print $2}'
}

# AOT-compile a single benchmark, returning either the build time
# (in seconds, with sub-millisecond precision) or a sentinel
# describing the failure mode. Output: "<seconds> <binary_path>"
# on success, "ERROR:<reason>" otherwise.
aot_build_one() {
  local script="$1"
  local out="$AOT_OUT/$(basename "$script" .wren)"
  rm -f "$out" 2>/dev/null
  local start build_status
  start=$(python3 -c 'import time; print(time.monotonic())')
  if ! "$WLIFT" "$script" --aot "$out" >/dev/null 2>"$AOT_OUT/build.err"; then
    echo "ERROR:build_failed"
    return
  fi
  local end
  end=$(python3 -c 'import time; print(time.monotonic())')
  if [[ ! -x "$out" ]]; then
    echo "ERROR:no_output"
    return
  fi
  local dt
  dt=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
  echo "$dt $out"
}

# Run a single binary or a wlift-style command N times, return best
# `elapsed:` time. Same exit-code handling as the JIT bench: 139
# = SIGSEGV, 65 = compile error, 70 = runtime error, 124 = timeout.
run_bench() {
  local best=""
  for ((i = 1; i <= RUNS; i++)); do
    local output exit_code
    output=$(timeout "$TIMEOUT_SEC" "$@" 2>/dev/null) && exit_code=$? || exit_code=$?

    case $exit_code in
      139) echo "CRASH"; return ;;
      65)  echo "COMPILE_ERR"; return ;;
      70)  echo "RUNTIME_ERR"; return ;;
      124) echo "TIMEOUT"; return ;;
      0)   ;;
      *)   echo "ERROR:$exit_code"; return ;;
    esac

    local t
    t=$(echo "$output" | extract_time)
    if [[ -z "$t" ]]; then
      echo "NO_OUTPUT"
      return
    fi
    if [[ -z "$best" ]] || (( $(echo "$t < $best" | bc -l) )); then
      best="$t"
    fi
  done
  echo "$best"
}

format_result() {
  local val="$1"
  case "$val" in
    CRASH|COMPILE_ERR|RUNTIME_ERR|TIMEOUT|NO_OUTPUT|ERROR:*)
      printf "${RED}%12s${RESET}" "$val" ;;
    *) printf "%11.4fs" "$val" ;;
  esac
}

is_numeric() {
  [[ "$1" =~ ^[0-9]+\.?[0-9]*$ ]]
}

# Header
printf "\n${BOLD}%-16s %10s %12s" "Benchmark" "compile" "AOT-run"
if $HAS_WREN; then
  printf " %12s %12s" "Wren-cli" "ratio"
fi
printf "${RESET}\n"
printf "%-16s %10s %12s" "──────────" "────────" "──────────"
if $HAS_WREN; then
  printf " %12s %12s" "──────────" "──────────"
fi
printf "\n"
if $HAS_WREN; then
  printf "${DIM}wren-cli: %s${RESET}\n" "$WREN_VERSION"
fi

for bench in "${BENCHMARKS[@]}"; do
  script="bench/${bench}.wren"
  if [[ ! -f "$script" ]]; then
    printf "${RED}%-16s  not found${RESET}\n" "$bench"
    continue
  fi

  printf "${DIM}Running %-12s ...${RESET}\r" "$bench"

  # AOT compile (one-shot)
  build_result=$(aot_build_one "$script")
  if [[ "$build_result" == ERROR:* ]]; then
    printf "%-16s ${RED}%10s${RESET}\n" "$bench" "$build_result"
    continue
  fi
  build_time="${build_result%% *}"
  binary_path="${build_result##* }"

  # Best-of-N run of the AOT binary
  aot_time=$(run_bench "$binary_path")

  printf "%-16s %9.3fs " "$bench" "$build_time"
  format_result "$aot_time"

  if $HAS_WREN; then
    wren_time=$(run_bench "$WREN" "$script")
    printf " "
    format_result "$wren_time"
    if is_numeric "$aot_time" && is_numeric "$wren_time"; then
      ratio=$(echo "scale=2; $aot_time / $wren_time" | bc -l)
      if (( $(echo "$ratio <= 1" | bc -l) )); then
        printf " ${GREEN}%11sx${RESET}" "$ratio"
      elif (( $(echo "$ratio <= 2" | bc -l) )); then
        printf " ${YELLOW}%11sx${RESET}" "$ratio"
      else
        printf " ${RED}%11sx${RESET}" "$ratio"
      fi
    else
      printf " %12s" "—"
    fi
  fi
  printf "\n"
done

printf "\n${DIM}Best of %d runs. Ratio = AOT-WrenLift / standard-Wren (lower is faster).${RESET}\n" "$RUNS"
printf "${DIM}Compile time excluded from the run column — it's a one-shot cost per bench.${RESET}\n"
printf "${DIM}AOT binaries kept under %s for inspection / repeat runs.${RESET}\n" "$AOT_OUT"
