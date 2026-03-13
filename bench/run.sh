#!/usr/bin/env bash
# Benchmark runner: compares WrenLift vs standard Wren
# Usage: ./bench/run.sh [benchmark_name]
#   Run all benchmarks:   ./bench/run.sh
#   Run one benchmark:    ./bench/run.sh fib

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=${BENCH_RUNS:-5}
WLIFT="./target/release/wlift"
WREN="${WREN_CLI:-wren_cli}"
TIMEOUT_SEC=${BENCH_TIMEOUT:-60}
WLIFT_STEP_LIMIT=${BENCH_STEP_LIMIT:-10000000000}
WLIFT_MODE=${BENCH_MODE:-tiered}

# Colors
BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

# Build release
printf "${DIM}Building release...${RESET}\n"
cargo build --release --quiet 2>/dev/null

# Check if standard wren is available
HAS_WREN=false
if command -v "$WREN" &>/dev/null; then
  HAS_WREN=true
  WREN_VERSION=$("$WREN" --version 2>&1 || echo "unknown")
else
  printf "${YELLOW}Warning: '$WREN' not found. Install wren-cli or set WREN_CLI env var.${RESET}\n"
  printf "${YELLOW}Showing WrenLift results only.${RESET}\n\n"
fi

BENCHMARKS=(fib method_call binary_trees delta_blue)

# Filter to specific benchmark if given
if [[ $# -gt 0 ]]; then
  BENCHMARKS=("$1")
fi

# Extract elapsed time from benchmark output (last line: "elapsed: X.XXX")
extract_time() {
  grep -oE 'elapsed: [0-9]+\.?[0-9]*' | tail -1 | awk '{print $2}'
}

# Run a single benchmark N times, return best time
# Handles crashes (exit code 139), compile errors (65), runtime errors (70), timeouts
run_bench() {
  local script="$1"
  shift
  local best=""

  for ((i = 1; i <= RUNS; i++)); do
    local output exit_code
    output=$(timeout "$TIMEOUT_SEC" "$@" "$script" 2>/dev/null) && exit_code=$? || exit_code=$?

    if [[ $exit_code -eq 139 ]]; then
      echo "CRASH"
      return
    elif [[ $exit_code -eq 65 ]]; then
      echo "COMPILE_ERR"
      return
    elif [[ $exit_code -eq 124 ]]; then
      echo "TIMEOUT"
      return
    elif [[ $exit_code -ne 0 ]]; then
      echo "ERROR:$exit_code"
      return
    fi

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

# Format a result cell
format_result() {
  local val="$1"
  case "$val" in
    CRASH)       printf "${RED}%12s${RESET}" "CRASH" ;;
    COMPILE_ERR) printf "${RED}%12s${RESET}" "COMPILE_ERR" ;;
    TIMEOUT)     printf "${RED}%12s${RESET}" "TIMEOUT" ;;
    ERROR:*)     printf "${RED}%12s${RESET}" "$val" ;;
    NO_OUTPUT)   printf "${RED}%12s${RESET}" "NO_OUTPUT" ;;
    *)           printf "%11.4fs" "$val" ;;
  esac
}

is_numeric() {
  [[ "$1" =~ ^[0-9]+\.?[0-9]*$ ]]
}

# Print results header
printf "\n${BOLD}%-20s %12s" "Benchmark" "WrenLift"
if $HAS_WREN; then
  printf " %12s %14s" "Wren 0.4" "Ratio"
fi
printf "${RESET}\n"
printf "%-20s %12s" "─────────────────" "──────────"
if $HAS_WREN; then
  printf " %12s %14s" "──────────" "────────────"
fi
printf "\n"

for bench in "${BENCHMARKS[@]}"; do
  script="bench/${bench}.wren"
  if [[ ! -f "$script" ]]; then
    printf "${RED}%-20s  not found${RESET}\n" "$bench"
    continue
  fi

  printf "${DIM}Running %-14s (%d runs each)...${RESET}\r" "$bench" "$RUNS"

  # WrenLift
  wlift_time=$(run_bench "$script" "$WLIFT" --mode "$WLIFT_MODE" --step-limit "$WLIFT_STEP_LIMIT")

  if $HAS_WREN; then
    # Standard Wren
    wren_time=$(run_bench "$script" "$WREN")

    # Calculate ratio
    printf "%-20s " "$bench"
    format_result "$wlift_time"
    printf " "
    format_result "$wren_time"

    if is_numeric "$wlift_time" && is_numeric "$wren_time"; then
      ratio=$(echo "scale=1; $wlift_time / $wren_time" | bc -l)
      if (( $(echo "$ratio <= 1" | bc -l) )); then
        printf " ${GREEN}%13sx${RESET}" "$ratio"
      elif (( $(echo "$ratio <= 2" | bc -l) )); then
        printf " ${YELLOW}%13sx${RESET}" "$ratio"
      else
        printf " ${RED}%13sx${RESET}" "$ratio"
      fi
    else
      printf " %14s" "—"
    fi
    printf "\n"
  else
    printf "%-20s " "$bench"
    format_result "$wlift_time"
    printf "\n"
  fi
done

printf "\n${DIM}Best of %d runs. Ratio = WrenLift / Wren (lower is better, <1.0 = faster).${RESET}\n" "$RUNS"

# Print known issues
printf "\n${BOLD}Known issues:${RESET}\n"
printf "  See bench/ISSUES.md for details and remediation plan.\n"
