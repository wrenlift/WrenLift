#!/usr/bin/env python3
"""Time the same Wren scripts on Wren, WrenJIT and WrenLift.

Every kernel prints its own `elapsed: <seconds>` from `System.clock`, so the
time compared is the kernel's measured region, warm-up included and process
start-up excluded. Each engine runs each kernel `--runs` times; the table
reports the median. An engine's output, less its `elapsed:` lines, must match
Wren's: a run whose output differs is reported, not timed into the table.

Standard library only; runs on Linux, macOS and Windows.

    python3 bench/compare/compare.py \\
        --wren /path/to/wren_cli \\
        --wrenjit /path/to/WrenJIT \\
        --wrenlift /path/to/wlift \\
        --runs 7 --pin 4-7 --out results.md
"""

import argparse
import hashlib
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ENGINES = ("wren", "wrenjit", "wrenlift")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def kernels(selected):
    """Our kernels, their larger variants, and WrenJIT's own two scripts."""
    found = []
    for path in sorted(BENCH.glob("*.wren")):
        if not path.name.startswith("test_"):
            found.append((path.stem, path))
    for path in sorted((BENCH / "ash").glob("*.wren")):
        found.append((path.stem + " (large)", path))
    for path in sorted((HERE / "wrenjit").glob("*.wren")):
        found.append(("wrenjit/" + path.stem, path))
    if selected:
        found = [k for k in found if k[0] in selected]
    return found


def run_once(engine, script, timeout, pin):
    """(elapsed seconds or None, output without elapsed lines, error text)."""
    cmd = [engine, str(script)]
    if pin and shutil.which("taskset"):
        cmd = ["taskset", "-c", pin] + cmd
    try:
        proc = subprocess.run(
            cmd,
            cwd=script.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "", f"timeout after {timeout}s"
    out = proc.stdout
    elapsed = None
    kept = []
    for line in out.splitlines():
        if line.startswith("elapsed:"):
            try:
                elapsed = float(line.split()[1])
            except (IndexError, ValueError):
                pass
        else:
            kept.append(line)
    if proc.returncode != 0 or elapsed is None:
        err = (proc.stderr or out).strip().splitlines()
        reason = next((l for l in err if l.strip()), f"exit {proc.returncode}")
        reason = re.sub(r"\x1b\[[0-9;]*m", "", reason)
        reason = re.sub(r"panicked at \S+", "panicked", reason)
        return None, "\n".join(kept), reason[:120]
    return elapsed, "\n".join(kept), ""


def same_values(a, b):
    """Outputs that differ only in how many digits a number prints."""
    if NUMBER.sub("#", a) != NUMBER.sub("#", b):
        return False
    xs, ys = NUMBER.findall(a), NUMBER.findall(b)
    for x, y in zip(xs, ys):
        fx, fy = float(x), float(y)
        if fx != fy and abs(fx - fy) > 1e-12 * max(abs(fx), abs(fy)):
            return False
    return True


def measure(engines, script, runs, timeout, pin):
    result = {}
    reference = None
    for name in ENGINES:
        times = []
        status = "ok"
        for _ in range(runs):
            t, out, err = run_once(engines[name], script, timeout, pin)
            if t is None:
                status = "fail: " + err
                break
            if name == "wren":
                reference = out
            elif reference is not None and out != reference:
                status = "format" if same_values(out, reference) else "wrong output"
                if status == "wrong output":
                    break
            times.append(t)
        result[name] = {
            "median": statistics.median(times) if times and status in ("ok", "format") else None,
            "status": status,
            "digest": hashlib.sha256(reference.encode()).hexdigest()[:8] if reference else "",
        }
    return result


def fmt_time(seconds):
    if seconds is None:
        return "—"
    return f"{seconds * 1000:.0f} ms" if seconds >= 0.1 else f"{seconds * 1000:.1f} ms"


def ratio(a, b):
    if a is None or b is None or b == 0:
        return "—"
    return f"{a / b:.2f}x"


def environment(engines):
    lines = [
        f"- OS: {platform.platform()}",
        f"- CPU: {cpu_name()}",
        f"- Python: {platform.python_version()}",
    ]
    for name in ENGINES:
        lines.append(f"- {name}: `{engines[name]}`")
    return "\n".join(lines)


def cpu_name():
    try:
        if sys.platform == "linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True,
            ).stdout.strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ENGINES:
        p.add_argument(f"--{name}", required=True, help=f"path to the {name} executable")
    p.add_argument("--runs", type=int, default=7)
    p.add_argument("--timeout", type=int, default=300, help="seconds per run")
    p.add_argument("--pin", default="", help="CPU list for taskset on Linux, e.g. 4-7")
    p.add_argument("--only", nargs="*", help="kernel names, e.g. fib 'nbody (large)'")
    p.add_argument("--out", help="also write the Markdown report here")
    args = p.parse_args()
    engines = {name: getattr(args, name) for name in ENGINES}

    rows = []
    for name, script in kernels(args.only):
        r = measure(engines, script, args.runs, args.timeout, args.pin)
        rows.append((name, r))
        cells = "  ".join(f"{e}={fmt_time(r[e]['median'])} [{r[e]['status']}]" for e in ENGINES)
        print(f"{name:24} {cells}", file=sys.stderr, flush=True)

    report = [
        "### Absolute times",
        "",
        f"Median of {args.runs} runs of each kernel's measured region (`System.clock`). Lower is better.",
        "",
        "| Benchmark | Wren | WrenJIT | WrenLift |",
        "|---|---:|---:|---:|",
    ]
    notes = []
    for name, r in rows:
        cells = []
        for e in ENGINES:
            s = r[e]["status"]
            cell = fmt_time(r[e]["median"])
            if s == "format":
                cell += " †"
            elif s != "ok":
                cell = "fails ‡"
                notes.append(f"- `{name}` on {e}: {s}")
            cells.append(cell)
        report.append(f"| `{name}` | " + " | ".join(cells) + " |")
    report += [
        "",
        "† Same values as Wren, printed with a different number of digits.",
        "‡ The engine did not finish the kernel; reasons below.",
        "",
        "### Relative speed",
        "",
        "Higher is better: the other engine's time over WrenLift's.",
        "",
        "| Benchmark | WrenLift vs Wren | WrenLift vs WrenJIT |",
        "|---|---:|---:|",
    ]
    for name, r in rows:
        lift = r["wrenlift"]["median"]
        report.append(
            f"| `{name}` | {ratio(r['wren']['median'], lift)} | {ratio(r['wrenjit']['median'], lift)} |"
        )
    if notes:
        report += ["", "### Runs that did not finish", ""] + notes
    report += ["", "### Environment", "", environment(engines), ""]
    text = "\n".join(report)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
