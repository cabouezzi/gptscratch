#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


PERFORMANCE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PERFORMANCE_DIR.parent


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_DIR,
        check=True,
        text=True,
        capture_output=True,
    )


def benchmark() -> dict[str, tuple[float | None, float]]:
    build_dir = PERFORMANCE_DIR / "build" / "o1"
    setup_command = [
        "meson",
        "setup",
        str(build_dir),
        str(PROJECT_DIR),
        "-Doptimization=1",
        "-Ddebug=false",
        "-Db_ndebug=true",
    ]

    if build_dir.exists():
        setup_command.insert(2, "--reconfigure")

    run(setup_command)
    run(["meson", "compile", "-C", str(build_dir), "matmul_performance"])

    executable = build_dir / "performance" / "matmul_performance"
    result = run([str(executable)])

    timings: dict[str, tuple[float | None, float]] = {}
    for line in result.stdout.splitlines():
        name, cpu_milliseconds, gpu_milliseconds = line.rsplit("\t", 2)
        cpu = None if cpu_milliseconds == "-" else float(cpu_milliseconds)
        timings[name] = (cpu, float(gpu_milliseconds))
    return timings


def markdown_table(results: dict[str, tuple[float | None, float]]) -> str:

    lines = [
        "# Matmul performance",
        "",
        "Median milliseconds per call; includes input copies, output allocation, and free.",
        "",
        "| Test case | CPU (-O1) | GPU |",
        "|---|---:|---:|",
    ]

    for test_case, (cpu, gpu) in results.items():
        cpu_text = "—" if cpu is None else f"{cpu:.6f} ms"
        lines.append(f"| {test_case} | {cpu_text} | {gpu:.6f} ms |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CPU -O1 matmul performance against Metal matmul."
    )
    parser.parse_args()

    results = benchmark()
    table = markdown_table(results)
    results_path = PERFORMANCE_DIR / "results.md"
    results_path.write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
