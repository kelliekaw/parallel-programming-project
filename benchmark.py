"""Measure total pipeline time and speedup (S = T1 / Tn) across worker counts.

Usage: python benchmark.py [--workers 1,2,4,6,8] [--runs 3]
Close other heavy programs first. Writes output/benchmark_results.csv.
"""
import argparse
import csv
import os
import statistics
import sys

from main import OUTPUT_DIR, check_tesseract, load_config, run_pipeline

BENCHMARK_FILE = os.path.join(OUTPUT_DIR, "benchmark_results.csv")


def main():
    sys.stdout.reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="1,2,4,6,8", help="comma-separated worker counts")
    parser.add_argument("--runs", type=int, default=3, help="runs per worker count")
    args = parser.parse_args()

    worker_counts = sorted({int(n) for n in args.workers.split(",")})
    if worker_counts[0] != 1:
        raise SystemExit("--workers must include 1, since speedup is relative to T1")

    check_tesseract()
    image_dir, _ = load_config()

    # The first pass reads images from a cold disk cache, which would inflate
    # whichever measurement came first. Run once and discard it.
    print("Warm-up run (discarded)...")
    run_pipeline(image_dir, worker_counts[-1], verbose=False)

    # Cycle through the counts on each pass so slow drift (e.g. thermal
    # throttling) affects them all equally instead of only the last ones.
    times = {n: [] for n in worker_counts}
    for run in range(1, args.runs + 1):
        for n in worker_counts:
            _, total_ms = run_pipeline(image_dir, n, verbose=False)
            times[n].append(total_ms)
            print(f"Run {run}/{args.runs}, {n} worker(s): {total_ms:.0f} ms")

    t1 = statistics.median(times[1])
    rows = []
    for n in worker_counts:
        tn = statistics.median(times[n])
        speedup = t1 / tn
        rows.append([n, f"{tn:.0f}", f"{speedup:.2f}", f"{speedup / n:.2f}"])

    header = ["Workers", "Median Time (ms)", "Speedup", "Efficiency"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(BENCHMARK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print()
    print("".join(f"{column:>18}" for column in header))
    for row in rows:
        print("".join(f"{value:>18}" for value in row))
    print(f"\nResults written to {BENCHMARK_FILE}")


if __name__ == "__main__":
    main()
