from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Any


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    elapsed_seconds: float
    items_per_second: float


def benchmark_callable(
    name: str,
    fn: Callable[[], Any],
    *,
    iterations: int = 10,
    warmup_iterations: int = 2,
) -> BenchmarkResult:
    for _ in range(warmup_iterations):
        fn()

    start = perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = perf_counter() - start

    items_per_second = iterations / elapsed if elapsed > 0 else 0.0
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        elapsed_seconds=elapsed,
        items_per_second=items_per_second,
    )


def format_benchmark_results(results: list[BenchmarkResult]) -> str:
    lines = ["Benchmark Results", "-----------------"]

    for result in results:
        lines.append(
            f"{result.name}: {result.items_per_second:.2f}/s "
            f"({result.elapsed_seconds:.4f}s for {result.iterations} runs)"
        )

    return "\n".join(lines)


