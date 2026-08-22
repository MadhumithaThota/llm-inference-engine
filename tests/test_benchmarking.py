from engine.benchmarking import benchmark_callable, format_benchmark_results


def test_benchmark_callable_returns_result():
    result = benchmark_callable("noop", lambda: None, iterations=3, warmup_iterations=0)

    assert result.name == "noop"
    assert result.iterations == 3
    assert result.elapsed_seconds >= 0


def test_format_benchmark_results_contains_names():
    result = benchmark_callable("noop", lambda: None, iterations=1, warmup_iterations=0)
    report = format_benchmark_results([result])

    assert "noop" in report


