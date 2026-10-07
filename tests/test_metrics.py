from observability.metrics import Metrics


def test_metrics_record_query() -> None:
    metrics = Metrics()

    metrics.record_query(100.0)
    metrics.record_query(200.0)

    snapshot = metrics.snapshot()

    assert snapshot["query_count"] == 2
    assert snapshot["total_query_duration_ms"] == 300.0


def test_metrics_record_retrieval_and_generation() -> None:
    metrics = Metrics()

    metrics.record_retrieval(10.0)
    metrics.record_generation(90.0)

    snapshot = metrics.snapshot()

    assert snapshot["total_retrieval_duration_ms"] == 10.0
    assert snapshot["total_generation_duration_ms"] == 90.0