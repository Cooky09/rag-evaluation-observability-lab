from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Metrics:
    """Simple in-process application metrics."""

    query_count: int = 0
    total_query_duration_ms: float = 0.0
    total_retrieval_duration_ms: float = 0.0
    total_generation_duration_ms: float = 0.0

    _lock: Lock = field(
        default_factory=Lock,
        repr=False,
    )

    def record_query(
        self,
        duration_ms: float,
    ) -> None:
        """Record a completed RAG query."""

        with self._lock:
            self.query_count += 1
            self.total_query_duration_ms += duration_ms

    def record_retrieval(
        self,
        duration_ms: float,
    ) -> None:
        """Record retrieval latency."""

        with self._lock:
            self.total_retrieval_duration_ms += duration_ms

    def record_generation(
        self,
        duration_ms: float,
    ) -> None:
        """Record generation latency."""

        with self._lock:
            self.total_generation_duration_ms += duration_ms

    def snapshot(self) -> dict[str, float | int]:
        """Return a point-in-time metrics snapshot."""

        with self._lock:
            query_count = self.query_count

            average_query = (
                self.total_query_duration_ms / query_count
                if query_count
                else 0.0
            )

            average_retrieval = (
                self.total_retrieval_duration_ms / query_count
                if query_count
                else 0.0
            )

            average_generation = (
                self.total_generation_duration_ms / query_count
                if query_count
                else 0.0
            )

            return {
                "query_count": query_count,
                "total_query_duration_ms": (
                    self.total_query_duration_ms
                ),
                "average_query_duration_ms": (
                    average_query
                ),
                "total_retrieval_duration_ms": (
                    self.total_retrieval_duration_ms
                ),
                "average_retrieval_duration_ms": (
                    average_retrieval
                ),
                "total_generation_duration_ms": (
                    self.total_generation_duration_ms
                ),
                "average_generation_duration_ms": (
                    average_generation
                ),
            }