import json
import subprocess
import sys
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    generate_latest,
)

from app.schemas import QueryRequest, QueryResponse, SourceResponse
from observability.logging import get_logger
from observability.metrics import Metrics
from rag.pipeline import RAGPipeline


router = APIRouter()
logger = get_logger(__name__)


def create_pipeline(metrics: Metrics) -> RAGPipeline:
    """Create the local RAG pipeline."""

    from rag.chunker import chunk_document
    from rag.document_loader import load_documents
    from rag.embeddings import EmbeddingModel
    from rag.generator import LocalLLMGenerator
    from rag.retriever import SemanticRetriever

    documents = load_documents("data/documents")

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    embedding_model = EmbeddingModel()
    embedded_chunks = embedding_model.encode(chunks)

    retriever = SemanticRetriever(
        embedding_model=embedding_model,
        embedded_chunks=embedded_chunks,
    )

    generator = LocalLLMGenerator()

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        metrics=metrics,
    )


metrics = Metrics()
pipeline = create_pipeline(metrics=metrics)


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    """Answer a question using the RAG pipeline."""

    start_time = time.perf_counter()

    logger.info(
        "RAG query started | question=%r | top_k=%d",
        request.question,
        request.top_k,
    )

    response = pipeline.answer(
        question=request.question,
        top_k=request.top_k,
    )

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    logger.info(
        "RAG query completed | duration_ms=%.2f | sources=%d",
        duration_ms,
        len(response.sources),
    )

    sources = [
        SourceResponse(
            chunk_id=result.chunk.chunk_id,
            document_id=result.chunk.document_id,
            source=result.chunk.source,
            score=result.score,
            text=result.chunk.text,
        )
        for result in response.sources
    ]

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=sources,
    )


@router.post("/automation/evaluate")
def run_evaluation() -> dict[str, object]:
    """Run the complete evaluation workflow for automation clients."""

    logger.info("Automated evaluation started")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_evaluation",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error(
            "Automated evaluation failed | exit_code=%d",
            result.returncode,
        )
    else:
        logger.info(
            "Automated evaluation completed successfully"
        )

    evaluation_report = None
    report_path = Path("results/evaluation_report.json")

    try:
        evaluation_report = json.loads(
            report_path.read_text(
                encoding="utf-8",
            )
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error(
            "Could not read evaluation report | error=%s",
            exc,
        )

    regression = None

    if isinstance(evaluation_report, dict):
        regression = evaluation_report.get("regression")

    return {
        "status": (
            "success"
            if result.returncode == 0
            else "failed"
        ),
        "exit_code": result.returncode,
        "regression": regression,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "reports": [
            "results/evaluation_report.json",
            "results/quality_performance_report.json",
        ],
    }


@router.get("/metrics")
def get_metrics() -> dict[str, float | int]:
    """Return current application metrics."""

    return metrics.snapshot()


@router.get("/metrics/prometheus")
def prometheus_metrics() -> Response:
    """Expose metrics in Prometheus text format."""

    registry = CollectorRegistry()

    query_count = Gauge(
        "rag_query_count",
        "Number of completed RAG queries.",
        registry=registry,
    )

    average_query_duration = Gauge(
        "rag_average_query_duration_ms",
        "Average RAG query duration in milliseconds.",
        registry=registry,
    )

    average_retrieval_duration = Gauge(
        "rag_average_retrieval_duration_ms",
        "Average retrieval duration in milliseconds.",
        registry=registry,
    )

    average_generation_duration = Gauge(
        "rag_average_generation_duration_ms",
        "Average generation duration in milliseconds.",
        registry=registry,
    )

    snapshot = metrics.snapshot()

    query_count.set(
        snapshot["query_count"]
    )

    average_query_duration.set(
        snapshot["average_query_duration_ms"]
    )

    average_retrieval_duration.set(
        snapshot["average_retrieval_duration_ms"]
    )

    average_generation_duration.set(
        snapshot["average_generation_duration_ms"]
    )

    return Response(
        content=generate_latest(registry),
        media_type="text/plain",
    )
