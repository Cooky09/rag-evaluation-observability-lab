from fastapi import FastAPI

from app.routes import router
from observability.logging import configure_logging, get_logger


configure_logging()

logger = get_logger(__name__)

app = FastAPI(
    title="RAG Evaluation Observability Lab",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Return the API health status."""

    logger.info("Health check requested")

    return {
        "status": "ok",
        "service": "rag-evaluation-observability-lab",
    }