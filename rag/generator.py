import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inference.device import get_device_info


class LocalLLMGenerator:
    """Generate answers using a locally running Ollama model."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:1b",
        )
        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )
        ).rstrip("/")

    def runtime_info(self) -> dict[str, str | bool | int]:
        """Return information about the available local hardware."""
        return get_device_info()

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        prompt = self._build_prompt(
            question=question,
            context=context,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as exc:
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            ) from exc

        answer = result.get("response", "").strip()

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return answer

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
    ) -> str:
        return f"""Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:"""