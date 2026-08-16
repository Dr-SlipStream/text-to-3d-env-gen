"""
Thin client for a locally-running Ollama server.

Why Ollama: it's free, runs entirely on your own GPU, needs no API key and
costs nothing per call -- which matters because this project has a zero budget.
The rest of the codebase only talks to the `LLMClient` interface, so swapping
in a different backend later means changing this one file.

Start the server with `ollama serve` (usually automatic after install), then
pull a model:  ollama pull qwen2.5:7b-instruct
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


class LLMUnavailable(RuntimeError):
    """Raised when the local model can't be reached or fails to respond."""


@dataclass
class LLMClient:
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    timeout: int = 120
    temperature: float = 0.2   # low: we want consistent structured output

    # -- health -----------------------------------------------------------
    def is_available(self) -> bool:
        """True if the Ollama server is up and our model is pulled."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            # Ollama reports "qwen2.5:7b-instruct"; accept a prefix match so
            # "qwen2.5:7b" also matches "qwen2.5:7b-instruct".
            return any(
                n == self.model or n.startswith(self.model.split(":")[0])
                for n in names
            )
        except Exception:
            return False

    def available_models(self) -> list:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return []

    # -- generation -------------------------------------------------------
    def generate_json(self, system: str, user: str) -> dict:
        """Ask the model for JSON and return it parsed.

        Uses Ollama's `format: json` mode, which constrains decoding to valid
        JSON -- far more reliable than asking politely in the prompt.
        """
        payload = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            r = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            r.raise_for_status()
            raw = r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError as e:
            raise LLMUnavailable(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running?"
            ) from e
        except Exception as e:
            raise LLMUnavailable(f"Ollama request failed: {e}") from e

        if not raw:
            raise LLMUnavailable("Ollama returned an empty response")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            # Last-ditch: pull the outermost {...} block out of the text.
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise LLMUnavailable(
                f"Model did not return valid JSON: {raw[:200]}"
            ) from e
