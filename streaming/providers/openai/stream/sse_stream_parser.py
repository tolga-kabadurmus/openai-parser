from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from genai_normalizer.enums import Endpoint
from genai_normalizer.models import NormalizedEvent, NormalizedResponse
from genai_normalizer.providers.openai.detectors import detect_endpoint
from genai_normalizer.providers.openai.event_normalizer import OpenAIEventNormalizer

from .response_builder import OpenAIStreamResponseBuilder


class SSEStreamParser:
    """
    OpenAI SSE parser for streaming calls.

    Supports:
      - /v1/chat/completions
      - /v1/responses

    The parser separates SSE framing from OpenAI event interpretation.
    """

    def __init__(
        self,
        *,
        request_payload: Any = None,
        endpoint: Optional[Endpoint | str] = None,
        event_normalizer: Optional[OpenAIEventNormalizer] = None,
        response_builder: Optional[OpenAIStreamResponseBuilder] = None,
    ) -> None:
        self.request_payload = request_payload
        self.explicit_endpoint = endpoint
        self.event_normalizer = event_normalizer or OpenAIEventNormalizer()
        self.response_builder = response_builder or OpenAIStreamResponseBuilder()

        self.buffer: str = ""
        self.raw_events: List[Dict[str, Any]] = []
        self.normalized_events: List[NormalizedEvent] = []
        self._finalized = False

    def feed(self, raw_chunk: str | bytes) -> None:
        if self._finalized:
            raise RuntimeError("Cannot feed data after finalize().")

        if isinstance(raw_chunk, bytes):
            raw_chunk = raw_chunk.decode("utf-8", errors="replace")

        if not isinstance(raw_chunk, str):
            raise TypeError(f"raw_chunk must be str or bytes, got {type(raw_chunk)!r}")

        self.buffer += raw_chunk
        self._drain_complete_events()

    def events(self) -> Sequence[NormalizedEvent]:
        return tuple(self.normalized_events)

    def endpoint(self) -> Endpoint:
        for raw_event in self.raw_events:
            detected = detect_endpoint(
                request_payload=self.request_payload,
                response_payload=raw_event,
                explicit_endpoint=self.explicit_endpoint,
            )
            if detected is not Endpoint.UNKNOWN:
                return detected

        return detect_endpoint(
            request_payload=self.request_payload,
            explicit_endpoint=self.explicit_endpoint,
        )

    def cost_usage(self) -> Optional[Dict[str, Any]]:
        for event in reversed(self.raw_events):
            usage = event.get("usage")
            if isinstance(usage, dict) and usage:
                return usage

            response = event.get("response")
            if isinstance(response, dict):
                nested_usage = response.get("usage")
                if isinstance(nested_usage, dict) and nested_usage:
                    return nested_usage

        return None

    def total_text(self) -> str:
        response = self.response_builder.build(
            raw_events=self.raw_events,
            normalized_events=self.normalized_events,
            request_payload=self.request_payload,
            endpoint=self.endpoint(),
        )
        return response.visible_text

    def finalize(self) -> NormalizedResponse:
        if not self._finalized:
            if self.buffer.strip():
                self.buffer += "\n\n"
                self._drain_complete_events()
            self._finalized = True

        return self.response_builder.build(
            raw_events=self.raw_events,
            normalized_events=self.normalized_events,
            request_payload=self.request_payload,
            endpoint=self.endpoint(),
        )

    def reset(self) -> None:
        self.buffer = ""
        self.raw_events.clear()
        self.normalized_events.clear()
        self._finalized = False

    def _drain_complete_events(self) -> None:
        normalized_buffer = self.buffer.replace("\r\n", "\n")
        parts = normalized_buffer.split("\n\n")

        self.buffer = parts[-1]
        complete_parts = parts[:-1]

        for part in complete_parts:
            self._parse_sse_event(part)

    def _parse_sse_event(self, event_text: str) -> None:
        data_lines: List[str] = []

        for line in event_text.splitlines():
            if not line or line.startswith(":"):
                continue

            if line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)

        if not data_lines:
            return

        payload_text = "\n".join(data_lines).strip()
        if not payload_text:
            return

        if payload_text == "[DONE]":
            raw_event: Dict[str, Any] = {"done": True}
            self.raw_events.append(raw_event)
            self.normalized_events.extend(
                self.event_normalizer.normalize(raw_event)
            )
            return

        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            return

        if not isinstance(parsed, dict):
            parsed = {"value": parsed}

        self.raw_events.append(parsed)
        self.normalized_events.extend(
            self.event_normalizer.normalize(parsed)
        )
