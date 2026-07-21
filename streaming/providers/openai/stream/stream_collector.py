from __future__ import annotations

from typing import Any, Iterable, Optional

from genai_normalizer.enums import Endpoint
from genai_normalizer.models import NormalizedResponse

from .sse_stream_parser import SSEStreamParser


class OpenAIStreamCollector:
    """Convenience wrapper for parsing an iterable of SSE chunks."""

    def collect(
        self,
        chunks: Iterable[str | bytes],
        *,
        request_payload: Any = None,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedResponse:
        parser = SSEStreamParser(
            request_payload=request_payload,
            endpoint=endpoint,
        )

        for chunk in chunks:
            parser.feed(chunk)

        return parser.finalize()
