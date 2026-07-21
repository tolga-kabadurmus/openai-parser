from __future__ import annotations

from typing import Iterable, Iterator

from genai_normalizer.models import NormalizedEvent

from .sse_stream_parser import SSEStreamParser


def iter_normalized_events(
    chunks: Iterable[str | bytes],
    parser: SSEStreamParser,
) -> Iterator[NormalizedEvent]:
    """
    Feed chunks and yield newly produced normalized events incrementally.
    """
    emitted = 0

    for chunk in chunks:
        parser.feed(chunk)
        current = parser.events()

        while emitted < len(current):
            yield current[emitted]
            emitted += 1

    parser.finalize()
    current = parser.events()

    while emitted < len(current):
        yield current[emitted]
        emitted += 1
