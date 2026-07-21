from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from genai_normalizer.models import (
    NormalizedAudio,
    NormalizedEmbedding,
    NormalizedEvent,
    NormalizedImage,
    NormalizedToolCall,
    NormalizedUsage,
)


@dataclass
class AggregationState:
    visible_parts: List[str] = field(default_factory=list)
    reasoning_parts: List[str] = field(default_factory=list)
    tool_buffers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    audio: List[NormalizedAudio] = field(default_factory=list)
    images: List[NormalizedImage] = field(default_factory=list)
    embeddings: List[NormalizedEmbedding] = field(default_factory=list)
    tool_calls: List[NormalizedToolCall] = field(default_factory=list)

    usage: NormalizedUsage = field(default_factory=NormalizedUsage)
    error: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    finish_reason: Optional[str] = None

    events: List[NormalizedEvent] = field(default_factory=list)
    seen_event_keys: Set[str] = field(default_factory=set)
