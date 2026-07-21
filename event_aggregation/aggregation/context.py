from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from genai_normalizer.enums import Endpoint, Provider, ResponseKind, StreamMode
from genai_normalizer.models import NormalizedRequest


@dataclass
class ResponseAggregationContext:
    provider: Provider = Provider.UNKNOWN
    endpoint: Endpoint = Endpoint.UNKNOWN
    response_kind: ResponseKind = ResponseKind.UNKNOWN
    stream_mode: StreamMode = StreamMode.NON_STREAMING

    request: Optional[NormalizedRequest] = None
    response_id: Optional[str] = None
    model: Optional[str] = None
    created_at: Optional[str] = None
    status: Optional[str] = None
    finish_reason: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None
