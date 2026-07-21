from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from genai_normalizer.enums import Endpoint, Provider
from genai_normalizer.models import NormalizedMessage, NormalizedRequest
from genai_normalizer.utils import extract_text

from .coercion import to_mapping
from .detectors import (
    detect_endpoint,
    detect_modalities,
    detect_response_kind,
    detect_stream_mode,
)


class OpenAIRequestNormalizer:
    """Normalizes OpenAI request payloads without requiring the OpenAI SDK."""

    def normalize(
        self,
        payload: Any,
        *,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedRequest:
        data = to_mapping(payload)
        detected_endpoint = detect_endpoint(
            request_payload=data,
            explicit_endpoint=endpoint,
        )

        messages = self._normalize_messages(data.get("messages"))
        instructions = extract_text(data.get("instructions"))
        input_text = extract_text(data.get("input"))

        if not input_text and detected_endpoint is Endpoint.OPENAI_EMBEDDINGS:
            input_text = extract_text(data.get("input"))

        if not input_text and "prompt" in data:
            input_text = extract_text(data.get("prompt"))

        return NormalizedRequest(
            provider=Provider.OPENAI,
            endpoint=detected_endpoint,
            response_kind=detect_response_kind(detected_endpoint),
            stream_mode=detect_stream_mode(data),
            modalities=detect_modalities(
                request_payload=data,
                endpoint=detected_endpoint,
            ),
            model=_as_optional_str(data.get("model")),
            instructions=instructions,
            input_text=input_text,
            messages=messages,
            tools=_normalize_tools(data.get("tools")),
            tool_choice=data.get("tool_choice"),
            temperature=_as_optional_float(data.get("temperature")),
            top_p=_as_optional_float(data.get("top_p")),
            max_output_tokens=_first_int(
                data.get("max_output_tokens"),
                data.get("max_completion_tokens"),
                data.get("max_tokens"),
            ),
            user=_extract_user(data),
            metadata=dict(data.get("metadata") or {})
            if isinstance(data.get("metadata"), Mapping)
            else {},
            raw_request=payload,
        )

    def _normalize_messages(self, value: Any) -> List[NormalizedMessage]:
        if not isinstance(value, list):
            return []

        result: List[NormalizedMessage] = []

        for message in value:
            if isinstance(message, str):
                result.append(NormalizedMessage(role="user", text=message, raw=message))
                continue

            data = to_mapping(message)
            if not data:
                continue

            content = data.get("content")
            parts = content if isinstance(content, list) else []
            result.append(
                NormalizedMessage(
                    role=str(data.get("role") or "unknown"),
                    text=extract_text(content),
                    name=_as_optional_str(data.get("name")),
                    content_parts=[
                        dict(part) for part in parts if isinstance(part, Mapping)
                    ],
                    raw=message,
                )
            )

        return result


def _normalize_tools(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [to_mapping(item) for item in value if to_mapping(item)]


def _extract_user(data: Mapping[str, Any]) -> Optional[str]:
    if data.get("user") is not None:
        return str(data["user"])

    safety_identifier = data.get("safety_identifier")
    return str(safety_identifier) if safety_identifier is not None else None


def _as_optional_str(value: Any) -> Optional[str]:
    return str(value) if value is not None else None


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
