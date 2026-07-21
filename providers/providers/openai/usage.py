from __future__ import annotations

from typing import Any, Mapping, Optional

from genai_normalizer.models import NormalizedUsage

from .coercion import to_mapping


def extract_usage(payload: Any) -> NormalizedUsage:
    data = to_mapping(payload)

    usage = data.get("usage")
    if isinstance(usage, Mapping):
        return NormalizedUsage.from_openai(dict(usage))

    response = data.get("response")
    response_data = to_mapping(response)
    nested_usage = response_data.get("usage")
    if isinstance(nested_usage, Mapping):
        return NormalizedUsage.from_openai(dict(nested_usage))

    return NormalizedUsage()
