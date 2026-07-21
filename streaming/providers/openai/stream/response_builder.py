from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from genai_normalizer.enums import (
    Endpoint,
    EventType,
    Provider,
    StreamMode,
)
from genai_normalizer.models import (
    NormalizedEvent,
    NormalizedReasoning,
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
)
from genai_normalizer.providers.openai.coercion import to_mapping
from genai_normalizer.providers.openai.detectors import (
    detect_endpoint,
    detect_modalities,
    detect_response_kind,
)
from genai_normalizer.providers.openai.usage import extract_usage


class OpenAIStreamResponseBuilder:
    """Aggregates raw and normalized OpenAI stream events."""

    def build(
        self,
        *,
        raw_events: Sequence[Dict[str, Any]],
        normalized_events: Sequence[NormalizedEvent],
        request_payload: Any = None,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedResponse:
        detected_endpoint = detect_endpoint(
            request_payload=request_payload,
            response_payload=raw_events[-1] if raw_events else None,
            explicit_endpoint=endpoint,
        )

        response = NormalizedResponse(
            provider=Provider.OPENAI,
            endpoint=detected_endpoint,
            response_kind=detect_response_kind(detected_endpoint),
            stream_mode=StreamMode.STREAMING,
            modalities=detect_modalities(
                request_payload=request_payload,
                response_payload=raw_events[-1] if raw_events else None,
                endpoint=detected_endpoint,
            ),
            events=list(normalized_events),
            raw_response=list(raw_events),
        )

        self._populate_identity(raw_events, response)
        self._populate_content(normalized_events, response)
        self._populate_status(raw_events, normalized_events, response)
        self._populate_usage(raw_events, response)
        self._populate_error(raw_events, response)

        return response

    def _populate_identity(
        self,
        raw_events: Sequence[Dict[str, Any]],
        response: NormalizedResponse,
    ) -> None:
        for event in raw_events:
            if response.response_id is None and event.get("id") is not None:
                response.response_id = str(event["id"])

            if response.model is None and event.get("model") is not None:
                response.model = str(event["model"])

            if response.created_at is None:
                created = event.get("created_at", event.get("created"))
                if created is not None:
                    response.created_at = str(created)

            nested = to_mapping(event.get("response"))
            if nested:
                if response.response_id is None and nested.get("id") is not None:
                    response.response_id = str(nested["id"])
                if response.model is None and nested.get("model") is not None:
                    response.model = str(nested["model"])
                if response.created_at is None:
                    created = nested.get("created_at", nested.get("created"))
                    if created is not None:
                        response.created_at = str(created)

    def _populate_content(
        self,
        normalized_events: Sequence[NormalizedEvent],
        response: NormalizedResponse,
    ) -> None:
        visible_parts: List[str] = []
        reasoning_parts: List[str] = []

        tool_buffers: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"name": None, "arguments": [], "raw": []}
        )

        for event in normalized_events:
            if event.event_type is EventType.TEXT_DELTA:
                visible_parts.append(event.text)

            elif event.event_type is EventType.REASONING_DELTA:
                reasoning_parts.append(event.text)

            elif event.event_type in {
                EventType.TOOL_CALL_DELTA,
                EventType.TOOL_CALL_DONE,
            }:
                key = (
                    event.tool_call_id
                    or event.item_id
                    or f"index:{event.sequence_number}"
                )
                buffer = tool_buffers[key]

                if event.tool_name:
                    buffer["name"] = event.tool_name

                if event.text:
                    if event.event_type is EventType.TOOL_CALL_DONE:
                        buffer["arguments"] = [event.text]
                    else:
                        buffer["arguments"].append(event.text)

                if event.raw is not None:
                    buffer["raw"].append(event.raw)

        response.visible_text = "".join(visible_parts)
        response.reasoning = NormalizedReasoning(
            text="".join(reasoning_parts)
        )

        for tool_call_id, value in tool_buffers.items():
            arguments_text = "".join(value["arguments"])
            response.tool_calls.append(
                NormalizedToolCall(
                    id=tool_call_id if not tool_call_id.startswith("index:") else None,
                    name=value["name"],
                    arguments_text=arguments_text,
                    arguments=_parse_json_object(arguments_text),
                    raw=value["raw"],
                )
            )

    def _populate_status(
        self,
        raw_events: Sequence[Dict[str, Any]],
        normalized_events: Sequence[NormalizedEvent],
        response: NormalizedResponse,
    ) -> None:
        for event in normalized_events:
            if event.event_type is EventType.FAILED:
                response.status = "failed"
            elif event.event_type is EventType.COMPLETED and response.status is None:
                response.status = "completed"

            finish_reason = event.metadata.get("finish_reason")
            if finish_reason is not None:
                response.finish_reason = str(finish_reason)

        for raw_event in reversed(raw_events):
            nested = to_mapping(raw_event.get("response"))
            if nested:
                status = nested.get("status")
                if status is not None:
                    response.status = str(status)

                incomplete = to_mapping(nested.get("incomplete_details"))
                if response.finish_reason is None and incomplete.get("reason") is not None:
                    response.finish_reason = str(incomplete["reason"])

            if response.status is None and raw_event.get("done") is True:
                response.status = "completed"

    def _populate_usage(
        self,
        raw_events: Sequence[Dict[str, Any]],
        response: NormalizedResponse,
    ) -> None:
        for raw_event in reversed(raw_events):
            usage = extract_usage(raw_event)
            if usage.raw:
                response.usage = usage
                return

        response.usage = NormalizedUsage()

    def _populate_error(
        self,
        raw_events: Sequence[Dict[str, Any]],
        response: NormalizedResponse,
    ) -> None:
        for raw_event in reversed(raw_events):
            error = raw_event.get("error")
            if isinstance(error, Mapping):
                response.error = dict(error)
                return

            nested = to_mapping(raw_event.get("response"))
            nested_error = nested.get("error")
            if isinstance(nested_error, Mapping):
                response.error = dict(nested_error)
                return


def _parse_json_object(value: str) -> Optional[Dict[str, Any]]:
    if not value.strip():
        return None

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None
