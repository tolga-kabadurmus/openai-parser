from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from genai_normalizer.enums import EventType
from genai_normalizer.models import (
    NormalizedAudio,
    NormalizedEmbedding,
    NormalizedEvent,
    NormalizedImage,
    NormalizedToolCall,
    NormalizedUsage,
)

from .state import AggregationState


class EventReducer(ABC):
    @abstractmethod
    def reduce(
        self,
        state: AggregationState,
        event: NormalizedEvent,
    ) -> None:
        raise NotImplementedError


class DefaultEventReducer(EventReducer):
    """Provider-neutral reducer for the canonical NormalizedEvent model."""

    def reduce(
        self,
        state: AggregationState,
        event: NormalizedEvent,
    ) -> None:
        event_key = self._event_key(event)
        if event_key in state.seen_event_keys:
            return

        state.seen_event_keys.add(event_key)
        state.events.append(event)

        if event.event_type is EventType.TEXT_DELTA:
            state.visible_parts.append(event.text)
            return

        if event.event_type is EventType.TEXT_DONE:
            if not state.visible_parts and event.text:
                state.visible_parts.append(event.text)
            return

        if event.event_type is EventType.REASONING_DELTA:
            state.reasoning_parts.append(event.text)
            return

        if event.event_type is EventType.REASONING_DONE:
            if not state.reasoning_parts and event.text:
                state.reasoning_parts.append(event.text)
            return

        if event.event_type in {
            EventType.TOOL_CALL_DELTA,
            EventType.TOOL_CALL_DONE,
        }:
            self._reduce_tool_call(state, event)
            return

        if event.event_type is EventType.AUDIO_DELTA:
            self._reduce_audio(state, event)
            return

        if event.event_type is EventType.AUDIO_TRANSCRIPT_DELTA:
            self._reduce_audio_transcript(state, event)
            return

        if event.event_type is EventType.IMAGE:
            self._reduce_image(state, event)
            return

        if event.event_type is EventType.EMBEDDING:
            self._reduce_embedding(state, event)
            return

        if event.event_type is EventType.USAGE:
            usage = event.metadata.get("usage")
            if isinstance(usage, dict):
                state.usage = NormalizedUsage.from_openai(usage)
            return

        if event.event_type is EventType.COMPLETED:
            state.status = str(event.metadata.get("status") or "completed")
            finish_reason = event.metadata.get("finish_reason")
            if finish_reason is not None:
                state.finish_reason = str(finish_reason)
            return

        if event.event_type is EventType.FAILED:
            state.status = "failed"
            error = event.metadata.get("error")
            if isinstance(error, dict):
                state.error = error
            return

        if event.event_type is EventType.DONE and state.status is None:
            state.status = "completed"

    def finalize_tool_calls(self, state: AggregationState) -> None:
        for key, value in state.tool_buffers.items():
            arguments_text = "".join(value["arguments"])
            state.tool_calls.append(
                NormalizedToolCall(
                    id=value.get("id"),
                    name=value.get("name"),
                    arguments_text=arguments_text,
                    arguments=_parse_json_object(arguments_text),
                    call_type=value.get("call_type") or "function",
                    status=value.get("status"),
                    raw=value.get("raw"),
                )
            )

    def _reduce_tool_call(
        self,
        state: AggregationState,
        event: NormalizedEvent,
    ) -> None:
        key = (
            event.tool_call_id
            or event.item_id
            or f"sequence:{event.sequence_number}"
        )

        buffer = state.tool_buffers.setdefault(
            key,
            {
                "id": event.tool_call_id,
                "name": None,
                "arguments": [],
                "call_type": "function",
                "status": None,
                "raw": [],
            },
        )

        if event.tool_name:
            buffer["name"] = event.tool_name

        if event.text:
            if event.event_type is EventType.TOOL_CALL_DONE:
                buffer["arguments"] = [event.text]
            else:
                buffer["arguments"].append(event.text)

        call_type = event.metadata.get("call_type")
        if call_type:
            buffer["call_type"] = str(call_type)

        status = event.metadata.get("status")
        if status:
            buffer["status"] = str(status)

        if event.raw is not None:
            buffer["raw"].append(event.raw)

    def _reduce_audio(
        self,
        state: AggregationState,
        event: NormalizedEvent,
    ) -> None:
        state.audio.append(
            NormalizedAudio(
                audio_id=event.item_id,
                data=event.text or None,
                format=_optional_str(event.metadata.get("format")),
                mime_type=_optional_str(event.metadata.get("mime_type")),
                raw=event.raw,
            )
        )

    def _reduce_audio_transcript(
        self,
        state: AggregationState,
        event: NormalizedEvent,
    ) -> None:
        state.audio.append(
            NormalizedAudio(
                audio_id=event.item_id,
                transcript=event.text,
                raw=event.raw,
            )
        )

    def _reduce_image(
        self,
        state: AggregationState,
        event: NormalizedEvent,
    ) -> None:
        state.images.append(
            NormalizedImage(
                image_id=event.item_id,
                url=_optional_str(event.metadata.get("url")),
                base64_data=_optional_str(event.metadata.get("base64_data")),
                mime_type=_optional_str(event.metadata.get("mime_type")),
                revised_prompt=_optional_str(
                    event.metadata.get("revised_prompt")
                ),
                raw=event.raw,
            )
        )

    def _reduce_embedding(
        self,
        state: AggregationState,
        event: NormalizedEvent,
    ) -> None:
        values = event.metadata.get("values")
        state.embeddings.append(
            NormalizedEmbedding(
                index=_optional_int(event.metadata.get("index")),
                values=list(values) if isinstance(values, list) else [],
                encoding_format=_optional_str(
                    event.metadata.get("encoding_format")
                ),
                raw=event.raw,
            )
        )

    def _event_key(self, event: NormalizedEvent) -> str:
        return "|".join(
            [
                event.event_type.value,
                str(event.sequence_number),
                str(event.item_id),
                str(event.tool_call_id),
                event.text,
            ]
        )


def _parse_json_object(value: str) -> Optional[Dict[str, Any]]:
    if not value.strip():
        return None

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _optional_str(value: Any) -> Optional[str]:
    return str(value) if value is not None else None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
