from __future__ import annotations

from typing import Any, Dict, List, Mapping

from genai_normalizer.enums import EventType, Modality
from genai_normalizer.models import NormalizedEvent

from .coercion import to_mapping


class OpenAIEventNormalizer:
    """
    Converts raw OpenAI streaming events into provider-neutral events.

    Supports Chat Completions chunks and common Responses API events.
    Unknown event types are preserved as EventType.UNKNOWN.
    """

    def normalize(self, value: Any) -> List[NormalizedEvent]:
        data = to_mapping(value)
        if not data:
            return []

        if data.get("done") is True:
            return [
                NormalizedEvent(
                    event_type=EventType.DONE,
                    raw=value,
                )
            ]

        if "choices" in data:
            return self._normalize_chat_completion_chunk(data, value)

        event_type = str(data.get("type") or "")
        if event_type.startswith("response."):
            return self._normalize_responses_event(event_type, data, value)

        return [
            NormalizedEvent(
                event_type=EventType.UNKNOWN,
                raw=value,
                metadata={"openai_event_type": event_type} if event_type else {},
            )
        ]

    def _normalize_chat_completion_chunk(
        self,
        data: Dict[str, Any],
        raw: Any,
    ) -> List[NormalizedEvent]:
        result: List[NormalizedEvent] = []

        for choice in data.get("choices") or []:
            choice_data = to_mapping(choice)
            delta = to_mapping(choice_data.get("delta"))
            index = choice_data.get("index")

            content = delta.get("content")
            if isinstance(content, str) and content:
                result.append(
                    NormalizedEvent(
                        event_type=EventType.TEXT_DELTA,
                        modality=Modality.TEXT,
                        text=content,
                        sequence_number=_as_int(index),
                        raw=raw,
                    )
                )

            reasoning = delta.get("reasoning_content", delta.get("reasoning"))
            reasoning_text = _extract_reasoning_text(reasoning)
            if reasoning_text:
                result.append(
                    NormalizedEvent(
                        event_type=EventType.REASONING_DELTA,
                        modality=Modality.REASONING,
                        text=reasoning_text,
                        sequence_number=_as_int(index),
                        raw=raw,
                    )
                )

            for tool_call in delta.get("tool_calls") or []:
                tool = to_mapping(tool_call)
                function = to_mapping(tool.get("function"))
                arguments = function.get("arguments")
                result.append(
                    NormalizedEvent(
                        event_type=EventType.TOOL_CALL_DELTA,
                        modality=Modality.TOOL,
                        text=arguments if isinstance(arguments, str) else "",
                        tool_call_id=_as_optional_str(tool.get("id")),
                        tool_name=_as_optional_str(function.get("name")),
                        sequence_number=_as_int(tool.get("index", index)),
                        raw=raw,
                    )
                )

            finish_reason = choice_data.get("finish_reason")
            if finish_reason is not None:
                result.append(
                    NormalizedEvent(
                        event_type=EventType.COMPLETED,
                        modality=Modality.TEXT,
                        sequence_number=_as_int(index),
                        metadata={"finish_reason": finish_reason},
                        raw=raw,
                    )
                )

        if data.get("usage"):
            result.append(
                NormalizedEvent(
                    event_type=EventType.USAGE,
                    metadata={"usage": data["usage"]},
                    raw=raw,
                )
            )

        return result

    def _normalize_responses_event(
        self,
        event_name: str,
        data: Dict[str, Any],
        raw: Any,
    ) -> List[NormalizedEvent]:
        sequence_number = _as_int(data.get("sequence_number"))
        item_id = _as_optional_str(data.get("item_id"))

        if event_name == "response.output_text.delta":
            return [
                NormalizedEvent(
                    event_type=EventType.TEXT_DELTA,
                    modality=Modality.TEXT,
                    text=str(data.get("delta") or ""),
                    item_id=item_id,
                    sequence_number=sequence_number,
                    raw=raw,
                )
            ]

        if event_name == "response.output_text.done":
            return [
                NormalizedEvent(
                    event_type=EventType.TEXT_DONE,
                    modality=Modality.TEXT,
                    text=str(data.get("text") or ""),
                    item_id=item_id,
                    sequence_number=sequence_number,
                    raw=raw,
                )
            ]

        if "reasoning" in event_name and event_name.endswith(".delta"):
            return [
                NormalizedEvent(
                    event_type=EventType.REASONING_DELTA,
                    modality=Modality.REASONING,
                    text=str(data.get("delta") or ""),
                    item_id=item_id,
                    sequence_number=sequence_number,
                    raw=raw,
                    metadata={"openai_event_type": event_name},
                )
            ]

        if event_name == "response.function_call_arguments.delta":
            return [
                NormalizedEvent(
                    event_type=EventType.TOOL_CALL_DELTA,
                    modality=Modality.TOOL,
                    text=str(data.get("delta") or ""),
                    item_id=item_id,
                    sequence_number=sequence_number,
                    tool_call_id=_as_optional_str(data.get("call_id")),
                    raw=raw,
                )
            ]

        if event_name == "response.function_call_arguments.done":
            return [
                NormalizedEvent(
                    event_type=EventType.TOOL_CALL_DONE,
                    modality=Modality.TOOL,
                    text=str(data.get("arguments") or ""),
                    item_id=item_id,
                    sequence_number=sequence_number,
                    tool_call_id=_as_optional_str(data.get("call_id")),
                    raw=raw,
                )
            ]

        if event_name in {"response.completed", "response.incomplete"}:
            return [
                NormalizedEvent(
                    event_type=EventType.COMPLETED,
                    metadata={
                        "openai_event_type": event_name,
                        "response": data.get("response"),
                    },
                    sequence_number=sequence_number,
                    raw=raw,
                )
            ]

        if event_name in {"response.failed", "error"}:
            return [
                NormalizedEvent(
                    event_type=EventType.FAILED,
                    metadata={
                        "openai_event_type": event_name,
                        "error": data.get("error"),
                        "response": data.get("response"),
                    },
                    sequence_number=sequence_number,
                    raw=raw,
                )
            ]

        return [
            NormalizedEvent(
                event_type=EventType.UNKNOWN,
                item_id=item_id,
                sequence_number=sequence_number,
                metadata={"openai_event_type": event_name},
                raw=raw,
            )
        ]


def _extract_reasoning_text(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, Mapping):
        for key in ("text", "content", "context", "summary"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate

    return ""


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
