from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

from genai_normalizer.enums import Endpoint, Provider, StreamMode
from genai_normalizer.models import (
    NormalizedAudio,
    NormalizedEmbedding,
    NormalizedImage,
    NormalizedReasoning,
    NormalizedResponse,
    NormalizedToolCall,
)

from .coercion import to_mapping
from .detectors import detect_endpoint, detect_modalities, detect_response_kind
from .usage import extract_usage


class OpenAIResponseNormalizer:
    """Normalizes non-streaming OpenAI responses."""

    def normalize(
        self,
        payload: Any,
        *,
        request_payload: Any = None,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedResponse:
        data = to_mapping(payload)
        detected_endpoint = detect_endpoint(
            request_payload=request_payload,
            response_payload=data,
            explicit_endpoint=endpoint,
        )

        response = NormalizedResponse(
            provider=Provider.OPENAI,
            endpoint=detected_endpoint,
            response_kind=detect_response_kind(detected_endpoint),
            stream_mode=StreamMode.NON_STREAMING,
            modalities=detect_modalities(
                request_payload=request_payload,
                response_payload=data,
                endpoint=detected_endpoint,
            ),
            response_id=_optional_str(data.get("id")),
            model=_optional_str(data.get("model")),
            created_at=_created_at(data),
            status=_optional_str(data.get("status")),
            usage=extract_usage(data),
            error=_normalize_error(data.get("error")),
            raw_response=payload,
        )

        if detected_endpoint is Endpoint.OPENAI_CHAT_COMPLETIONS:
            self._populate_chat_completions(data, response)
        elif detected_endpoint is Endpoint.OPENAI_RESPONSES:
            self._populate_responses(data, response)
        elif detected_endpoint is Endpoint.OPENAI_EMBEDDINGS:
            self._populate_embeddings(data, response)
        elif detected_endpoint in {
            Endpoint.OPENAI_AUDIO_TRANSCRIPTIONS,
            Endpoint.OPENAI_AUDIO_TRANSLATIONS,
        }:
            self._populate_transcription(data, response)
        elif detected_endpoint is Endpoint.OPENAI_IMAGES_GENERATIONS:
            self._populate_images(data, response)
        else:
            self._populate_best_effort(data, response)

        return response

    def _populate_chat_completions(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        visible: List[str] = []
        reasoning: List[str] = []

        for choice in data.get("choices") or []:
            choice_data = to_mapping(choice)
            message = to_mapping(choice_data.get("message"))

            content = message.get("content")
            if isinstance(content, str):
                visible.append(content)
            elif isinstance(content, list):
                visible.extend(_text_from_content_parts(content))

            reasoning_value = message.get("reasoning_content", message.get("reasoning"))
            reasoning_text = _reasoning_text(reasoning_value)
            if reasoning_text:
                reasoning.append(reasoning_text)

            response.tool_calls.extend(_tool_calls(message.get("tool_calls")))

            if response.finish_reason is None and choice_data.get("finish_reason") is not None:
                response.finish_reason = str(choice_data["finish_reason"])

            audio = to_mapping(message.get("audio"))
            if audio:
                response.audio.append(
                    NormalizedAudio(
                        transcript=str(audio.get("transcript") or ""),
                        audio_id=_optional_str(audio.get("id")),
                        data=_optional_str(audio.get("data")),
                        raw=message.get("audio"),
                    )
                )

        response.visible_text = "".join(visible)
        response.reasoning = NormalizedReasoning(text="".join(reasoning))

    def _populate_responses(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        visible: List[str] = []
        reasoning: List[str] = []

        output_text = data.get("output_text")
        if isinstance(output_text, str):
            visible.append(output_text)

        for item in data.get("output") or []:
            item_data = to_mapping(item)
            item_type = str(item_data.get("type") or "")

            if item_type in {"message", "output_message"}:
                for content in item_data.get("content") or []:
                    content_data = to_mapping(content)
                    content_type = str(content_data.get("type") or "")

                    if content_type in {"output_text", "text"}:
                        text = content_data.get("text")
                        if isinstance(text, str):
                            visible.append(text)

                    if "refusal" in content_type:
                        refusal = content_data.get("refusal")
                        if isinstance(refusal, str):
                            visible.append(refusal)

            elif item_type in {"function_call", "tool_call"}:
                response.tool_calls.append(
                    NormalizedToolCall(
                        id=_optional_str(item_data.get("call_id") or item_data.get("id")),
                        name=_optional_str(item_data.get("name")),
                        arguments_text=str(item_data.get("arguments") or ""),
                        arguments=_parse_json_object(item_data.get("arguments")),
                        status=_optional_str(item_data.get("status")),
                        raw=item,
                    )
                )

            elif "reasoning" in item_type:
                summary_parts = []
                for summary in item_data.get("summary") or []:
                    summary_data = to_mapping(summary)
                    text = summary_data.get("text")
                    if isinstance(text, str):
                        summary_parts.append(text)
                if summary_parts:
                    reasoning.extend(summary_parts)

        response.visible_text = "".join(visible)
        response.reasoning = NormalizedReasoning(text="".join(reasoning))

        incomplete_details = to_mapping(data.get("incomplete_details"))
        if response.finish_reason is None and incomplete_details:
            response.finish_reason = _optional_str(incomplete_details.get("reason"))

    def _populate_embeddings(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        for item in data.get("data") or []:
            item_data = to_mapping(item)
            embedding = item_data.get("embedding")
            values = list(embedding) if isinstance(embedding, list) else [embedding] if embedding is not None else []
            response.embeddings.append(
                NormalizedEmbedding(
                    index=_as_int(item_data.get("index")),
                    values=values,
                    encoding_format=_optional_str(data.get("encoding_format")),
                    raw=item,
                )
            )

    def _populate_transcription(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        text = data.get("text")
        if isinstance(text, str):
            response.audio.append(
                NormalizedAudio(
                    transcript=text,
                    duration_seconds=_as_float(data.get("duration")),
                    raw=data,
                )
            )

    def _populate_images(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        for item in data.get("data") or []:
            item_data = to_mapping(item)
            response.images.append(
                NormalizedImage(
                    url=_optional_str(item_data.get("url")),
                    base64_data=_optional_str(item_data.get("b64_json")),
                    revised_prompt=_optional_str(item_data.get("revised_prompt")),
                    raw=item,
                )
            )

    def _populate_best_effort(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        text = data.get("text")
        if isinstance(text, str):
            response.visible_text = text


def _text_from_content_parts(parts: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for part in parts:
        data = to_mapping(part)
        text = data.get("text")
        if isinstance(text, str):
            result.append(text)
    return result


def _tool_calls(value: Any) -> List[NormalizedToolCall]:
    result: List[NormalizedToolCall] = []

    for item in value or []:
        data = to_mapping(item)
        function = to_mapping(data.get("function"))
        arguments_text = str(function.get("arguments") or "")
        result.append(
            NormalizedToolCall(
                id=_optional_str(data.get("id")),
                name=_optional_str(function.get("name")),
                arguments_text=arguments_text,
                arguments=_parse_json_object(arguments_text),
                call_type=str(data.get("type") or "function"),
                raw=item,
            )
        )

    return result


def _parse_json_object(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _reasoning_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    data = to_mapping(value)
    for key in ("text", "content", "context", "summary"):
        candidate = data.get(key)
        if isinstance(candidate, str):
            return candidate
    return ""


def _normalize_error(value: Any) -> Optional[Dict[str, Any]]:
    data = to_mapping(value)
    return data or None


def _created_at(data: Mapping[str, Any]) -> Optional[str]:
    value = data.get("created_at", data.get("created"))
    return str(value) if value is not None else None


def _optional_str(value: Any) -> Optional[str]:
    return str(value) if value is not None else None


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
