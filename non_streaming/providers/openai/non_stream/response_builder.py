from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

from genai_normalizer.enums import Endpoint, Provider, StreamMode
from genai_normalizer.models import (
    NormalizedAudio,
    NormalizedEmbedding,
    NormalizedImage,
    NormalizedReasoning,
    NormalizedRequest,
    NormalizedResponse,
    NormalizedToolCall,
)
from genai_normalizer.providers.openai.detectors import (
    detect_endpoint,
    detect_modalities,
    detect_response_kind,
)
from genai_normalizer.providers.openai.usage import extract_usage

from .object_parser import OpenAIObjectParser


class OpenAINonStreamResponseBuilder:
    """Builds a unified response from an OpenAI non-streaming payload."""

    def __init__(
        self,
        object_parser: Optional[OpenAIObjectParser] = None,
    ) -> None:
        self.object_parser = object_parser or OpenAIObjectParser()

    def build(
        self,
        response_payload: Any,
        *,
        request_payload: Any = None,
        normalized_request: Optional[NormalizedRequest] = None,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedResponse:
        data = self.object_parser.to_dict(response_payload)
        request_data = self.object_parser.to_dict(request_payload)

        detected_endpoint = detect_endpoint(
            request_payload=request_data,
            response_payload=data,
            explicit_endpoint=endpoint,
        )

        response = NormalizedResponse(
            provider=Provider.OPENAI,
            endpoint=detected_endpoint,
            response_kind=detect_response_kind(detected_endpoint),
            stream_mode=StreamMode.NON_STREAMING,
            modalities=detect_modalities(
                request_payload=request_data,
                response_payload=data,
                endpoint=detected_endpoint,
            ),
            request=normalized_request,
            response_id=_optional_str(data.get("id")),
            model=_optional_str(data.get("model")),
            created_at=_created_at(data),
            status=_optional_str(data.get("status")),
            usage=extract_usage(data),
            error=_mapping_or_none(data.get("error"), self.object_parser),
            raw_response=response_payload,
        )

        if detected_endpoint is Endpoint.OPENAI_CHAT_COMPLETIONS:
            self._build_chat_completions(data, response)
        elif detected_endpoint is Endpoint.OPENAI_RESPONSES:
            self._build_responses(data, response)
        elif detected_endpoint is Endpoint.OPENAI_EMBEDDINGS:
            self._build_embeddings(data, response)
        elif detected_endpoint in {
            Endpoint.OPENAI_AUDIO_TRANSCRIPTIONS,
            Endpoint.OPENAI_AUDIO_TRANSLATIONS,
        }:
            self._build_audio_transcription(data, response)
        elif detected_endpoint is Endpoint.OPENAI_AUDIO_SPEECH:
            self._build_audio_speech(data, response)
        elif detected_endpoint is Endpoint.OPENAI_IMAGES_GENERATIONS:
            self._build_images(data, response)
        else:
            self._build_best_effort(data, response)

        return response

    def _build_chat_completions(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        visible_parts: List[str] = []
        reasoning_parts: List[str] = []

        for choice_value in data.get("choices") or []:
            choice = self.object_parser.to_dict(choice_value)
            message = self.object_parser.to_dict(choice.get("message"))

            visible_parts.extend(
                _extract_content_text(
                    message.get("content"),
                    self.object_parser,
                )
            )

            reasoning = _extract_reasoning(
                message.get(
                    "reasoning_content",
                    message.get("reasoning"),
                ),
                self.object_parser,
            )
            if reasoning:
                reasoning_parts.append(reasoning)

            response.tool_calls.extend(
                _extract_chat_tool_calls(
                    message.get("tool_calls"),
                    self.object_parser,
                )
            )

            function_call = self.object_parser.to_dict(
                message.get("function_call")
            )
            if function_call:
                arguments_text = str(
                    function_call.get("arguments") or ""
                )
                response.tool_calls.append(
                    NormalizedToolCall(
                        name=_optional_str(function_call.get("name")),
                        arguments_text=arguments_text,
                        arguments=_parse_json_object(arguments_text),
                        call_type="function",
                        raw=message.get("function_call"),
                    )
                )

            audio = self.object_parser.to_dict(message.get("audio"))
            if audio:
                response.audio.append(
                    NormalizedAudio(
                        transcript=str(audio.get("transcript") or ""),
                        audio_id=_optional_str(audio.get("id")),
                        format=_optional_str(audio.get("format")),
                        mime_type=_optional_str(audio.get("mime_type")),
                        data=_optional_str(audio.get("data")),
                        duration_seconds=_optional_float(
                            audio.get("duration_seconds")
                        ),
                        raw=message.get("audio"),
                    )
                )

            if (
                response.finish_reason is None
                and choice.get("finish_reason") is not None
            ):
                response.finish_reason = str(choice["finish_reason"])

        response.visible_text = "".join(visible_parts)
        response.reasoning = NormalizedReasoning(
            text="".join(reasoning_parts)
        )

        if response.status is None:
            response.status = "completed"

    def _build_responses(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        visible_parts: List[str] = []
        reasoning_parts: List[str] = []

        output_text = data.get("output_text")
        if isinstance(output_text, str):
            visible_parts.append(output_text)

        for output_value in data.get("output") or []:
            output = self.object_parser.to_dict(output_value)
            output_type = str(output.get("type") or "").lower()

            if output_type in {"message", "output_message"}:
                self._extract_response_message(
                    output,
                    visible_parts,
                    response,
                )

            elif output_type in {
                "function_call",
                "tool_call",
                "computer_call",
                "file_search_call",
                "web_search_call",
            }:
                response.tool_calls.append(
                    _response_tool_call(
                        output,
                        raw=output_value,
                    )
                )

            elif "reasoning" in output_type:
                reasoning_text = _extract_response_reasoning(
                    output,
                    self.object_parser,
                )
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)

        response.visible_text = _deduplicate_output_text(
            visible_parts,
            output_text,
        )
        response.reasoning = NormalizedReasoning(
            text="".join(reasoning_parts)
        )

        incomplete = self.object_parser.to_dict(
            data.get("incomplete_details")
        )
        if (
            response.finish_reason is None
            and incomplete.get("reason") is not None
        ):
            response.finish_reason = str(incomplete["reason"])

    def _extract_response_message(
        self,
        output: Dict[str, Any],
        visible_parts: List[str],
        response: NormalizedResponse,
    ) -> None:
        for content_value in output.get("content") or []:
            content = self.object_parser.to_dict(content_value)
            content_type = str(content.get("type") or "").lower()

            if content_type in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    visible_parts.append(text)

            elif content_type in {"refusal", "output_refusal"}:
                refusal = content.get("refusal", content.get("text"))
                if isinstance(refusal, str):
                    visible_parts.append(refusal)

            elif "audio" in content_type:
                response.audio.append(
                    NormalizedAudio(
                        transcript=str(
                            content.get("transcript")
                            or content.get("text")
                            or ""
                        ),
                        audio_id=_optional_str(content.get("id")),
                        format=_optional_str(content.get("format")),
                        mime_type=_optional_str(content.get("mime_type")),
                        data=_optional_str(
                            content.get("data")
                            or content.get("audio")
                        ),
                        raw=content_value,
                    )
                )

            elif "image" in content_type:
                response.images.append(
                    NormalizedImage(
                        image_id=_optional_str(content.get("id")),
                        url=_optional_str(content.get("url")),
                        base64_data=_optional_str(
                            content.get("b64_json")
                            or content.get("data")
                        ),
                        mime_type=_optional_str(content.get("mime_type")),
                        revised_prompt=_optional_str(
                            content.get("revised_prompt")
                        ),
                        raw=content_value,
                    )
                )

    def _build_embeddings(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        for item_value in data.get("data") or []:
            item = self.object_parser.to_dict(item_value)
            embedding = item.get("embedding")

            if isinstance(embedding, list):
                values = list(embedding)
            elif embedding is None:
                values = []
            else:
                values = [embedding]

            response.embeddings.append(
                NormalizedEmbedding(
                    index=_optional_int(item.get("index")),
                    values=values,
                    encoding_format=_optional_str(
                        data.get("encoding_format")
                    ),
                    raw=item_value,
                )
            )

        if response.status is None:
            response.status = "completed"

    def _build_audio_transcription(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        text = data.get("text")
        transcript = text if isinstance(text, str) else ""

        response.audio.append(
            NormalizedAudio(
                transcript=transcript,
                duration_seconds=_optional_float(data.get("duration")),
                raw=response.raw_response,
            )
        )
        response.visible_text = transcript

        if response.status is None:
            response.status = "completed"

    def _build_audio_speech(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        response.audio.append(
            NormalizedAudio(
                format=_optional_str(
                    data.get("format")
                    or data.get("response_format")
                ),
                mime_type=_optional_str(data.get("mime_type")),
                data=_optional_str(
                    data.get("data")
                    or data.get("audio")
                    or data.get("b64_json")
                ),
                raw=response.raw_response,
            )
        )

        if response.status is None:
            response.status = "completed"

    def _build_images(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        for item_value in data.get("data") or []:
            item = self.object_parser.to_dict(item_value)
            response.images.append(
                NormalizedImage(
                    image_id=_optional_str(item.get("id")),
                    url=_optional_str(item.get("url")),
                    base64_data=_optional_str(
                        item.get("b64_json")
                        or item.get("data")
                    ),
                    mime_type=_optional_str(item.get("mime_type")),
                    revised_prompt=_optional_str(
                        item.get("revised_prompt")
                    ),
                    raw=item_value,
                )
            )

        if response.status is None:
            response.status = "completed"

    def _build_best_effort(
        self,
        data: Dict[str, Any],
        response: NormalizedResponse,
    ) -> None:
        for key in ("text", "output_text", "content", "transcript"):
            value = data.get(key)
            if isinstance(value, str):
                response.visible_text = value
                break


def _extract_content_text(
    value: Any,
    object_parser: OpenAIObjectParser,
) -> List[str]:
    if isinstance(value, str):
        return [value]

    if not isinstance(value, list):
        return []

    result: List[str] = []
    for part_value in value:
        part = object_parser.to_dict(part_value)
        text = part.get("text")
        if isinstance(text, str):
            result.append(text)
            continue

        nested_text = object_parser.to_dict(text).get("value")
        if isinstance(nested_text, str):
            result.append(nested_text)

    return result


def _extract_chat_tool_calls(
    value: Any,
    object_parser: OpenAIObjectParser,
) -> List[NormalizedToolCall]:
    result: List[NormalizedToolCall] = []

    for item_value in value or []:
        item = object_parser.to_dict(item_value)
        function = object_parser.to_dict(item.get("function"))
        arguments_text = str(function.get("arguments") or "")

        result.append(
            NormalizedToolCall(
                id=_optional_str(item.get("id")),
                name=_optional_str(function.get("name")),
                arguments_text=arguments_text,
                arguments=_parse_json_object(arguments_text),
                call_type=str(item.get("type") or "function"),
                status=_optional_str(item.get("status")),
                raw=item_value,
            )
        )

    return result


def _response_tool_call(
    output: Dict[str, Any],
    *,
    raw: Any,
) -> NormalizedToolCall:
    arguments_value = output.get("arguments")

    if isinstance(arguments_value, Mapping):
        arguments = dict(arguments_value)
        arguments_text = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    else:
        arguments_text = str(arguments_value or "")
        arguments = _parse_json_object(arguments_text)

    return NormalizedToolCall(
        id=_optional_str(
            output.get("call_id")
            or output.get("id")
        ),
        name=_optional_str(
            output.get("name")
            or output.get("tool_name")
        ),
        arguments_text=arguments_text,
        arguments=arguments,
        call_type=str(output.get("type") or "function"),
        status=_optional_str(output.get("status")),
        raw=raw,
    )


def _extract_reasoning(
    value: Any,
    object_parser: OpenAIObjectParser,
) -> str:
    if isinstance(value, str):
        return value

    data = object_parser.to_dict(value)
    for key in ("text", "content", "context", "summary"):
        candidate = data.get(key)
        if isinstance(candidate, str):
            return candidate

    return ""


def _extract_response_reasoning(
    output: Dict[str, Any],
    object_parser: OpenAIObjectParser,
) -> str:
    parts: List[str] = []

    for summary_value in output.get("summary") or []:
        summary = object_parser.to_dict(summary_value)
        text = summary.get("text")
        if isinstance(text, str):
            parts.append(text)

    content = output.get("content")
    if isinstance(content, str):
        parts.append(content)

    return "".join(parts)


def _deduplicate_output_text(
    visible_parts: List[str],
    output_text: Any,
) -> str:
    combined = "".join(visible_parts)

    if isinstance(output_text, str):
        if not combined:
            return output_text
        if combined == output_text:
            return output_text

    return combined


def _parse_json_object(value: str) -> Optional[Dict[str, Any]]:
    if not value.strip():
        return None

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _mapping_or_none(
    value: Any,
    object_parser: OpenAIObjectParser,
) -> Optional[Dict[str, Any]]:
    data = object_parser.to_dict(value)
    return data or None


def _created_at(data: Mapping[str, Any]) -> Optional[str]:
    value = data.get("created_at", data.get("created"))
    return str(value) if value is not None else None


def _optional_str(value: Any) -> Optional[str]:
    return str(value) if value is not None else None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
