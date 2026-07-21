from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from genai_normalizer.enums import Endpoint, Modality, ResponseKind, StreamMode

from .coercion import to_mapping


def detect_endpoint(
    request_payload: Any = None,
    response_payload: Any = None,
    explicit_endpoint: Optional[Endpoint | str] = None,
) -> Endpoint:
    if explicit_endpoint is not None:
        endpoint = _coerce_endpoint(explicit_endpoint)
        if endpoint is not Endpoint.UNKNOWN:
            return endpoint

    request = to_mapping(request_payload)
    response = to_mapping(response_payload)

    endpoint_hint = request.get("endpoint") or request.get("api") or response.get("endpoint")
    if endpoint_hint:
        endpoint = _coerce_endpoint(str(endpoint_hint))
        if endpoint is not Endpoint.UNKNOWN:
            return endpoint

    object_name = str(response.get("object", "")).lower()
    event_type = str(response.get("type", "")).lower()

    if object_name in {"chat.completion", "chat.completion.chunk"} or "choices" in response:
        return Endpoint.OPENAI_CHAT_COMPLETIONS

    if (
        object_name == "response"
        or event_type.startswith("response.")
        or "output" in response
        or "instructions" in request
    ):
        return Endpoint.OPENAI_RESPONSES

    if object_name in {"list", "embedding"} and "data" in response:
        return Endpoint.OPENAI_EMBEDDINGS

    if _looks_like_embedding_request(request):
        return Endpoint.OPENAI_EMBEDDINGS

    if "voice" in request or "audio" in request:
        if request.get("input") is not None and request.get("voice") is not None:
            return Endpoint.OPENAI_AUDIO_SPEECH

    if "file" in request and ("model" in request or "language" in request):
        return Endpoint.OPENAI_AUDIO_TRANSCRIPTIONS

    if "prompt" in request and any(key in request for key in ("size", "quality", "background")):
        return Endpoint.OPENAI_IMAGES_GENERATIONS

    return Endpoint.UNKNOWN


def detect_stream_mode(request_payload: Any = None) -> StreamMode:
    request = to_mapping(request_payload)
    return (
        StreamMode.STREAMING
        if bool(request.get("stream"))
        else StreamMode.NON_STREAMING
    )


def detect_modalities(
    request_payload: Any = None,
    response_payload: Any = None,
    endpoint: Endpoint = Endpoint.UNKNOWN,
) -> list[Modality]:
    request = to_mapping(request_payload)
    response = to_mapping(response_payload)
    found: list[Modality] = []

    def add(modality: Modality) -> None:
        if modality not in found:
            found.append(modality)

    if endpoint in {Endpoint.OPENAI_CHAT_COMPLETIONS, Endpoint.OPENAI_RESPONSES}:
        add(Modality.TEXT)

    if endpoint is Endpoint.OPENAI_EMBEDDINGS:
        add(Modality.EMBEDDING)

    if endpoint in {
        Endpoint.OPENAI_AUDIO_SPEECH,
        Endpoint.OPENAI_AUDIO_TRANSCRIPTIONS,
        Endpoint.OPENAI_AUDIO_TRANSLATIONS,
    }:
        add(Modality.AUDIO)

    if endpoint is Endpoint.OPENAI_IMAGES_GENERATIONS:
        add(Modality.IMAGE)

    if endpoint is Endpoint.OPENAI_VIDEOS:
        add(Modality.VIDEO)

    response_type = str(response.get("type", "")).lower()
    if "reasoning" in response_type or "reasoning" in response:
        add(Modality.REASONING)
    if "function_call" in response_type or "tool_calls" in response:
        add(Modality.TOOL)

    for container in (request, response):
        if _contains_type(container, ("input_audio", "audio", "output_audio")):
            add(Modality.AUDIO)
        if _contains_type(container, ("input_image", "image_url", "image")):
            add(Modality.IMAGE)
        if _contains_type(container, ("function", "function_call", "tool_call")):
            add(Modality.TOOL)

    if len(found) > 1:
        add(Modality.MULTIMODAL)

    return found or [Modality.UNKNOWN]


def detect_response_kind(endpoint: Endpoint) -> ResponseKind:
    if endpoint in {Endpoint.OPENAI_CHAT_COMPLETIONS, Endpoint.OPENAI_RESPONSES}:
        return ResponseKind.LLM
    if endpoint in {
        Endpoint.OPENAI_AUDIO_TRANSCRIPTIONS,
        Endpoint.OPENAI_AUDIO_TRANSLATIONS,
    }:
        return ResponseKind.VOICE_STT
    if endpoint is Endpoint.OPENAI_AUDIO_SPEECH:
        return ResponseKind.VOICE_TTS
    if endpoint is Endpoint.OPENAI_EMBEDDINGS:
        return ResponseKind.EMBEDDING
    if endpoint is Endpoint.OPENAI_IMAGES_GENERATIONS:
        return ResponseKind.IMAGE
    if endpoint is Endpoint.OPENAI_VIDEOS:
        return ResponseKind.VIDEO
    return ResponseKind.UNKNOWN


def _coerce_endpoint(value: Endpoint | str) -> Endpoint:
    if isinstance(value, Endpoint):
        return value

    normalized = str(value).strip().lower().rstrip("/")
    aliases = {
        "/v1/chat/completions": Endpoint.OPENAI_CHAT_COMPLETIONS,
        "chat.completions": Endpoint.OPENAI_CHAT_COMPLETIONS,
        "chat_completions": Endpoint.OPENAI_CHAT_COMPLETIONS,
        "/v1/responses": Endpoint.OPENAI_RESPONSES,
        "responses": Endpoint.OPENAI_RESPONSES,
        "/v1/embeddings": Endpoint.OPENAI_EMBEDDINGS,
        "embeddings": Endpoint.OPENAI_EMBEDDINGS,
        "/v1/audio/transcriptions": Endpoint.OPENAI_AUDIO_TRANSCRIPTIONS,
        "/v1/audio/translations": Endpoint.OPENAI_AUDIO_TRANSLATIONS,
        "/v1/audio/speech": Endpoint.OPENAI_AUDIO_SPEECH,
        "/v1/images/generations": Endpoint.OPENAI_IMAGES_GENERATIONS,
        "/v1/videos": Endpoint.OPENAI_VIDEOS,
    }
    return aliases.get(normalized, Endpoint.UNKNOWN)


def _looks_like_embedding_request(request: Mapping[str, Any]) -> bool:
    encoding_format = request.get("encoding_format")
    dimensions = request.get("dimensions")
    return encoding_format is not None or dimensions is not None


def _contains_type(value: Any, candidate_types: Iterable[str]) -> bool:
    candidate_set = {item.lower() for item in candidate_types}

    if isinstance(value, Mapping):
        value_type = str(value.get("type", "")).lower()
        if value_type in candidate_set:
            return True
        return any(_contains_type(item, candidate_set) for item in value.values())

    if isinstance(value, (list, tuple)):
        return any(_contains_type(item, candidate_set) for item in value)

    return False
