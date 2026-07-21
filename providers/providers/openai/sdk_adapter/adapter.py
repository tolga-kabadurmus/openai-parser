from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, TypeAdapter, ValidationError

from genai_normalizer.enums import Endpoint
from genai_normalizer.models import (
    NormalizedReasoning,
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
)
from genai_normalizer.providers.openai.request_normalizer import (
    OpenAIRequestNormalizer,
)
from genai_normalizer.providers.openai.response_normalizer import (
    OpenAIResponseNormalizer,
)

from .coercion import sdk_object_to_python
from .models import (
    ChatCompletionResponseModel,
    OpenAIRequestEnvelope,
    ResponsesAPIResponseModel,
)

T = TypeVar("T", bound=BaseModel)


class OpenAISDKAdapter:
    """
    Pydantic v2 validation and adaptation layer for OpenAI SDK objects.

    Validation is intentionally separate from normalization:
      SDK/raw object -> Pydantic v2 model -> existing normalized models
    """

    def __init__(self) -> None:
        self.request_normalizer = OpenAIRequestNormalizer()
        self.response_normalizer = OpenAIResponseNormalizer()

    def validate_request(self, value: Any) -> OpenAIRequestEnvelope:
        payload = sdk_object_to_python(value)
        return TypeAdapter(OpenAIRequestEnvelope).validate_python(payload)

    def validate_chat_completion(
        self,
        value: Any,
    ) -> ChatCompletionResponseModel:
        payload = sdk_object_to_python(value)
        return TypeAdapter(
            ChatCompletionResponseModel
        ).validate_python(payload)

    def validate_responses_api(
        self,
        value: Any,
    ) -> ResponsesAPIResponseModel:
        payload = sdk_object_to_python(value)
        return TypeAdapter(
            ResponsesAPIResponseModel
        ).validate_python(payload)

    def normalize_request(
        self,
        value: Any,
        *,
        endpoint: Optional[Endpoint | str] = None,
    ):
        validated = self.validate_request(value)
        return self.request_normalizer.normalize(
            validated.model_dump(mode="python"),
            endpoint=endpoint,
        )

    def normalize_chat_completion(
        self,
        value: Any,
        *,
        request_payload: Any = None,
    ) -> NormalizedResponse:
        validated = self.validate_chat_completion(value)
        return self.response_normalizer.normalize(
            validated.model_dump(mode="python"),
            request_payload=sdk_object_to_python(request_payload),
            endpoint=Endpoint.OPENAI_CHAT_COMPLETIONS,
        )

    def normalize_responses_api(
        self,
        value: Any,
        *,
        request_payload: Any = None,
    ) -> NormalizedResponse:
        validated = self.validate_responses_api(value)
        return self.response_normalizer.normalize(
            validated.model_dump(mode="python"),
            request_payload=sdk_object_to_python(request_payload),
            endpoint=Endpoint.OPENAI_RESPONSES,
        )

    def normalize_response(
        self,
        value: Any,
        *,
        request_payload: Any = None,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedResponse:
        payload = sdk_object_to_python(value)

        if endpoint in {
            Endpoint.OPENAI_CHAT_COMPLETIONS,
            "/v1/chat/completions",
            "chat.completions",
        }:
            return self.normalize_chat_completion(
                payload,
                request_payload=request_payload,
            )

        if endpoint in {
            Endpoint.OPENAI_RESPONSES,
            "/v1/responses",
            "responses",
        }:
            return self.normalize_responses_api(
                payload,
                request_payload=request_payload,
            )

        object_name = (
            str(payload.get("object", ""))
            if isinstance(payload, dict)
            else ""
        )

        if object_name == "chat.completion":
            return self.normalize_chat_completion(
                payload,
                request_payload=request_payload,
            )

        if object_name == "response":
            return self.normalize_responses_api(
                payload,
                request_payload=request_payload,
            )

        return self.response_normalizer.normalize(
            payload,
            request_payload=sdk_object_to_python(request_payload),
            endpoint=endpoint,
        )

    @staticmethod
    def validation_errors(exc: ValidationError) -> List[Dict[str, Any]]:
        return exc.errors(include_url=False)
