from __future__ import annotations

from typing import Any, Optional

from genai_normalizer.enums import Endpoint, Provider
from genai_normalizer.interfaces import ProviderParser
from genai_normalizer.models import NormalizedRequest, NormalizedResponse

from .request_normalizer import OpenAIRequestNormalizer
from .response_normalizer import OpenAIResponseNormalizer


class OpenAIParser(ProviderParser):
    """
    Provider façade for non-streaming OpenAI request/response normalization.

    Streaming parsing is supplied by Module 3.
    """

    def __init__(
        self,
        request_normalizer: Optional[OpenAIRequestNormalizer] = None,
        response_normalizer: Optional[OpenAIResponseNormalizer] = None,
    ) -> None:
        self.request_normalizer = request_normalizer or OpenAIRequestNormalizer()
        self.response_normalizer = response_normalizer or OpenAIResponseNormalizer()

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    def normalize_request(
        self,
        payload: Any,
        *,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedRequest:
        return self.request_normalizer.normalize(payload, endpoint=endpoint)

    def normalize_response(
        self,
        payload: Any,
        *,
        request_payload: Any = None,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedResponse:
        normalized_request = (
            self.normalize_request(request_payload, endpoint=endpoint)
            if request_payload is not None
            else None
        )

        response = self.response_normalizer.normalize(
            payload,
            request_payload=request_payload,
            endpoint=endpoint,
        )
        response.request = normalized_request
        return response
