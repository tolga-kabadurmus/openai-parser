from __future__ import annotations

from typing import Any, Optional

from genai_normalizer.enums import Endpoint, Provider
from genai_normalizer.models import (
    NormalizedRequest,
    NormalizedResponse,
)
from genai_normalizer.providers.openai.request_normalizer import (
    OpenAIRequestNormalizer,
)

from .object_parser import OpenAIObjectParser
from .response_builder import OpenAINonStreamResponseBuilder


class NonStreamParser:
    """
    OpenAI parser for non-streaming SDK calls.

    Accepts either raw dictionaries or OpenAI SDK model objects and returns
    the same `NormalizedResponse` structure used by the streaming parser.
    """

    def __init__(
        self,
        *,
        request_normalizer: Optional[OpenAIRequestNormalizer] = None,
        object_parser: Optional[OpenAIObjectParser] = None,
        response_builder: Optional[OpenAINonStreamResponseBuilder] = None,
    ) -> None:
        self.request_normalizer = (
            request_normalizer or OpenAIRequestNormalizer()
        )
        self.object_parser = object_parser or OpenAIObjectParser()
        self.response_builder = (
            response_builder
            or OpenAINonStreamResponseBuilder(
                object_parser=self.object_parser
            )
        )

        self._request: Optional[NormalizedRequest] = None
        self._response: Optional[NormalizedResponse] = None

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def request(self) -> Optional[NormalizedRequest]:
        return self._request

    @property
    def response(self) -> Optional[NormalizedResponse]:
        return self._response

    @property
    def usage(self):
        return self._response.usage if self._response else None

    def normalize_request(
        self,
        request_payload: Any,
        *,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedRequest:
        self._request = self.request_normalizer.normalize(
            self.object_parser.to_python(request_payload),
            endpoint=endpoint,
        )
        return self._request

    def normalize_response(
        self,
        response_payload: Any,
        *,
        request_payload: Any = None,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedResponse:
        normalized_request = None

        if request_payload is not None:
            normalized_request = self.normalize_request(
                request_payload,
                endpoint=endpoint,
            )

        self._response = self.response_builder.build(
            response_payload,
            request_payload=request_payload,
            normalized_request=normalized_request,
            endpoint=endpoint,
        )
        return self._response

    def parse(
        self,
        *,
        request_payload: Any,
        response_payload: Any,
        endpoint: Optional[Endpoint | str] = None,
    ) -> NormalizedResponse:
        return self.normalize_response(
            response_payload,
            request_payload=request_payload,
            endpoint=endpoint,
        )

    def normalized_request_text(self) -> str:
        return self._request.normalized_text() if self._request else ""

    def normalized_response_text(
        self,
        *,
        include_reasoning: bool = False,
        include_tools: bool = False,
    ) -> str:
        if self._response is None:
            return ""

        return self._response.normalized_text(
            include_reasoning=include_reasoning,
            include_tools=include_tools,
        )

    def reset(self) -> None:
        self._request = None
        self._response = None
