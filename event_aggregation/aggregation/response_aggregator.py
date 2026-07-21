from __future__ import annotations

from typing import Iterable, Optional

from genai_normalizer.enums import Modality
from genai_normalizer.models import (
    NormalizedEvent,
    NormalizedReasoning,
    NormalizedResponse,
)

from .context import ResponseAggregationContext
from .reducers import DefaultEventReducer, EventReducer
from .state import AggregationState


class ResponseAggregator:
    """Aggregates provider-neutral events into one NormalizedResponse."""

    def __init__(
        self,
        reducer: Optional[EventReducer] = None,
    ) -> None:
        self.reducer = reducer or DefaultEventReducer()

    def aggregate(
        self,
        events: Iterable[NormalizedEvent],
        *,
        context: Optional[ResponseAggregationContext] = None,
    ) -> NormalizedResponse:
        context = context or ResponseAggregationContext()
        state = AggregationState()

        for event in events:
            self.reducer.reduce(state, event)

        finalize = getattr(self.reducer, "finalize_tool_calls", None)
        if callable(finalize):
            finalize(state)

        modalities = []
        for event in state.events:
            if event.modality is not Modality.UNKNOWN:
                if event.modality not in modalities:
                    modalities.append(event.modality)

        if len(modalities) > 1 and Modality.MULTIMODAL not in modalities:
            modalities.append(Modality.MULTIMODAL)

        return NormalizedResponse(
            provider=context.provider,
            endpoint=context.endpoint,
            response_kind=context.response_kind,
            stream_mode=context.stream_mode,
            modalities=modalities or [Modality.UNKNOWN],
            request=context.request,
            response_id=context.response_id,
            model=context.model,
            created_at=context.created_at,
            status=state.status or context.status,
            finish_reason=state.finish_reason or context.finish_reason,
            visible_text="".join(state.visible_parts),
            reasoning=NormalizedReasoning(
                text="".join(state.reasoning_parts)
            ),
            tool_calls=state.tool_calls,
            audio=state.audio,
            images=state.images,
            embeddings=state.embeddings,
            events=state.events,
            usage=state.usage,
            error=state.error,
            metadata=dict(context.metadata),
            raw_response=context.raw_response,
        )
