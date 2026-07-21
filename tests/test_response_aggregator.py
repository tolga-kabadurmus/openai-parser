from genai_normalizer.aggregation import (
    ResponseAggregationContext,
    ResponseAggregator,
)
from genai_normalizer.enums import (
    Endpoint,
    EventType,
    Modality,
    Provider,
    ResponseKind,
    StreamMode,
)
from genai_normalizer.models import NormalizedEvent


def test_text_reasoning_usage_and_status():
    events = [
        NormalizedEvent(
            event_type=EventType.TEXT_DELTA,
            modality=Modality.TEXT,
            text="Hel",
            sequence_number=1,
        ),
        NormalizedEvent(
            event_type=EventType.TEXT_DELTA,
            modality=Modality.TEXT,
            text="lo",
            sequence_number=2,
        ),
        NormalizedEvent(
            event_type=EventType.REASONING_DELTA,
            modality=Modality.REASONING,
            text="Think",
            sequence_number=3,
        ),
        NormalizedEvent(
            event_type=EventType.USAGE,
            metadata={
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                }
            },
            sequence_number=4,
        ),
        NormalizedEvent(
            event_type=EventType.COMPLETED,
            metadata={"finish_reason": "stop"},
            sequence_number=5,
        ),
    ]

    response = ResponseAggregator().aggregate(
        events,
        context=ResponseAggregationContext(
            provider=Provider.OPENAI,
            endpoint=Endpoint.OPENAI_RESPONSES,
            response_kind=ResponseKind.LLM,
            stream_mode=StreamMode.STREAMING,
        ),
    )

    assert response.visible_text == "Hello"
    assert response.reasoning.text == "Think"
    assert response.usage.total_tokens == 6
    assert response.finish_reason == "stop"


def test_tool_call_reconstruction():
    events = [
        NormalizedEvent(
            event_type=EventType.TOOL_CALL_DELTA,
            modality=Modality.TOOL,
            text='{"q":',
            tool_call_id="call_1",
            tool_name="lookup",
            sequence_number=1,
        ),
        NormalizedEvent(
            event_type=EventType.TOOL_CALL_DELTA,
            modality=Modality.TOOL,
            text='"x"}',
            tool_call_id="call_1",
            sequence_number=2,
        ),
    ]

    response = ResponseAggregator().aggregate(events)

    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"q": "x"}
