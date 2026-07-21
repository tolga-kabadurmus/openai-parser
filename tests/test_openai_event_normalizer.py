from genai_normalizer.enums import EventType
from genai_normalizer.providers.openai import OpenAIEventNormalizer


def test_chat_chunk():
    events = OpenAIEventNormalizer().normalize(
        {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Hel"}}],
        }
    )
    assert events[0].event_type is EventType.TEXT_DELTA
    assert events[0].text == "Hel"


def test_responses_text_delta():
    events = OpenAIEventNormalizer().normalize(
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "delta": "lo",
        }
    )
    assert events[0].event_type is EventType.TEXT_DELTA
    assert events[0].text == "lo"


def test_responses_tool_delta():
    events = OpenAIEventNormalizer().normalize(
        {
            "type": "response.function_call_arguments.delta",
            "call_id": "call_1",
            "delta": '{"q"',
        }
    )
    assert events[0].event_type is EventType.TOOL_CALL_DELTA
    assert events[0].tool_call_id == "call_1"
