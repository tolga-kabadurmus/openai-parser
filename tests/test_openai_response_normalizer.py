from genai_normalizer.enums import Endpoint
from genai_normalizer.providers.openai import OpenAIResponseNormalizer


def test_chat_completions_response():
    response = OpenAIResponseNormalizer().normalize(
        {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"q":"x"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }
    )

    assert response.endpoint is Endpoint.OPENAI_CHAT_COMPLETIONS
    assert response.visible_text == "Hello"
    assert response.tool_calls[0].name == "lookup"
    assert response.usage.input_tokens == 5


def test_responses_response():
    response = OpenAIResponseNormalizer().normalize(
        {
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "model": "gpt-test",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Hello from Responses"}
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                },
            ],
            "usage": {
                "input_tokens": 8,
                "output_tokens": 4,
                "total_tokens": 12,
            },
        }
    )

    assert response.endpoint is Endpoint.OPENAI_RESPONSES
    assert response.visible_text == "Hello from Responses"
    assert response.tool_calls[0].arguments == {"q": "x"}
    assert response.usage.output_tokens == 4
