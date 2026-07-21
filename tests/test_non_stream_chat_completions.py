from genai_normalizer.enums import Endpoint
from genai_normalizer.providers.openai.non_stream import NonStreamParser


def test_chat_completions_non_stream():
    parser = NonStreamParser()

    response = parser.parse(
        request_payload={
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": False,
        },
        response_payload={
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello back",
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
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        },
        endpoint="/v1/chat/completions",
    )

    assert response.endpoint is Endpoint.OPENAI_CHAT_COMPLETIONS
    assert response.visible_text == "Hello back"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"q": "x"}
    assert response.usage.input_tokens == 5
    assert parser.normalized_request_text() == "Hello"
