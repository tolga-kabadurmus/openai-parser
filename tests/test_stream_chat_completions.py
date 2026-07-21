from genai_normalizer.enums import Endpoint
from genai_normalizer.providers.openai.stream import SSEStreamParser


def test_stream_chat_completions():
    parser = SSEStreamParser(
        request_payload={
            "model": "gpt-test",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        },
        endpoint="/v1/chat/completions",
    )

    parser.feed(
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-test",'
        '"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
    )
    parser.feed(
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-test",'
        '"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}\n\n'
    )
    parser.feed(
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-test",'
        '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
    )
    parser.feed("data: [DONE]\n\n")

    response = parser.finalize()

    assert response.endpoint is Endpoint.OPENAI_CHAT_COMPLETIONS
    assert response.visible_text == "Hello"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 5
    assert parser.cost_usage()["prompt_tokens"] == 3
