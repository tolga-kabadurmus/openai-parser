from genai_normalizer.providers.openai.stream import OpenAIStreamCollector


def test_collector():
    response = OpenAIStreamCollector().collect(
        [
            'data: {"object":"chat.completion.chunk","choices":[{"index":0,'
            '"delta":{"content":"A"},"finish_reason":null}]}\n\n',
            'data: {"object":"chat.completion.chunk","choices":[{"index":0,'
            '"delta":{"content":"B"},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ],
        endpoint="/v1/chat/completions",
    )

    assert response.visible_text == "AB"
