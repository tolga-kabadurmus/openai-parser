from genai_normalizer.enums import Endpoint, Provider, StreamMode
from genai_normalizer.providers.openai import OpenAIRequestNormalizer


def test_chat_completions_request():
    request = OpenAIRequestNormalizer().normalize(
        {
            "model": "gpt-test",
            "stream": True,
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Hello"},
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert request.provider is Provider.OPENAI
    assert request.endpoint is Endpoint.OPENAI_CHAT_COMPLETIONS
    assert request.stream_mode is StreamMode.STREAMING
    assert request.normalized_text() == "Be precise. Hello"


def test_responses_request():
    request = OpenAIRequestNormalizer().normalize(
        {
            "model": "gpt-test",
            "instructions": "Be precise.",
            "input": [{"role": "user", "content": "Hello"}],
        }
    )

    assert request.endpoint is Endpoint.OPENAI_RESPONSES
    assert "Hello" in request.normalized_text()
