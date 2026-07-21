import pydantic

from genai_normalizer.providers.openai.sdk_adapter import OpenAISDKAdapter


def test_pydantic_major_version():
    assert int(pydantic.VERSION.split(".")[0]) == 2


def test_chat_completion_adapter():
    response = OpenAISDKAdapter().normalize_chat_completion(
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
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }
    )

    assert response.visible_text == "Hello"
    assert response.usage.total_tokens == 5


def test_responses_adapter():
    response = OpenAISDKAdapter().normalize_responses_api(
        {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello",
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
        }
    )

    assert response.visible_text == "Hello"
    assert response.usage.input_tokens == 4
