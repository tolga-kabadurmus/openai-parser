from genai_normalizer.providers.openai.non_stream import NonStreamParser


class FakeSDKResponse:
    def model_dump(self, **kwargs):
        return {
            "id": "chatcmpl_sdk",
            "object": "chat.completion",
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "SDK object",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        }


def test_sdk_like_object():
    parser = NonStreamParser()
    response = parser.normalize_response(
        FakeSDKResponse(),
        endpoint="/v1/chat/completions",
    )

    assert response.visible_text == "SDK object"
    assert response.usage.total_tokens == 4
