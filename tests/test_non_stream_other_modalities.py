from genai_normalizer.providers.openai.non_stream import NonStreamParser


def test_embeddings():
    response = NonStreamParser().normalize_response(
        {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2],
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "total_tokens": 3,
            },
        },
        endpoint="/v1/embeddings",
    )

    assert response.embeddings[0].values == [0.1, 0.2]


def test_transcription():
    response = NonStreamParser().normalize_response(
        {
            "text": "transcribed text",
            "duration": 1.5,
        },
        endpoint="/v1/audio/transcriptions",
    )

    assert response.visible_text == "transcribed text"
    assert response.audio[0].duration_seconds == 1.5


def test_images():
    response = NonStreamParser().normalize_response(
        {
            "data": [
                {
                    "url": "https://example.invalid/image.png",
                    "revised_prompt": "A revised prompt",
                }
            ]
        },
        endpoint="/v1/images/generations",
    )

    assert response.images[0].revised_prompt == "A revised prompt"
