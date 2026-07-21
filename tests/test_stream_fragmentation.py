from genai_normalizer.providers.openai.stream import SSEStreamParser


def test_fragmented_sse_event():
    parser = SSEStreamParser(endpoint="/v1/responses")

    parser.feed('data: {"type":"response.output_text.delta",')
    parser.feed('"delta":"Hello"}\n')
    parser.feed('\n')

    response = parser.finalize()
    assert response.visible_text == "Hello"


def test_crlf_boundaries():
    parser = SSEStreamParser(endpoint="/v1/responses")
    parser.feed(
        'data: {"type":"response.output_text.delta","delta":"Hello"}\r\n\r\n'
    )
    response = parser.finalize()
    assert response.visible_text == "Hello"
