from genai_normalizer.enums import Endpoint
from genai_normalizer.providers.openai.stream import SSEStreamParser


def test_stream_responses():
    parser = SSEStreamParser(
        request_payload={
            "model": "gpt-test",
            "stream": True,
            "instructions": "Be concise.",
            "input": "Hello",
        },
        endpoint="/v1/responses",
    )

    parser.feed(
        'data: {"type":"response.output_text.delta","sequence_number":1,"delta":"Hel"}\n\n'
    )
    parser.feed(
        'data: {"type":"response.output_text.delta","sequence_number":2,"delta":"lo"}\n\n'
    )
    parser.feed(
        'data: {"type":"response.function_call_arguments.delta","sequence_number":3,'
        '"call_id":"call_1","delta":"{\\"q\\":"}\n\n'
    )
    parser.feed(
        'data: {"type":"response.function_call_arguments.delta","sequence_number":4,'
        '"call_id":"call_1","delta":"\\"x\\"}"}\n\n'
    )
    parser.feed(
        'data: {"type":"response.completed","sequence_number":5,'
        '"response":{"id":"resp_1","object":"response","status":"completed","model":"gpt-test",'
        '"usage":{"input_tokens":4,"output_tokens":3,"total_tokens":7}}}\n\n'
    )

    response = parser.finalize()

    assert response.endpoint is Endpoint.OPENAI_RESPONSES
    assert response.visible_text == "Hello"
    assert response.tool_calls[0].arguments == {"q": "x"}
    assert response.status == "completed"
    assert response.usage.output_tokens == 3
