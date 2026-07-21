from genai_normalizer.enums import Endpoint
from genai_normalizer.providers.openai.non_stream import NonStreamParser


def test_responses_non_stream():
    parser = NonStreamParser()

    response = parser.parse(
        request_payload={
            "model": "gpt-test",
            "instructions": "Be concise.",
            "input": "Hello",
            "stream": False,
        },
        response_payload={
            "id": "resp_1",
            "object": "response",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello from Responses",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                    "status": "completed",
                },
                {
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Checked the request.",
                        }
                    ],
                },
            ],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 4,
                "total_tokens": 11,
            },
        },
        endpoint="/v1/responses",
    )

    assert response.endpoint is Endpoint.OPENAI_RESPONSES
    assert response.visible_text == "Hello from Responses"
    assert response.reasoning.text == "Checked the request."
    assert response.tool_calls[0].arguments == {"q": "x"}
    assert response.usage.output_tokens == 4
