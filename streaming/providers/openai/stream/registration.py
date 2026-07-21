from genai_normalizer.providers.openai.stream.sse_stream_parser import SSEStreamParser


def create_openai_stream_parser(**kwargs):
    """Small factory helper kept separate from the non-stream parser factory."""
    return SSEStreamParser(**kwargs)
