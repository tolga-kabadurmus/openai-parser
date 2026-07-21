from genai_normalizer.enums import Provider
from genai_normalizer.factory import ParserFactory

from .non_stream_parser import NonStreamParser


def register_openai_non_stream_parser() -> None:
    """
    Register NonStreamParser for OpenAI.

    This intentionally replaces any previously registered OpenAI parser in
    Module 1's simple provider registry.
    """
    ParserFactory.register(Provider.OPENAI, NonStreamParser)
