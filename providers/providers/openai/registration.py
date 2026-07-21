from genai_normalizer.enums import Provider
from genai_normalizer.factory import ParserFactory

from .parser import OpenAIParser


def register_openai_parser() -> None:
    """Register OpenAIParser in Module 1's ParserFactory."""
    ParserFactory.register(Provider.OPENAI, OpenAIParser)
