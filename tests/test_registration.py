from genai_normalizer.enums import Provider
from genai_normalizer.factory import ParserFactory
from genai_normalizer.providers.openai import OpenAIParser, register_openai_parser


def test_registration():
    register_openai_parser()
    parser = ParserFactory.create(Provider.OPENAI)
    assert isinstance(parser, OpenAIParser)
