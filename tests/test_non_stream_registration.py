from genai_normalizer.enums import Provider
from genai_normalizer.factory import ParserFactory
from genai_normalizer.providers.openai.non_stream import (
    NonStreamParser,
    register_openai_non_stream_parser,
)


def test_registration():
    register_openai_non_stream_parser()
    parser = ParserFactory.create(Provider.OPENAI)
    assert isinstance(parser, NonStreamParser)
