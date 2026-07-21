# GenAI Normalizer — Module 2: OpenAI Provider

This archive extends Module 1 with OpenAI-specific provider logic.

Implemented:

- OpenAI endpoint detection
- OpenAI request normalization
- OpenAI non-streaming response normalization
- OpenAI event normalization for Chat Completions and Responses API events
- Modality and response-kind detection
- Usage extraction
- Provider parser façade
- ParserFactory registration helper
- Tests using raw dictionaries only; the OpenAI Python SDK is not required

Module 1 must already be present on the Python path.
