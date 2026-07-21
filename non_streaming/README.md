# GenAI Normalizer — Module 4: OpenAI Non-Streaming

This archive extends Modules 1 and 2 with a dedicated OpenAI non-streaming layer.

Implemented:

- `NonStreamParser` public façade
- OpenAI SDK model and raw dictionary coercion
- `/v1/chat/completions` parsing
- `/v1/responses` parsing
- Embeddings parsing
- Audio transcription and translation parsing
- Image generation parsing
- Best-effort generic text response parsing
- Unified `NormalizedResponse` output
- Request normalization attachment
- Usage extraction for Chat Completions and Responses API
- ParserFactory-compatible registration helper
- Tests using raw dictionaries and SDK-like objects

Modules 1 and 2 must already be present on the Python path.
