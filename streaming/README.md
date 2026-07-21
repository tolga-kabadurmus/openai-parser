# GenAI Normalizer — Module 3: OpenAI Streaming

This archive extends Modules 1 and 2 with OpenAI streaming support.

Implemented:

- SSE framing and chunk buffering
- Support for `data: [DONE]`
- Support for both LF and CRLF SSE boundaries
- Raw OpenAI event storage
- Conversion into `NormalizedEvent`
- Aggregation into a unified `NormalizedResponse`
- Chat Completions text, reasoning, tool-call, finish-reason, and usage handling
- Responses API text, reasoning, function arguments, completion status, and usage handling
- `SSEStreamParser` integration with Module 2's `OpenAIEventNormalizer`
- Stream parser factory helper
- Tests using raw SSE strings

Modules 1 and 2 must already be present on the Python path.
