# GenAI Normalizer — Module 5: Provider-Independent Aggregation

Depends on Module 1.

Provides a provider-neutral aggregation pipeline:

Raw provider events
    -> NormalizedEvent
    -> ResponseAggregator
    -> NormalizedResponse

Implemented:

- ResponseAggregationContext
- ResponseAggregator
- EventReducer protocol
- DefaultEventReducer
- Tool-call delta reconstruction
- Text and reasoning aggregation
- Status, finish reason, usage and error aggregation
- Duplicate-safe event handling
- Tests
