from enum import Enum
class EventType(str,Enum):
    TEXT_DELTA="text_delta"; TEXT_DONE="text_done"; REASONING_DELTA="reasoning_delta"; REASONING_DONE="reasoning_done"
    TOOL_CALL_DELTA="tool_call_delta"; TOOL_CALL_DONE="tool_call_done"; AUDIO_DELTA="audio_delta"; AUDIO_TRANSCRIPT_DELTA="audio_transcript_delta"
    IMAGE="image"; VIDEO="video"; EMBEDDING="embedding"; USAGE="usage"; COMPLETED="completed"; FAILED="failed"; DONE="done"; UNKNOWN="unknown"
