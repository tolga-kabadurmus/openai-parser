from enum import Enum
class ResponseKind(str,Enum):
    LLM="llm"; VOICE_STT="voice_stt"; VOICE_TTS="voice_tts"; IMAGE="image"; VIDEO="video"; EMBEDDING="embedding"; UNKNOWN="unknown"
