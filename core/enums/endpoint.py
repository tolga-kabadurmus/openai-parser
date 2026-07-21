from enum import Enum
class Endpoint(str,Enum):
    OPENAI_CHAT_COMPLETIONS="/v1/chat/completions"
    OPENAI_RESPONSES="/v1/responses"
    OPENAI_EMBEDDINGS="/v1/embeddings"
    OPENAI_AUDIO_TRANSCRIPTIONS="/v1/audio/transcriptions"
    OPENAI_AUDIO_TRANSLATIONS="/v1/audio/translations"
    OPENAI_AUDIO_SPEECH="/v1/audio/speech"
    OPENAI_IMAGES_GENERATIONS="/v1/images/generations"
    OPENAI_VIDEOS="/v1/videos"
    UNKNOWN="unknown"
