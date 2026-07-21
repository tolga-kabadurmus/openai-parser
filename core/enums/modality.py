from enum import Enum
class Modality(str,Enum):
    TEXT="text"; REASONING="reasoning"; TOOL="tool"; AUDIO="audio"; IMAGE="image"; VIDEO="video"; EMBEDDING="embedding"; MULTIMODAL="multimodal"; UNKNOWN="unknown"
