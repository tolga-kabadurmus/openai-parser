from enum import Enum
class Provider(str,Enum):
    OPENAI="openai"; ANTHROPIC="anthropic"; COHERE="cohere"; UNKNOWN="unknown"
