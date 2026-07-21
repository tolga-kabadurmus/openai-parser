from genai_normalizer.dispatcher import ProviderDispatcher
from genai_normalizer.enums import Endpoint,Provider
from genai_normalizer.models import NormalizedRequest,NormalizedResponse,NormalizedUsage

def test_provider_detection():assert ProviderDispatcher.detect({"object":"chat.completion"}) is Provider.OPENAI
def test_usage_chat():
    u=NormalizedUsage.from_openai({"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}); assert (u.input_tokens,u.output_tokens,u.total_tokens)==(10,4,14)
def test_usage_responses():
    u=NormalizedUsage.from_openai({"input_tokens":7,"output_tokens":3,"input_tokens_details":{"cached_tokens":2}}); assert u.total_tokens==10 and u.cached_input_tokens==2
def test_models():
    r=NormalizedRequest(provider=Provider.OPENAI,endpoint=Endpoint.OPENAI_RESPONSES,instructions="Answer.",input_text="Hello")
    assert r.normalized_text()=="Answer. Hello"
    response=NormalizedResponse(provider=Provider.OPENAI,visible_text="Hi")
    assert response.to_dict()["provider"]=="openai"
