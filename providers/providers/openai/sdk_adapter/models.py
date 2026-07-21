from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class OpenAIUsageModel(OpenAIBaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_tokens_details: Optional[Dict[str, Any]] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None
    input_tokens_details: Optional[Dict[str, Any]] = None
    output_tokens_details: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def fill_total(self) -> "OpenAIUsageModel":
        if self.total_tokens is None:
            input_value = (
                self.input_tokens
                if self.input_tokens is not None
                else self.prompt_tokens
            )
            output_value = (
                self.output_tokens
                if self.output_tokens is not None
                else self.completion_tokens
            )
            if input_value is not None and output_value is not None:
                self.total_tokens = input_value + output_value
        return self


class OpenAIContentPart(OpenAIBaseModel):
    type: Optional[str] = None
    text: Optional[str] = None
    refusal: Optional[str] = None
    transcript: Optional[str] = None
    image_url: Any = None
    input_audio: Any = None


class OpenAIFunctionModel(OpenAIBaseModel):
    name: Optional[str] = None
    arguments: str = ""


class OpenAIToolCallModel(OpenAIBaseModel):
    id: Optional[str] = None
    call_id: Optional[str] = None
    type: str = "function"
    function: Optional[OpenAIFunctionModel] = None
    name: Optional[str] = None
    arguments: Optional[Union[str, Dict[str, Any]]] = None
    status: Optional[str] = None


class ChatCompletionMessageModel(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[Union[str, List[OpenAIContentPart]]] = None
    reasoning: Any = None
    reasoning_content: Any = None
    tool_calls: List[OpenAIToolCallModel] = Field(default_factory=list)
    function_call: Optional[OpenAIFunctionModel] = None
    audio: Any = None


class ChatCompletionChoiceModel(OpenAIBaseModel):
    index: int = 0
    message: ChatCompletionMessageModel
    finish_reason: Optional[str] = None


class ChatCompletionResponseModel(OpenAIBaseModel):
    id: Optional[str] = None
    object: Literal["chat.completion"] = "chat.completion"
    model: Optional[str] = None
    created: Optional[int] = None
    choices: List[ChatCompletionChoiceModel]
    usage: Optional[OpenAIUsageModel] = None


class ResponsesContentPartModel(OpenAIBaseModel):
    type: str
    text: Optional[str] = None
    refusal: Optional[str] = None
    transcript: Optional[str] = None


class ResponsesOutputItemModel(OpenAIBaseModel):
    id: Optional[str] = None
    call_id: Optional[str] = None
    type: str
    role: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[Union[str, Dict[str, Any]]] = None
    status: Optional[str] = None
    content: List[ResponsesContentPartModel] = Field(default_factory=list)
    summary: List[ResponsesContentPartModel] = Field(default_factory=list)


class ResponsesAPIResponseModel(OpenAIBaseModel):
    id: Optional[str] = None
    object: Literal["response"] = "response"
    model: Optional[str] = None
    created_at: Optional[int] = None
    status: Optional[str] = None
    output_text: Optional[str] = None
    output: List[ResponsesOutputItemModel] = Field(default_factory=list)
    usage: Optional[OpenAIUsageModel] = None
    error: Optional[Dict[str, Any]] = None
    incomplete_details: Optional[Dict[str, Any]] = None


class OpenAIRequestEnvelope(OpenAIBaseModel):
    model: Optional[str] = None
    stream: bool = False
    messages: Optional[List[Dict[str, Any]]] = None
    instructions: Any = None
    input: Any = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Any = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
