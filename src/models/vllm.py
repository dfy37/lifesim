import re
from utils.utils import get_logger
from models.base import BaseModel

DEFAULT_BASE_URL = "http://0.0.0.0:8000/v1"

class VLLMModel(BaseModel):
    """Generic vLLM-served model via OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str, base_url: str = None):
        super().__init__(model=model, api_key=api_key, base_url=base_url or DEFAULT_BASE_URL)
        self.logger = get_logger(__name__)

    def chat(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1.0,
            )
            response_content = response.choices[0].message.content
            logged = messages.copy()
            logged.append({"role": "assistant", "content": response_content})
            self.messages.append(logged)
        except Exception as e:
            self.logger.info(e)
            response_content = ""
        return response_content


class Qwen3API(VLLMModel):
    """Qwen3 via vLLM: disables chain-of-thought and strips <think> tags."""

    def chat(self, messages: list) -> str:
        messages[-1]['content'] += ' /no_think'
        response_content = super().chat(messages)
        return re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()


class Gemma3API(VLLMModel):
    """Gemma3 via vLLM: wraps message content in multimodal format."""

    def chat(self, messages: list) -> str:
        formatted = [
            {"role": m['role'], "content": [{"type": "text", "text": m['content']}]}
            for m in messages
        ]
        return super().chat(formatted)
