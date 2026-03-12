import time
from utils.utils import get_logger
from models.base import BaseModel


class APIModel(BaseModel):
    """Cloud API models (GPT, Claude, etc.) via third-party proxy."""

    BASE_URL = "https://api.ai-gaochao.cn/v1"

    def __init__(self, model: str, api_key: str, logger_silent: bool = False):
        super().__init__(model=model, api_key=api_key, base_url=self.BASE_URL)
        self.logger = get_logger(__name__, silent=logger_silent)

    def chat(self, messages: list) -> str:
        max_retries = 5
        response_content = ""
        for attempt in range(max_retries):
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
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Error after {max_retries} attempts: {e}")
                else:
                    time.sleep(2 ** attempt)
        return response_content


class DeepSeek(BaseModel):
    """DeepSeek cloud API models."""

    BASE_URL = "https://api.deepseek.com"

    def __init__(self, model: str = "deepseek-chat", api_key: str = None):
        super().__init__(model=model, api_key=api_key, base_url=self.BASE_URL)
        self.logger = get_logger(__name__)

    def chat(self, messages: list) -> str:
        max_retries = 5
        response_content = ""
        for attempt in range(max_retries):
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
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Error after {max_retries} attempts: {e}")
                else:
                    time.sleep(2 ** attempt)
        return response_content
