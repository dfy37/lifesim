from openai import OpenAI
import time
from utils.utils import get_logger

class APIModel:
    def __init__(self, api_key, model, logger_silent=False):
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.ai-gaochao.cn/v1"
        )
        self.messages = []
        self.model = model
        self.logger = get_logger(__name__, silent=logger_silent)
    
    def chat(self, messages):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1.0
                )
                response_content = response.choices[0].message.content
                messages1 = messages.copy()
                messages1.append({"role": "assistant", "content": response_content})
                self.messages.append(messages1)
                break
            except Exception as e:
                self.logger.error(f"Error occurred during API call: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    self.logger.error(f"Error occurred after {max_retries} attempts: {e}")
                    response_content = ""
                else:
                    # Wait before retry
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

        return response_content
    
    def save(self, file_path: str = None):
        import json
        if not file_path:
            file_path = f"./api_model_{self.model}.jsonl"
        with open(file_path, 'w') as file:
            for message in self.messages:
                file.write(f"{json.dumps(message, ensure_ascii=False)}\n")