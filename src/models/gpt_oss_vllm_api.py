from openai import OpenAI
import re
from utils.utils import get_logger
import os

logger = get_logger(__name__)

class GPTOssAPI:
    def __init__(self, api_key, model, base_url: str = None):
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url if base_url else "http://0.0.0.0:8000/v1" 
        )
        self.messages = []
        self.model = model
    
    def chat(self, messages):
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
        except Exception as e:
            logger.info(e)
            response_content = ""

        return response_content
    
    def save(self, file_path: str = None):
        import json
        if not file_path:
            model_name = os.path.basename(self.model)
            file_path = f"./api_model_{model_name}.jsonl"
        with open(file_path, 'w') as file:
            for message in self.messages:
                file.write(f"{json.dumps(message, ensure_ascii=False)}\n")