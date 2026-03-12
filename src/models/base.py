import json
import os
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model: str, api_key: str, base_url: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.messages = []

    @abstractmethod
    def chat(self, messages: list) -> str:
        ...

    def save(self, file_path: str = None):
        if not file_path:
            model_name = os.path.basename(self.model)
            file_path = f"./model_{model_name}.jsonl"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for message in self.messages:
                f.write(json.dumps(message, ensure_ascii=False) + '\n')
