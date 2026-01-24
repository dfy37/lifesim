from openai import OpenAI

class DeepSeek:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.deepseek.com"
        )
        self.messages = []
    
    def chat(self, messages):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                temperature=1.0
            )
            response_content = response.choices[0].message.content
            messages1 = messages.copy()
            messages1.append({"role": "assistant", "content": response_content})
            self.messages.append(messages1)
        except Exception as e:
            response_content = ""

        return response_content
    
    def save(self, file_path = "./deepseek_model.jsonl"):
        import json
        with open(file_path, 'w') as file:
            for message in self.messages:
                file.write(f"{json.dumps(message, ensure_ascii=False)}\n")