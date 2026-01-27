from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Qwen3:
    def __init__(self, model_path="Qwen/Qwen2.5-32B-Instruct", **kwargs):
        """
        初始化Qwen32B模型
        
        Args:
            model_path: 模型路径，可以是本地路径或HuggingFace模型名称
            **kwargs: 传递给vLLM的其他参数，如gpu_memory_utilization, tensor_parallel_size等
        """
        # 设置默认参数
        default_kwargs = {
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
            "dtype": "auto",
            "max_model_len": 8192,
            "tensor_parallel_size": 1
        }
        default_kwargs.update(kwargs)
        
        # 初始化vLLM模型
        self.llm = LLM(model=model_path, **default_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=default_kwargs['trust_remote_code'])
        self.messages = []
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=2048,
            top_p=0.9,
        )
    
    def _format_messages_to_prompt(self, messages):
        """
        将OpenAI格式的messages转换为Qwen的prompt格式
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return prompt
    
    def chat(self, messages):
        """
        与模型对话
        
        Args:
            messages: OpenAI格式的消息列表
        
        Returns:
            str: 模型回复内容
        """
        try:
            prompt = self._format_messages_to_prompt(messages)
            
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            response_content = outputs[0].outputs[0].text.strip().rstrip()
            
            messages1 = messages.copy()
            messages1.append({"role": "assistant", "content": response_content})
            self.messages.append(messages1)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            response_content = ""
        
        return response_content
    
    def save(self, file_path="./qwen3_model.jsonl"):
        """
        保存对话历史到文件
        
        Args:
            file_path: 保存文件路径
        """
        import json
        with open(file_path, 'w', encoding='utf-8') as file:
            for message in self.messages:
                file.write(f"{json.dumps(message, ensure_ascii=False)}\n")