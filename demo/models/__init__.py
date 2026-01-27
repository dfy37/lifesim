from models.api_model import APIModel
from models.deepseek import DeepSeek
from models.qwen3 import Qwen3
from models.qwen3_vllm_api import Qwen3API
from models.gemma3_vllm_api import Gemma3API
from models.gpt_oss_vllm_api import GPTOssAPI
from models.llama3_vllm_api import Llama3API

API_MODELS = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-5-mini'
]

def load_model(model_name: str, api_key: str = None, model_path: str = None, vllmapi=True, **kwargs):
    model_name = model_name.lower()
    if model_name in API_MODELS:
        assert api_key, "API models need api_key !"
        return APIModel(model=model_name, api_key=api_key)
    elif model_name == 'deepseek-chat':
        assert api_key, "DeepSeek-Chat models need api_key !"
        return DeepSeek(api_key=api_key)
    elif vllmapi and model_name.startswith('qwen3'):
        assert api_key, "Qwen3 API models need api_key!"
        assert model_path, "Qwen3 API models need model_path!"
        return Qwen3API(model=model_path, api_key=api_key, **kwargs)
    elif vllmapi and model_name.startswith('gemma'):
        assert api_key, "Gemma3 API models need api_key!"
        assert model_path, "Gemma3 API models need model_path!"
        return Gemma3API(model=model_path, api_key=api_key, **kwargs)
    elif vllmapi and model_name.startswith('gpt-oss'):
        assert api_key, "GPT oss API models need api_key!"
        assert model_path, "GPT oss API models need model_path!"
        return GPTOssAPI(model=model_path, api_key=api_key, **kwargs)
    elif vllmapi and model_name.startswith('meta-llama'):
        assert api_key, "Llama3 API models need api_key!"
        assert model_path, "Llama3 API models need model_path!"
        return Llama3API(model=model_path, api_key=api_key, **kwargs)
    elif model_name.startswith('qwen3'):
        assert model_path, "Qwen3 models need model path!"
        return Qwen3(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unsupported model '{model_name}' !")