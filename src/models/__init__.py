from models.cloud import APIModel, DeepSeek
from models.vllm import VLLMModel, Qwen3API, Gemma3API

CLOUD_API_MODELS = {
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-5-mini',
    'gpt-5',
    'claude-sonnet-4-5-20250929',
}

DEEPSEEK_MODELS = {
    'deepseek-chat',
    'deepseek-reasoner',
}

def load_model(model_name: str, api_key: str = None, model_path: str = None, vllmapi: bool = True, **kwargs):
    name = model_name.lower()

    if name in CLOUD_API_MODELS:
        assert api_key, "Cloud API models require an api_key!"
        return APIModel(model=name, api_key=api_key)

    if name in DEEPSEEK_MODELS:
        assert api_key, "DeepSeek models require an api_key!"
        return DeepSeek(model=name, api_key=api_key)

    if vllmapi:
        assert api_key,    f"vLLM model '{name}' requires an api_key!"
        assert model_path, f"vLLM model '{name}' requires a model_path!"

        if name.startswith('qwen3'):
            return Qwen3API(model=model_path, api_key=api_key, **kwargs)
        if name.startswith('gemma'):
            return Gemma3API(model=model_path, api_key=api_key, **kwargs)
        if name.startswith('gpt-oss') or name.startswith('meta-llama'):
            return VLLMModel(model=model_path, api_key=api_key, **kwargs)

    raise ValueError(f"Unsupported model '{model_name}'!")
