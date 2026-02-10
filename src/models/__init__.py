from models.api_model import APIModel
from models.deepseek import DeepSeek
from models.qwen3 import Qwen3
from models.qwen3_vllm_api import Qwen3API
from models.gemma3_vllm_api import Gemma3API
from models.gpt_oss_vllm_api import GPTOssAPI
from models.llama3_vllm_api import Llama3API
from utils.utils import get_logger

logger = get_logger(__name__)

API_MODELS = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-5-mini'
]

def load_model(model_name: str, api_key: str = None, model_path: str = None, vllmapi=True, **kwargs):
    logger.info("Loading model: name=%s, vllmapi=%s", model_name, vllmapi)
    model_name = model_name.lower()
    if model_name in API_MODELS:
        assert api_key, "API models need api_key !"
        model = APIModel(model=model_name, api_key=api_key)
        logger.info("Initialized APIModel for %s", model_name)
        return model
    elif model_name == 'deepseek-chat':
        assert api_key, "DeepSeek-Chat models need api_key !"
        model = DeepSeek(api_key=api_key)
        logger.info("Initialized DeepSeek model")
        return model
    elif vllmapi and model_name.startswith('qwen3'):
        assert api_key, "Qwen3 API models need api_key!"
        assert model_path, "Qwen3 API models need model_path!"
        model = Qwen3API(model=model_path, api_key=api_key, **kwargs)
        logger.info("Initialized Qwen3API model_path=%s", model_path)
        return model
    elif vllmapi and model_name.startswith('gemma'):
        assert api_key, "Gemma3 API models need api_key!"
        assert model_path, "Gemma3 API models need model_path!"
        model = Gemma3API(model=model_path, api_key=api_key, **kwargs)
        logger.info("Initialized Gemma3API model_path=%s", model_path)
        return model
    elif vllmapi and model_name.startswith('gpt-oss'):
        assert api_key, "GPT oss API models need api_key!"
        assert model_path, "GPT oss API models need model_path!"
        model = GPTOssAPI(model=model_path, api_key=api_key, **kwargs)
        logger.info("Initialized GPTOssAPI model_path=%s", model_path)
        return model
    elif vllmapi and model_name.startswith('meta-llama'):
        assert api_key, "Llama3 API models need api_key!"
        assert model_path, "Llama3 API models need model_path!"
        model = Llama3API(model=model_path, api_key=api_key, **kwargs)
        logger.info("Initialized Llama3API model_path=%s", model_path)
        return model
    elif model_name.startswith('qwen3'):
        assert model_path, "Qwen3 models need model path!"
        model = Qwen3(model_path=model_path, **kwargs)
        logger.info("Initialized local Qwen3 model_path=%s", model_path)
        return model
    else:
        raise ValueError(f"Unsupported model '{model_name}' !")
