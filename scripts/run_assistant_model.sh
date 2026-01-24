MODEL_PATH=/remote-home/share/LLM_CKPT/huggingface_models/Qwen3-32B

MAX_MODEL_LENGTH=16384

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len ${MAX_MODEL_LENGTH} \
    --api-key 123
