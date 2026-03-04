# python sample_generation.py \
#     --profiles-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/demo_data/users.jsonl \
#     --query-database-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/intention_kept_pool/sports.jsonl \
#     --output-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/test_data/sports_conv_history.jsonl \
#     --model-name gpt-oss-120b \
#     --model-api-key "123" \
#     --model-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/gpt-oss-120b \
#     --model-url http://0.0.0.0:8001/v1 \
#     --use-vllm \
#     --max-conv-turns 20 \
#     --max-events-number 10 \
#     --max-profiles 15 \
#     --seed 42 \
#     --retriever-model-name /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B \
#     --retriever-collection-name sport_intention_database \
#     --retriever-embedding-dim 1024 \
#     --retriever-persist-dir /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/chrome_db \
#     --retriever-distance cosine \
#     --retriever-device cpu \
#     --retriever-text-field intention \
#     --retriever-id-field id \
#     --retriever-batch-size 128 \
#     --max-retrievals 3 \
#     --num-workers 2 \
#     --theme sport_health


python sample_generation.py \
    --profiles-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/demo_data/users.jsonl \
    --query-database-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/intention_kept_pool/travel.jsonl \
    --output-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/test_data/travel_conv_history.jsonl \
    --model-name gpt-oss-120b \
    --model-api-key "123" \
    --model-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/gpt-oss-120b \
    --model-url http://0.0.0.0:8001/v1 \
    --use-vllm \
    --max-conv-turns 20 \
    --max-events-number 10 \
    --max-profiles 15 \
    --seed 42 \
    --retriever-model-name /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B \
    --retriever-collection-name sport_intention_database \
    --retriever-embedding-dim 1024 \
    --retriever-persist-dir /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/chrome_db \
    --retriever-distance cosine \
    --retriever-device cpu \
    --retriever-text-field intention \
    --retriever-id-field id \
    --retriever-batch-size 128 \
    --max-retrievals 3 \
    --num-workers 2 \
    --theme travel

# python sample_generation.py \
#     --profiles-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/demo_data/users.jsonl \
#     --query-database-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/intention_kept_pool/mental_health.jsonl \
#     --output-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/test_data/mental_health_conv_history.jsonl \
#     --model-name gpt-oss-120b \
#     --model-api-key "123" \
#     --model-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/gpt-oss-120b \
#     --model-url http://0.0.0.0:8001/v1 \
#     --use-vllm \
#     --max-conv-turns 20 \
#     --max-events-number 10 \
#     --max-profiles 15 \
#     --seed 42 \
#     --retriever-model-name /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B \
#     --retriever-collection-name sport_intention_database \
#     --retriever-embedding-dim 1024 \
#     --retriever-persist-dir /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/chrome_db \
#     --retriever-distance cosine \
#     --retriever-device cpu \
#     --retriever-text-field intention \
#     --retriever-id-field id \
#     --retriever-batch-size 128 \
#     --max-retrievals 3 \
#     --num-workers 2 \
#     --theme mental_health

# python sample_generation.py \
#     --profiles-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/demo_data/users.jsonl \
#     --query-database-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/intention_kept_pool/entertainment.jsonl \
#     --output-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/test_data/entertainment_conv_history.jsonl \
#     --model-name gpt-oss-120b \
#     --model-api-key "123" \
#     --model-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/gpt-oss-120b \
#     --model-url http://0.0.0.0:8001/v1 \
#     --use-vllm \
#     --max-conv-turns 20 \
#     --max-events-number 10 \
#     --max-profiles 15 \
#     --seed 42 \
#     --retriever-model-name /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B \
#     --retriever-collection-name sport_intention_database \
#     --retriever-embedding-dim 1024 \
#     --retriever-persist-dir /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/chrome_db \
#     --retriever-distance cosine \
#     --retriever-device cpu \
#     --retriever-text-field intention \
#     --retriever-id-field id \
#     --retriever-batch-size 128 \
#     --max-retrievals 3 \
#     --num-workers 2 \
#     --theme entertainment

# python sample_generation.py \
#     --profiles-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/demo_data/users.jsonl \
#     --query-database-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/intention_kept_pool/elderlycare.jsonl \
#     --output-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/test_data/elderlycare_conv_history.jsonl \
#     --model-name gpt-oss-120b \
#     --model-api-key "123" \
#     --model-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/gpt-oss-120b \
#     --model-url http://0.0.0.0:8001/v1 \
#     --use-vllm \
#     --max-conv-turns 20 \
#     --max-events-number 10 \
#     --max-profiles 15 \
#     --seed 42 \
#     --retriever-model-name /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B \
#     --retriever-collection-name sport_intention_database \
#     --retriever-embedding-dim 1024 \
#     --retriever-persist-dir /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/chrome_db \
#     --retriever-distance cosine \
#     --retriever-device cpu \
#     --retriever-text-field intention \
#     --retriever-id-field id \
#     --retriever-batch-size 128 \
#     --max-retrievals 3 \
#     --num-workers 2 \
#     --theme elderlycare

# python sample_generation.py \
#     --profiles-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/demo_data/users.jsonl \
#     --query-database-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/intention_kept_pool/education.jsonl \
#     --output-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/test_data/education_conv_history.jsonl \
#     --model-name gpt-oss-120b \
#     --model-api-key "123" \
#     --model-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/gpt-oss-120b \
#     --model-url http://0.0.0.0:8001/v1 \
#     --use-vllm \
#     --max-conv-turns 20 \
#     --max-events-number 10 \
#     --max-profiles 15 \
#     --seed 42 \
#     --retriever-model-name /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B \
#     --retriever-collection-name sport_intention_database \
#     --retriever-embedding-dim 1024 \
#     --retriever-persist-dir /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/chrome_db \
#     --retriever-distance cosine \
#     --retriever-device cpu \
#     --retriever-text-field intention \
#     --retriever-id-field id \
#     --retriever-batch-size 128 \
#     --max-retrievals 3 \
#     --num-workers 2 \
#     --theme education

# python sample_generation.py \
#     --profiles-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/demo_data/users.jsonl \
#     --query-database-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/intention_kept_pool/dining.jsonl \
#     --output-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/test_data/dining_conv_history.jsonl \
#     --model-name gpt-oss-120b \
#     --model-api-key "123" \
#     --model-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/gpt-oss-120b \
#     --model-url http://0.0.0.0:8001/v1 \
#     --use-vllm \
#     --max-conv-turns 20 \
#     --max-events-number 10 \
#     --max-profiles 15 \
#     --seed 42 \
#     --retriever-model-name /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B \
#     --retriever-collection-name sport_intention_database \
#     --retriever-embedding-dim 1024 \
#     --retriever-persist-dir /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/chrome_db \
#     --retriever-distance cosine \
#     --retriever-device cpu \
#     --retriever-text-field intention \
#     --retriever-id-field id \
#     --retriever-batch-size 128 \
#     --max-retrievals 3 \
#     --num-workers 2 \
#     --theme dining

python sample_generation.py \
    --profiles-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/demo_data/users.jsonl \
    --query-database-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/intention_kept_pool/childcare.jsonl \
    --output-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/test_data/childcare_conv_history.jsonl \
    --model-name gpt-oss-120b \
    --model-api-key "123" \
    --model-path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/gpt-oss-120b \
    --model-url http://0.0.0.0:8001/v1 \
    --use-vllm \
    --max-conv-turns 20 \
    --max-events-number 10 \
    --max-profiles 15 \
    --seed 42 \
    --retriever-model-name /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B \
    --retriever-collection-name sport_intention_database \
    --retriever-embedding-dim 1024 \
    --retriever-persist-dir /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/chrome_db \
    --retriever-distance cosine \
    --retriever-device cpu \
    --retriever-text-field intention \
    --retriever-id-field id \
    --retriever-batch-size 128 \
    --max-retrievals 3 \
    --num-workers 2 \
    --theme childcare