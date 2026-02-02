python -m extraction \
  --input-root /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/twitter_archive_cleaned/twitter_archive_cleaned/shards \
  --output-root /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/events_labeled_data \
  --model /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-32B \
  --base-url http://0.0.0.0:8000/v1 \
  --api-key "123" \
  --text-col tweet_text \
  --time-col created_at \
  --workers 16

# python -m check \
#   --input-root /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/events_labeled_data \
#   --output-root /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/events_labeled_data_checked \
#   --generator-model /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-32B \
#   --validator-model /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-32B \
#   --base-url http://0.0.0.0:8000/v1 \
#   --api-key "123" \
#   --epsilon 0.35 \
#   --max-rounds 3 \
#   --workers 16 \