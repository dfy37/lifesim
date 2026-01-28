python -m extraction \
  --input-root /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/twitter_archive_cleaned/twitter_archive_cleaned/shards \
  --output-root /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/events_labeled_data \
  --model deepseek-chat \
  --base-url https://api.deepseek.com \
  --api-key sk-785db80201014ade891d1db0525e48fd \
  --text-col tweet_text \
  --time-col created_at \
  --workers 8 \
  --chunk-size 10