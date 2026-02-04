#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_sample_generation.sh \
    --profiles-path <profiles.jsonl> \
    --event-sequences-path <events.jsonl> \
    --query-database-path <queries.jsonl> \
    --output-path <output.jsonl> \
    --model-name <model> \
    [--model-api-key <key>] \
    [--model-path <path>] \
    [--model-url <url>] \
    [--use-vllm] \
    [--max-conv-turns <n>] \
    [--max-events-number <n>] \
    [--max-profiles <n>] \
    [--random-state <n>] \
    [--seed <n>] \
    [--retriever-model-name <name>] \
    [--retriever-collection-name <name>] \
    [--retriever-embedding-dim <dim>] \
    [--retriever-persist-dir <dir>] \
    [--retriever-distance <cosine|l2|ip>] \
    [--retriever-device <cpu|cuda>] \
    [--retriever-use-custom-embeddings] \
    [--retriever-text-field <field>] \
    [--retriever-id-field <field>] \
    [--retriever-batch-size <n>] \
    [--max-retrievals <n>]

Example:
  ./run_sample_generation.sh \
    --profiles-path data/profiles.jsonl \
    --event-sequences-path data/events.jsonl \
    --query-database-path data/queries.jsonl \
    --output-path outputs/conv_histories.jsonl \
    --model-name gpt-4o-mini \
    --model-api-key "$OPENAI_API_KEY"
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -eq 0 ]]; then
  usage
  exit 0
fi

python sample_generation.py "$@"
