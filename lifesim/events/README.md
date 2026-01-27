# Tweet Event Builder

This folder contains two scripts to build user event sequences from tweet data.

## Input data layout

```
root/
  bucket_id=0/
    data.parquet
  bucket_id=1/
    data.parquet
```

Each bucket is processed independently so records do not cross buckets. All columns
from the source parquet are preserved in the output records.

## Module B — Event Extraction

Extract atomic events from tweets per bucket using the OpenAI API.

```
python -m lifesim.events.extraction \
  --input-root /path/to/root \
  --output-root /path/to/output \
  --model gpt-4o-mini \
  --text-col text \
  --time-col created_at \
  --workers 8 \
  --chunk-size 10
```

Outputs:

```
/output/
  bucket_id=0/events_raw.jsonl
```

## Module C — Event Check (Faithfulness Audit)

Run semantic divergence checks, request revisions, and flag problematic samples.

```
python -m lifesim.events.check \
  --input-root /path/to/output \
  --output-root /path/to/output_checked \
  --generator-model gpt-4o-mini \
  --validator-model gpt-4o-mini \
  --epsilon 0.35 \
  --max-rounds 3 \
  --workers 8 \
  --chunk-size 10
```

Outputs:

```
/output_checked/
  bucket_id=0/events_checked.jsonl
```

Each record includes `delta`, `attempts`, and `status` (passed/flagged).
