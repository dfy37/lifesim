#!/usr/bin/env python3
"""Module B: Extract atomic narrative events from tweet data per bucket."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm

import duckdb
from openai import OpenAI
from json_repair import loads as repair_json

SYSTEM_PROMPT = """You are an Event Extraction agent.
Extract a single atomic Event from the given tweet.
Return ONLY valid JSON for the event with the exact schema:
{
  "time": "string or null",
  "location": "string or null",
  "environment": "string or null",
  "action": "string (first-person)",
  "observation": "string (first-person)",
  "inner_thought": "string (first-person)"
}

Hard constraints:
- Use first-person phrasing for action/observation/inner_thought.
- If information is missing, set it to null.
- For inner_thought: do NOT use abstract labels (e.g., anxious, happy).
  Use explicit inner speech or observable psychological behavior.
- Do NOT add causal explanations, motivations, or personality judgments.
- Do NOT infer facts not in the tweet.
"""

REVISION_PROMPT = """You are refining a previously extracted Event.
Given the tweet and validator feedback, produce a corrected Event JSON.
Follow the same schema and constraints. Output ONLY the JSON object.
"""


@dataclass
class Event:
    time: str | None
    location: str | None
    environment: str | None
    action: str | None
    observation: str | None
    inner_thought: str | None

    @staticmethod
    def from_mapping(data: dict[str, Any]) -> "Event":
        return Event(
            time=_normalize_optional_str(data.get("time")),
            location=_normalize_optional_str(data.get("location")),
            environment=_normalize_optional_str(data.get("environment")),
            action=_normalize_optional_str(data.get("action")),
            observation=_normalize_optional_str(data.get("observation")),
            inner_thought=_normalize_optional_str(data.get("inner_thought")),
        )


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return str(value).strip() or None


def _parse_event_json(text: str) -> Event:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = repair_json(text)
    if not isinstance(payload, dict):
        raise ValueError("Event payload must be a JSON object.")
    return Event.from_mapping(payload)


class EventExtractor:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def extract(self, tweet_text: str, tweet_time: str | None) -> Event:
        user_prompt = (
            "Tweet:\n"
            f"{tweet_text}\n\n"
            f"Tweet time: {tweet_time or 'unknown'}\n"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0,
        )
        event = _parse_event_json(response.output_text)
        return event

    def revise(self, tweet_text: str, tweet_time: str | None, feedback: str) -> Event:
        user_prompt = (
            "Tweet:\n"
            f"{tweet_text}\n\n"
            f"Tweet time: {tweet_time or 'unknown'}\n\n"
            f"Validator feedback: {feedback}\n"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": REVISION_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        event = _parse_event_json(response.output_text)
        return event


_EXTRACTOR: EventExtractor | None = None


def _init_worker(model: str, api_key: str | None, base_url: str | None) -> None:
    global _EXTRACTOR
    _EXTRACTOR = EventExtractor(model=model, api_key=api_key, base_url=base_url)

# def _process_row(
#     row_dict: dict[str, Any],
#     bucket_id: str,
#     text_col: str,
#     time_col: str | None,
# ) -> dict[str, Any] | None:
#     if _EXTRACTOR is None:
#         raise RuntimeError("Extractor worker is not initialized.")
#     tweet_text = row_dict.get(text_col)
#     if tweet_text is None:
#         return None
#     tweet_time = row_dict.get(time_col) if time_col else None
#     event = _EXTRACTOR.extract(str(tweet_text), _normalize_optional_str(tweet_time))
#     record = dict(row_dict)
#     record["bucket_id"] = bucket_id
#     record["tweet_time"] = _normalize_optional_str(tweet_time)
#     record["tweet_text"] = str(tweet_text)
#     record["event"] = asdict(event)
#     return record

def _process_row(
    row_dict: dict[str, Any],
    bucket_id: str,
    text_col: str,
    time_col: str | None,
) -> dict[str, Any] | None:
    if _EXTRACTOR is None:
        raise RuntimeError("Extractor worker is not initialized.")
    try:
        tweet_text = row_dict.get(text_col)
        if tweet_text is None:
            return None
        tweet_time = row_dict.get(time_col) if time_col else None
        event = _EXTRACTOR.extract(
            str(tweet_text),
            _normalize_optional_str(tweet_time),
        )
        record = dict(row_dict)
        record["bucket_id"] = bucket_id
        record["tweet_time"] = _normalize_optional_str(tweet_time)
        record["tweet_text"] = str(tweet_text)
        record["event"] = asdict(event)
        record["error"] = None
        return record
    except Exception as e:
        return {
            **row_dict,
            "bucket_id": bucket_id,
            "tweet_text": row_dict.get(text_col),
            "tweet_time": row_dict.get(time_col),
            "event": None,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
            },
        }



def iter_bucket_parquets(root: Path) -> Iterable[tuple[str, Path]]:
    for bucket_dir in sorted(root.glob("bucket_id=*")):
        if not bucket_dir.is_dir():
            continue
        parquet_path = bucket_dir / "data.parquet"
        if parquet_path.exists():
            yield bucket_dir.name, parquet_path


def load_bucket_dataframe(parquet_path: Path, limit: int | None = None):
    query = "SELECT * FROM read_parquet(?)"
    if limit is not None:
        query += " LIMIT ?"

    con = duckdb.connect()
    try:
        if limit is not None:
            df = con.execute(query, [str(parquet_path), limit]).fetch_df()
        else:
            df = con.execute(query, [str(parquet_path)]).fetch_df()
    finally:
        con.close()
    return df


def _select_users(df, user_col: str, max_users: int | None):
    if max_users is None:
        return df
    if user_col not in df.columns:
        raise KeyError(f"User column '{user_col}' not found in data.")
    users = df[user_col].dropna().unique().tolist()
    selected = set(users[:max_users])
    return df[df[user_col].isin(selected)]


def _aggregate_by_user(
    records: Iterable[dict[str, Any] | None],
    bucket_id: str,
    user_col: str,
) -> list[dict[str, Any]]:
    grouped: dict[Any, dict[str, Any]] = {}
    for record in records:
        if record is None:
            continue
        user_id = record.get(user_col)
        entry = grouped.get(user_id)
        if entry is None:
            entry = {
                "bucket_id": bucket_id,
                "user_id": user_id,
                "posts": [],
            }
            grouped[user_id] = entry
        entry["posts"].append(record)
    return list(grouped.values())


def write_jsonl(path: Path, records: Iterable[dict[str, Any] | None]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record is None:
                continue
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _default_workers() -> int:
    return max(1, multiprocessing.cpu_count() - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--time-col", default="created_at")
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--max-buckets", type=int, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    bucket_count = 0
    for bucket_id, parquet_path in iter_bucket_parquets(args.input_root):
        if args.max_buckets is not None and bucket_count >= args.max_buckets:
            break
        print(f"Processing bucket {bucket_id} ...")
        df = load_bucket_dataframe(
            parquet_path,
            limit=100
        )
        df = _select_users(df, args.user_col, args.max_users)
        total_rows = len(df)
        
        bucket_output = args.output_root / bucket_id
        bucket_output.mkdir(parents=True, exist_ok=True)
        rows_iter = (row._asdict() for row in df.itertuples(index=False))
        worker = partial(
            _process_row,
            bucket_id=bucket_id,
            text_col=args.text_col,
            time_col=args.time_col,
        )
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(args.model, args.api_key, args.base_url),
        ) as executor:
            futures = [
                executor.submit(worker, row_dict)
                for row_dict in rows_iter
            ]
            with tqdm(
                total=total_rows,
                desc=f"Processing {bucket_id}",
                unit="row",
            ) as progress:
                def _iter_results():
                    for future in as_completed(futures):
                        progress.update(1)
                        yield future.result()

                aggregated = _aggregate_by_user(
                    _iter_results(),
                    bucket_id=bucket_id,
                    user_col=args.user_col,
                )
                write_jsonl(bucket_output / "events_raw.jsonl", aggregated)
        bucket_count += 1

if __name__ == "__main__":
    main()
