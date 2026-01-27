#!/usr/bin/env python3
"""Module C: Faithfulness audit and revision loop for extracted events."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import OpenAI
from json_repair import loads as repair_json

from event_extraction import Event, EventExtractor, _normalize_optional_str

VALIDATOR_PROMPT = """You are a Validator Agent.
Given a source tweet and a generated Event JSON, extract atomic facts from both.
Return ONLY valid JSON with the schema:
{
  "facts_source": ["..."],
  "facts_generated": ["..."],
  "feedback": "..."
}

Guidelines:
- Facts should be short atomic propositions (entities, actions, locations, timestamps).
- Use plain text strings; do not add reasoning.
- Feedback should point out missing or hallucinated facts.
"""


def _parse_validator_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = repair_json(text)
    if not isinstance(payload, dict):
        raise ValueError("Validator payload must be a JSON object.")
    return payload


def _normalize_fact(fact: str) -> str:
    return " ".join(fact.lower().strip().split())


def compute_delta(facts_source: list[str], facts_generated: list[str]) -> float:
    source_set = {_normalize_fact(fact) for fact in facts_source if fact}
    generated_set = {_normalize_fact(fact) for fact in facts_generated if fact}
    if not source_set and not generated_set:
        return 0.0
    if not source_set:
        return 0.6
    if not generated_set:
        return 0.4
    intersection = source_set & generated_set
    omission = 1 - (len(intersection) / len(source_set))
    hallucination = len(generated_set - source_set) / len(generated_set)
    alpha = 0.4
    beta = 0.6
    return alpha * omission + beta * hallucination


class EventValidator:
    def __init__(self, model: str, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def validate(self, tweet_text: str, event: Event) -> tuple[list[str], list[str], str]:
        payload = {
            "tweet": tweet_text,
            "event": asdict(event),
        }
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": VALIDATOR_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
        )
        result = _parse_validator_json(response.output_text)
        facts_source = result.get("facts_source") or []
        facts_generated = result.get("facts_generated") or []
        feedback = result.get("feedback") or ""
        return facts_source, facts_generated, feedback


def revision_loop(
    extractor: EventExtractor,
    validator: EventValidator,
    tweet_text: str,
    tweet_time: str | None,
    initial_event: Event,
    epsilon: float,
    max_rounds: int,
) -> tuple[Event, float, int, str]:
    event = initial_event
    for attempt in range(1, max_rounds + 1):
        facts_source, facts_generated, feedback = validator.validate(tweet_text, event)
        delta = compute_delta(facts_source, facts_generated)
        if delta <= epsilon:
            return event, delta, attempt, "passed"
        event = extractor.revise(tweet_text, tweet_time, feedback)
    facts_source, facts_generated, feedback = validator.validate(tweet_text, event)
    delta = compute_delta(facts_source, facts_generated)
    status = "flagged" if delta > epsilon else "passed"
    return event, delta, max_rounds, status


def iter_bucket_jsonl(root: Path) -> list[tuple[str, Path]]:
    buckets: list[tuple[str, Path]] = []
    for bucket_dir in sorted(root.glob("bucket_id=*")):
        jsonl_path = bucket_dir / "events_raw.jsonl"
        if jsonl_path.exists():
            buckets.append((bucket_dir.name, jsonl_path))
    return buckets


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generator-model", default="gpt-4o-mini")
    parser.add_argument("--validator-model", default="gpt-4o-mini")
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    extractor = EventExtractor(model=args.generator_model, api_key=args.api_key)
    validator = EventValidator(model=args.validator_model, api_key=args.api_key)

    args.output_root.mkdir(parents=True, exist_ok=True)

    for bucket_id, jsonl_path in iter_bucket_jsonl(args.input_root):
        records = read_jsonl(jsonl_path)
        checked_records: list[dict[str, Any]] = []
        for record in records:
            tweet_text = record.get("tweet_text", "")
            tweet_time = _normalize_optional_str(record.get("tweet_time"))
            event = Event.from_mapping(record.get("event", {}))
            checked_event, delta, attempts, status = revision_loop(
                extractor,
                validator,
                tweet_text=tweet_text,
                tweet_time=tweet_time,
                initial_event=event,
                epsilon=args.epsilon,
                max_rounds=args.max_rounds,
            )
            record["event_checked"] = asdict(checked_event)
            record["delta"] = delta
            record["attempts"] = attempts
            record["status"] = status
            checked_records.append(record)
        bucket_output = args.output_root / bucket_id
        bucket_output.mkdir(parents=True, exist_ok=True)
        write_jsonl(bucket_output / "events_checked.jsonl", checked_records)


if __name__ == "__main__":
    main()
