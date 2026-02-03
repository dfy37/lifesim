import json
import re
from typing import Any, Dict, List, Optional

from json_repair import loads as repair_json

from utils.utils import get_logger


DEFAULT_FAST_CONV_SYSTEM_PROMPT = """You are simulating a conversation between a user and an AI assistant.
The conversation should reflect the given event context and the user's intention.
Keep the dialogue realistic, concise, and grounded in the user's profile and beliefs.
Output only valid JSON as a list of turns in order."""


DEFAULT_FAST_CONV_PROMPT = """Profile:
{profile}

Beliefs:
{beliefs}

Event:
{event}

User intention:
{intention}

Requirements:
- Generate a multi-turn dialogue between the user and assistant.
- Each turn must be an object with fields: role ("user" or "assistant") and content (string).
- Keep the total number of turns within {max_turns}.
- Ensure the user's utterances align with the event and intention, and the assistant responds helpfully.
- Output a JSON array enclosed in ```json ...```."""


class FastConvSimulator:
    def __init__(
        self,
        model,
        max_turns: int = 6,
        logger_silent: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = system_prompt or DEFAULT_FAST_CONV_SYSTEM_PROMPT
        self.logger = get_logger(__name__, silent=logger_silent)

    def simulate(
        self,
        event: Dict[str, Any],
        intention: str,
        beliefs: List[Any],
        profile: str,
        max_turns: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        max_turns = max_turns or self.max_turns
        event_text = event.get("life_event") or event.get("event", "")
        prompt = DEFAULT_FAST_CONV_PROMPT.format(
            profile=profile,
            beliefs=json.dumps(beliefs, ensure_ascii=False),
            event=event_text,
            intention=intention,
            max_turns=max_turns,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.model.chat(messages)
        dialogue = self._parse_dialogue(response)
        if not dialogue:
            dialogue = self._fallback_dialogue(intention)
        return dialogue

    def _parse_dialogue(self, text: str) -> List[Dict[str, str]]:
        if not isinstance(text, str) or not text.strip():
            return []
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        json_str = match.group(1).strip() if match else text.strip()
        try:
            data = repair_json(json_str)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        cleaned = []
        for item in data:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"}:
                continue
            if not isinstance(content, str):
                continue
            cleaned.append({"role": role, "content": content})
        return cleaned

    @staticmethod
    def _fallback_dialogue(intention: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": intention or "我需要一些帮助。"},
            {"role": "assistant", "content": "当然可以，请告诉我更多细节，我会尽力帮你。"},
        ]
