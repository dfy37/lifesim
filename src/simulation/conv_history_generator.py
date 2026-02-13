import math
import random
from typing import Any, Dict, List, Optional

from utils.utils import get_logger, parse_json_dict_response

DESIRE_QUERY_PROMPTS = {
    "environmental": """You are generating user intention retrieval queries.
Based on user profile, current event, and existing beliefs, generate ONE short retrieval query from the environmental influence dimension
(e.g., weather, time, location, surrounding facilities, social scene).

Output JSON:
```json
{{
  "query": "..."
}}
```

[User Profile]
{profile}
[Current Event]
{event}
[User Beliefs]
{beliefs}
[Output]
""",
    "physiological": """You are generating user intention retrieval queries.
Based on user profile, current event, and existing beliefs, generate ONE short retrieval query from the physiological influence dimension
(e.g., fatigue, hunger, thirst, pain, sleep, exercise recovery, body status).

Output JSON:
```json
{{
  "query": "..."
}}
```

[User Profile]
{profile}
[Current Event]
{event}
[User Beliefs]
{beliefs}
[Output]
""",
    "psychological": """You are generating user intention retrieval queries.
Based on user profile, current event, and existing beliefs, generate ONE short retrieval query from the psychological influence dimension
(e.g., mood, stress, motivation, worry, confidence, social emotion, preference).

Output JSON:
```json
{{
  "query": "..."
}}
```

[User Profile]
{profile}
[Current Event]
{event}
[User Beliefs]
{beliefs}
[Output]
""",
}

RERANK_INTENTION_PROMPT = """### Requirements
You will be given candidate user intentions retrieved from a memory bank.
Based on user profile, current event, and user beliefs, rank intentions from most likely to least likely for the next dialogue turn.

Return JSON between ```json and ```:
```json
{{
  "ranked_intentions": [x, x, x],
  "has_possible_intention": "true/false"
}}
```

Rules:
- ranked_intentions uses 1-based indices from candidate list.
- Exclude impossible, irrelevant, duplicated, or contradictory intentions.
- If none is plausible, set has_possible_intention to false.
- Think about context coherence, realistic human askability, and immediate conversational goal.

[User Profile]
{profile}
[Current Event]
{event}
[User Beliefs]
{beliefs}
[Candidate Intentions]
{candidates_text}
[Output]
"""

REFINE_INTENTION_PROMPT = '''You will be given one candidate user intention.
Your task is to revise and refine it so that it aligns with the user profile, current event, and beliefs.
### Requirements
- Keep the original intention meaning, but adapt details (subject/location/weather/time/body status/emotional state) to fit context.
- Ensure the intention is realistic for the current event and does not contradict known beliefs.
- The intention should be one short, concrete conversational goal for a single dialogue turn.
- Remove placeholders or meaningless symbols (e.g., "NAME_1", "XXX", "...").
### Output Format:
Please output your final answer strictly in the following JSON structure (enclosed within ```json and ```):
{{
    "intention": "Describe the user’s corresponding intent under this event context."
}}
You may provide reasoning before the final JSON.

### Input
[User Profile]
{user_profile}
[Current Event]
{event}
[User Beliefs]
{beliefs}
[Current Intention]
{intention}
[Output]
'''


def _event_text(event: Dict[str, Any]) -> str:
    return str(
        event.get("life_event")
        or event.get("event")
        or event.get("action")
        or ""
    )


def generate_query_by_dimension(
    model,
    profile: str,
    beliefs: List[Any],
    event: Dict[str, Any],
    dimension: str,
    logger=None,
) -> str:
    dimension_prompt = DESIRE_QUERY_PROMPTS.get(dimension)
    if not dimension_prompt:
        return ""

    prompt = dimension_prompt.format(
        profile=profile,
        event=_event_text(event),
        beliefs=beliefs,
    )
    response = model.chat([{"role": "user", "content": prompt}])
    data = parse_json_dict_response(response, keys=["query"])
    query = data.get("query", "")
    query = query.strip() if isinstance(query, str) else ""
    if logger and query:
        logger.info("Generated %s-dimension query: %s", dimension, query)
    return query


def generate_desires(model, profile: str, beliefs: List[Any], event: Dict[str, Any], logger=None) -> List[str]:
    dimensions = ["environmental", "physiological", "psychological"]
    desires: List[str] = []

    for dimension in dimensions:
        for _ in range(3):
            query = generate_query_by_dimension(
                model=model,
                profile=profile,
                beliefs=beliefs,
                event=event,
                dimension=dimension,
                logger=logger,
            )
            if query:
                desires.append(query)

    return desires


def intention_retrieval(retriever, desires: List[str], top_k: int = 5, logger=None) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for desire in desires:
        if logger:
            logger.info("Retrieving intentions for desire: %s", desire)
        results = retriever.search(query=desire, top_k=top_k)
        for res in results:
            item = res.get("data") if isinstance(res, dict) else None
            score = res.get("score") if isinstance(res, dict) else None
            if item is None:
                continue
            candidates.append({"intent": item, "score": score})
    return candidates


def rerank_intentions(
    model,
    candidates: List[Dict[str, Any]],
    profile: str,
    beliefs: List[Any],
    event: Dict[str, Any],
    n_keep: int = 5,
    logger=None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    candidates_text = "\n".join(
        [f"({i + 1}) {candidate.get('intent', '')}" for i, candidate in enumerate(candidates)]
    )
    prompt = RERANK_INTENTION_PROMPT.format(
        profile=profile,
        event=_event_text(event),
        beliefs=beliefs,
        candidates_text=candidates_text,
    )

    response = model.chat([{"role": "user", "content": prompt}])
    parsed = parse_json_dict_response(response, keys=["ranked_intentions", "has_possible_intention"])
    if logger:
        logger.info("Rerank intentions response: %s", parsed)

    has_possible = parsed.get("has_possible_intention", "false")
    if isinstance(has_possible, str):
        has_possible = has_possible.lower() == "true"
    elif not isinstance(has_possible, bool):
        has_possible = False
    if not has_possible:
        return []

    indices = parsed.get("ranked_intentions", [])
    ranked_candidates: List[Dict[str, Any]] = []
    for idx in indices:
        try:
            candidate_idx = int(idx) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= candidate_idx < len(candidates):
            ranked_candidates.append(candidates[candidate_idx])

    if not ranked_candidates:
        return []
    return ranked_candidates[:n_keep]


def softmax_sampling_by_rank(candidates: List[Dict[str, Any]], seed: Optional[int] = None) -> str:
    if not candidates:
        return ""
    rank_scores = [-i for i in range(1, len(candidates) + 1)]
    exp_scores = [math.exp(score) for score in rank_scores]
    total = sum(exp_scores) or 1.0
    probabilities = [s / total for s in exp_scores]
    rng = random.Random(seed)
    selected = rng.choices(candidates, weights=probabilities, k=1)[0]
    return selected.get("intent", "")


def refine_intention(model, intention: str, profile: str, beliefs: List[Any], event: Dict[str, Any], logger=None) -> str:
    if not intention:
        return ""

    prompt = REFINE_INTENTION_PROMPT.format(
        user_profile=profile,
        event=_event_text(event),
        beliefs=beliefs,
        intention=intention,
    )
    response = model.chat([{"role": "user", "content": prompt}])
    if logger:
        logger.info("Refining selected intention")
    data = parse_json_dict_response(response, keys=["intention"])
    refined = data.get("intention")
    return refined.strip() if isinstance(refined, str) and refined.strip() else intention.strip()


class ConvHistoryGenerator:
    def __init__(
        self,
        life_event_engine,
        user_agent,
        fast_conv_simulator,
        model,
        retriever,
        max_retrievals: int = 5,
        logger_silent: bool = False,
    ) -> None:
        self.life_event_engine = life_event_engine
        self.user_agent = user_agent
        self.fast_conv_simulator = fast_conv_simulator
        self.model = model
        self.retriever = retriever
        self.max_retrievals = max_retrievals
        self.logger = get_logger(__name__, silent=logger_silent)

    def generate(
        self,
        max_events_number: int,
        max_conv_turns: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        conv_history: List[Dict[str, Any]] = []
        self.logger.info("Start conversation history generation: max_events=%s", max_events_number)
        for event_idx in range(max_events_number):
            if hasattr(self.life_event_engine, "has_next_event") and not self.life_event_engine.has_next_event():
                break
            event = self.life_event_engine.generate_event()
            self.logger.info("Processing event index %s", event_idx + 1)
            if not event:
                break
            beliefs = self.user_agent.get_beliefs()
            profile = self.user_agent.get_profile()

            desires = generate_desires(self.model, profile, beliefs, event, logger=self.logger)
            self.logger.info("Generated %s desire queries", len(desires))
            candidates = intention_retrieval(self.retriever, desires, top_k=self.max_retrievals, logger=self.logger)
            self.logger.info("Retrieved %s intention candidates", len(candidates))

            reranked_candidates = rerank_intentions(
                model=self.model,
                candidates=candidates,
                profile=profile,
                beliefs=beliefs,
                event=event,
                n_keep=self.max_retrievals,
                logger=self.logger,
            )
            selected_intention = softmax_sampling_by_rank(reranked_candidates, seed=seed)
            if not selected_intention and candidates:
                selected_intention = candidates[0].get("intent", "")

            refined_intention = refine_intention(
                self.model,
                selected_intention,
                profile=profile,
                beliefs=beliefs,
                event=event,
                logger=self.logger,
            )
            self.logger.info("Final refined intention: %s", refined_intention)

            dialogue = self.fast_conv_simulator.simulate(
                event=event,
                intention=refined_intention,
                beliefs=beliefs,
                profile=profile,
                max_turns=max_conv_turns,
            )

            conv_history.append(
                {
                    "event": event,
                    "intention": refined_intention,
                    "dialogue": dialogue,
                }
            )

            self.user_agent.update_belief_from_event(event)
            self.logger.info("Finished event index %s", event_idx + 1)
        self.logger.info("Conversation history generation finished: total=%s", len(conv_history))
        return conv_history
