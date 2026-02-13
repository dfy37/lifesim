import math
import random
from typing import Any, Dict, List, Optional

from utils.utils import get_logger, parse_json_dict_response

DESIRE_PROMPT = """You are a user intention generation assistant. Based on the user profile, the current event, and the user's existing beliefs, generate a list of possible user desires.

Requirements:
- Output JSON containing the field "desires", whose value is an array of strings.
- The desires should be closely related to the event and aligned with the user profile.
- Provide only 3–6 items.

User Profile:
{profile}

Current Event:
{event}

User Beliefs:
{beliefs}
"""

REFINE_INTENTION_PROMPT = '''You will be given one candidate user intention.
Your task is to revise and refine it so that align with the user’s profile, and current environmental context.
### Requirements
- Adjust details such as subject, location, weather, time, or other contextual factors to make the event realistic and coherent with the given user profile and prior events.
- The intent should remain essentially the same in meaning but must be expressed naturally and fit the updated event context.
- The intent should represent a single conversational goal (i.e., the user’s focus within one dialogue turn), not a long-term plan.
- Remove any placeholders or meaningless symbols (e.g., "NAME_1", "XXX", "...").
### Output Format:
Please output your final answer strictly in the following JSON structure (enclosed within ```json and ```):
{{
    "intention": "Describe the user’s corresponding intent under this event context."
}}
Provide your reasoning before the final answer. 
In your reasoning, consider: (1) whether the event and intent satisfy the requirements; (2) whether the intent is realistically something a human would ask an AI assistant.
### Examples:
If the intention is “The user feels the sun is strong and wants the assistant to give hydration advice,” but the weather is cloudy, revise it to “The user has exercised for a long time and sweated a lot, wants the assistant to give hydration advice.”

### Input
[User Profile]
{user_profile}
[Current Environment]
{env}
[Current Event and Intention]
Current user intention: {intention}
[Output]
'''


def generate_desires(model, profile: str, beliefs: List[Any], event: Dict[str, Any], logger=None) -> List[str]:
    prompt = DESIRE_PROMPT.format(
        profile=profile,
        event=event.get("life_event") or event.get("event", ""),
        beliefs=beliefs,
    )
    response = model.chat([{"role": "user", "content": prompt}])
    if logger:
        logger.info("Generated desire candidates from model response")
    data = parse_json_dict_response(response, keys=["desires"])
    desires = data.get("desires", [])
    if not isinstance(desires, list):
        return []
    return [d for d in desires if isinstance(d, str) and d.strip()]


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


def rerank_and_sample(candidates: List[Dict[str, Any]], seed: Optional[int] = None) -> str:
    if not candidates:
        return ""
    scores = []
    for item in candidates:
        score = item.get("score")
        scores.append(score if isinstance(score, (int, float)) else 0.0)
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores) or 1.0
    probabilities = [s / total for s in exp_scores]
    rng = random.Random(seed)
    chosen = rng.choices(candidates, weights=probabilities, k=1)[0]
    return chosen.get("intent", "")


def refine_intention(model, intention: str, logger=None) -> str:
    if not intention:
        return ""
    prompt = REFINE_INTENTION_PROMPT.format(intention=intention)
    response = model.chat([{"role": "user", "content": prompt}])
    if logger:
        logger.info("Refining selected intention")
    data = parse_json_dict_response(response, keys=["intention"])
    refined = data.get("intention")
    return refined.strip() if isinstance(refined, str) else intention.strip()

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
            self.logger.info("Generated %s desires", len(desires))
            candidates = intention_retrieval(self.retriever, desires, top_k=self.max_retrievals, logger=self.logger)
            self.logger.info("Retrieved %s intention candidates", len(candidates))
            selected_intention = rerank_and_sample(candidates, seed=seed)
            refined_intention = refine_intention(self.model, selected_intention, logger=self.logger)
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
