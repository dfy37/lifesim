import math
import random
from typing import Any, Dict, List, Optional

from utils.utils import get_logger, parse_json_dict_response

DESIRE_PROMPT = """你是用户意图生成助手。请基于用户画像、当前事件和用户已有的信念，生成用户可能的诉求列表。
要求：
- 输出 JSON，包含字段 "desires"，值为字符串数组。
- 诉求应与事件紧密相关，贴合用户画像。
- 只给出 3-6 条。

用户画像:
{profile}

当前事件:
{event}

用户信念:
{beliefs}
"""

REFINE_INTENTION_PROMPT = """请基于给定的用户意图候选，生成一个更清晰、具体且自然的意图表述。
要求：
- 仅输出 JSON，包含字段 "intention"。
- 保持语义不变，但更简洁清晰。

候选意图:
{intention}
"""


def generate_desires(model, profile: str, beliefs: List[Any], event: Dict[str, Any]) -> List[str]:
    prompt = DESIRE_PROMPT.format(
        profile=profile,
        event=event.get("life_event") or event.get("event", ""),
        beliefs=beliefs,
    )
    response = model.chat([{"role": "user", "content": prompt}])
    data = parse_json_dict_response(response, keys=["desires"])
    desires = data.get("desires", [])
    if not isinstance(desires, list):
        return []
    return [d for d in desires if isinstance(d, str) and d.strip()]


def intention_retrieval(retriever, desires: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for desire in desires:
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


def refine_intention(model, intention: str) -> str:
    if not intention:
        return ""
    prompt = REFINE_INTENTION_PROMPT.format(intention=intention)
    response = model.chat([{"role": "user", "content": prompt}])
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
        max_conv_turns: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        conv_history: List[Dict[str, Any]] = []
        for _ in range(max_events_number):
            event = self.life_event_engine.generate_event()
            beliefs = self.user_agent.get_beliefs()
            profile = self.user_agent.get_profile()

            desires = generate_desires(self.model, profile, beliefs, event)
            candidates = intention_retrieval(self.retriever, desires, top_k=self.max_retrievals)
            selected_intention = rerank_and_sample(candidates, seed=seed)
            refined_intention = refine_intention(self.model, selected_intention)

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
        return conv_history
