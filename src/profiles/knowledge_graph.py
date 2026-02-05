from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from utils.utils import get_logger
from profiles.profile_generator import UserProfile


Triple = Tuple[str, str, str]


@dataclass
class KnowledgeGraph:
    """Lightweight knowledge graph representation with simple deduping."""

    triples: List[Triple] = field(default_factory=list)

    def add(self, triple: Triple) -> None:
        self.triples.append(triple)

    def extend(self, triples: Iterable[Triple]) -> None:
        self.triples.extend(triples)

    def dedupe(self) -> None:
        seen = set()
        deduped = []
        for head, relation, tail in self.triples:
            normalized = (
                normalize_entity(head),
                normalize_relation(relation),
                normalize_entity(tail),
            )
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append((head, relation, tail))
        self.triples = deduped

    def to_dict(self) -> List[dict]:
        return [
            {"head": head, "relation": relation, "tail": tail}
            for head, relation, tail in self.triples
        ]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class UserProfileKnowledgeGraphBuilder:
    """Build a simple knowledge graph from user profiles.

    The builder can optionally use an LLM to extract triples. If no LLM is
    provided, a light heuristic extraction is used.
    """

    def __init__(self, llm=None, logger_silent: bool = False):
        self.llm = llm
        self.logger = get_logger(__name__, silent=logger_silent)

    def build(self, profile: UserProfile) -> KnowledgeGraph:
        description = profile.desc()
        base_triples = self._profile_attribute_triples(profile)
        llm_triples = self._extract_triples(description)
        graph = KnowledgeGraph(base_triples + llm_triples)
        graph.dedupe()
        return graph

    def _profile_attribute_triples(self, profile: UserProfile) -> List[Triple]:
        triples = []
        subject = profile.uuid

        def add(attr: str, relation: str, value: Optional[object]) -> None:
            if value is None or value == "":
                return
            if isinstance(value, list):
                for item in value:
                    triples.append((subject, relation, str(item)))
            else:
                triples.append((subject, relation, str(value)))

        add("sex", "has_sex", profile.sex)
        add("age", "has_age", profile.age)
        add("marital_status", "has_marital_status", profile.marital_status)
        add("education_level", "has_education_level", profile.education_level)
        add("bachelors_field", "has_bachelors_field", profile.bachelors_field)
        add("occupation", "has_occupation", profile.occupation)
        add("city", "lives_in", profile.city)
        add("state", "lives_in_state", profile.state)
        add("zipcode", "has_zipcode", profile.zipcode)
        add("country", "lives_in_country", profile.country)
        add("cultural_background", "has_cultural_background", profile.cultural_background)
        add("persona", "has_persona", profile.persona)
        add("professional_persona", "has_professional_persona", profile.professional_persona)
        add("sports_persona", "has_sports_persona", profile.sports_persona)
        add("arts_persona", "has_arts_persona", profile.arts_persona)
        add("travel_persona", "has_travel_persona", profile.travel_persona)
        add("culinary_persona", "has_culinary_persona", profile.culinary_persona)
        add("skills_and_expertise", "has_skills_and_expertise", profile.skills_and_expertise)
        add("skills_and_expertise_list", "has_skill", profile.skills_and_expertise_list)
        add("hobbies_and_interests", "has_hobbies_and_interests", profile.hobbies_and_interests)
        add("hobbies_and_interests_list", "has_hobby", profile.hobbies_and_interests_list)
        add("career_goals_and_ambitions", "has_career_goals", profile.career_goals_and_ambitions)
        add("life_events", "has_life_event", profile.life_events)

        return triples

    def _extract_triples(self, text: str) -> List[Triple]:
        if not text:
            return []
        if self.llm is None:
            return heuristic_extract_triples(text)

        prompt = (
            "Extract user profile knowledge triples from the following text. "
            "Return JSON list of [head, relation, tail] triples. "
            "Use concise relations, avoid duplicates.\n\n"
            f"Text: {text}"
        )
        response = self.llm.chat([
            {"role": "user", "content": prompt}
        ])
        triples = parse_llm_triples(response)
        return triples


def normalize_entity(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_relation(text: str) -> str:
    return re.sub(r"\s+", "_", str(text).strip().lower())


def heuristic_extract_triples(text: str) -> List[Triple]:
    triples = []
    sentences = re.split(r"[。.!?]\s*", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        matches = re.findall(r"([\w\s'\-]+)\s+is\s+([\w\s'\-]+)", sentence, re.IGNORECASE)
        for head, tail in matches:
            triples.append((head.strip(), "is", tail.strip()))
        matches = re.findall(r"([\w\s'\-]+)\s+works as\s+([\w\s'\-]+)", sentence, re.IGNORECASE)
        for head, tail in matches:
            triples.append((head.strip(), "occupation", tail.strip()))
        matches = re.findall(r"lives in\s+([\w\s'\-]+)", sentence, re.IGNORECASE)
        for tail in matches:
            triples.append(("user", "lives_in", tail.strip()))
    return triples


def parse_llm_triples(response: str) -> List[Triple]:
    if not response:
        return []
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return []

    triples = []
    if isinstance(data, list):
        for item in data:
            if (
                isinstance(item, Sequence)
                and len(item) == 3
                and all(isinstance(x, (str, int, float)) for x in item)
            ):
                head, relation, tail = item
                triples.append((str(head), str(relation), str(tail)))
    return triples
