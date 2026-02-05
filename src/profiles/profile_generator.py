import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass, field, fields, asdict

from utils.utils import load_jsonl_data, get_logger

@dataclass
class UserProfile:
    uuid: str
    professional_persona: str
    sports_persona: str
    arts_persona: str
    travel_persona: str
    culinary_persona: str
    persona: str
    cultural_background: str
    skills_and_expertise: str
    skills_and_expertise_list: list
    hobbies_and_interests: str
    hobbies_and_interests_list: list
    career_goals_and_ambitions: str
    sex: str
    age: int
    marital_status: str
    education_level: str
    bachelors_field: str
    occupation: str
    city: str
    state: str
    zipcode: str
    country: str
    life_events: list
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        data = {k.lower(): v for k, v in data.items()}

        field_names = {f.name for f in fields(cls) if f.name != "extra"}
        
        known = {name: data.get(name, None) for name in field_names}
        if known.get("age") is not None:
            try:
                known["age"] = int(known["age"])
            except (TypeError, ValueError):
                pass
        if known.get("life_events") is None:
            known["life_events"] = []
        extra = {k: v for k, v in data.items() if k not in field_names}
        
        return cls(**known, extra=extra)

    def to_dict(self) -> dict:
        base = asdict(self)
        base.update(self.extra)
        base.pop("extra", None)
        return base

    def desc(self, keys_to_drop: list = []):
        """
        Convert structured profile to a descriptive string.
        """
        base_info_parts = []

        attrs = [
            ("sex", "{}"),
            ("age", "{}"),
            ("marital_status", "{}"),
            ("education_level", "education level is {}"),
            ("bachelors_field", "studied {}"),
            ("occupation", "works as {}"),
            ("cultural_background", "cultural background: {}"),
            ("city", "lives in {}"),
            ("state", "{}"),
        ]

        for attr_name, fmt in attrs:
            if attr_name in keys_to_drop:
                continue
            value = getattr(self, attr_name, None)
            if value is None or value == "":
                continue
            base_info_parts.append(fmt.format(value))

        base_info = ", ".join(base_info_parts)

        persona_parts = []
        persona_attrs = [
            ("persona", "Overall persona: {}"),
            ("professional_persona", "Professional persona: {}"),
            ("sports_persona", "Sports persona: {}"),
            ("arts_persona", "Arts persona: {}"),
            ("travel_persona", "Travel persona: {}"),
            ("culinary_persona", "Culinary persona: {}"),
        ]
        for attr_name, fmt in persona_attrs:
            if attr_name in keys_to_drop:
                continue
            value = getattr(self, attr_name, None)
            if value:
                persona_parts.append(fmt.format(value))
        persona_text = " ".join(persona_parts)

        skills_text = ""
        if "skills_and_expertise" not in keys_to_drop and self.skills_and_expertise:
            skills_text = f"Skills and expertise: {self.skills_and_expertise}"

        hobbies_text = ""
        if "hobbies_and_interests" not in keys_to_drop and self.hobbies_and_interests:
            hobbies_text = f"Hobbies and interests: {self.hobbies_and_interests}"

        goals_text = ""
        if "career_goals_and_ambitions" not in keys_to_drop and self.career_goals_and_ambitions:
            goals_text = f"Career goals and ambitions: {self.career_goals_and_ambitions}"

        parts = [base_info, persona_text, skills_text, hobbies_text, goals_text]
        return "\n".join([p for p in parts if p])

    def __str__(self) -> str:
        return self.desc()

class UserProfileGenerator:
    def __init__(self, profiles_path: str, random_state: int = None, logger_silent: bool = False):
        self.logger = get_logger(__name__, silent=logger_silent)

        self.logger.info("Loading user pool...")
        self.profiles = load_jsonl_data(profiles_path)
        self.profiles = [UserProfile.from_dict(x) for x in self.profiles]
        self.logger.info("User pool loading finished.")
        if random_state is not None:
            random.seed(random_state)
            random.shuffle(self.profiles)
        
        self.id2profile = {x.uuid: x for x in self.profiles}

    def get_profile_str(self, n: int = -1):
        strs = []
        if n == -1:
            n = len(self.profiles)
        for x in self.profiles[:n]:
            strs.append({
                'uuid': x.uuid,
                'profile': x.to_dict(), 
                'profile_str': str(x)
            })
        
        return strs
    
    def calculate_weight(self, max_iter=10, weights_path='./weights.npy'):
        """
        Use the IPF algorithm to calculate the sampling weight for each sample, 
        ensuring that the sampled user attribute marginal distributions are uniform.
        """
        dims = set()
        for p in self.profiles:
            preferences_value = getattr(p, "preferences_value", []) or []
            for d in list(preferences_value):
                dims.add(list(d.keys())[0])
        dims = list(dims)
        if not dims:
            self.logger.warning("No preference dimensions found; skipping weight calculation.")
            return

        users = []
        for p in self.profiles:
            pv = {}
            preferences_value = getattr(p, "preferences_value", []) or []
            for x in preferences_value:
                pv.update(x)
            for d in dims:
                if d not in pv:
                    pv[d] = 'middle'
            users.append(pv)
        
        df = pd.DataFrame(users)

        target_marginals = {}
        for d in dims:
            target_marginals[d] = [1.0/3, 1.0/3, 1.0/3]
        
        weights = ipf_weights(
            df,
            target_marginals=target_marginals,
            max_iter=max_iter
        )

        np.save(weights_path, weights)
    
    def profile_filter(self, filter_keys: dict = None):
        if not filter_keys:
            return self.profiles, [i for i in range(len(self.profiles))]

        results = [
            (i, x) for i, x in enumerate(self.profiles)
            if any(getattr(x, k, None) != v for k, v in filter_keys.items())
        ]
        filtered_profiles = [x[1] for x in results]
        index = [x[0] for x in results]
        return filtered_profiles, index
    
    def profile_keep(self, keep_keys: dict = None):
        if not keep_keys:
            return self.profiles, [i for i in range(len(self.profiles))]

        results = [
            (i, x) for i, x in enumerate(self.profiles)
            if all(getattr(x, k, None) == v for k, v in keep_keys.items())
        ]
        filtered_profiles = [x[1] for x in results]
        index = [x[0] for x in results]
        return filtered_profiles, index

    def sample_profile_by_weights(self, n=1, weights_path=None, random_state=42, keep_keys: dict = None, filter_keys: dict = None):
        weights = np.load(weights_path)
        rng = np.random.default_rng(random_state)

        if filter_keys:
            filtered_profiles, index = self.profile_filter(filter_keys)
        elif keep_keys:
            filtered_profiles, index = self.profile_keep(keep_keys)
        else:
            filtered_profiles, index = self.profile_filter(filter_keys)
        weights = weights[index] / weights[index].sum()
        profiles = rng.choice(filtered_profiles, size=n, replace=False, p=weights)
        return profiles

    def get_profile(self, n=1):
        return self.profiles[:n]

    def get_profile_by_id(self, _id):
        return self.id2profile[_id]


def ipf_weights(df, target_marginals, max_iter=5, tol=1e-6):
    N = len(df)
    weights = np.ones(N) / N

    for _ in tqdm(range(max_iter)):
        old_weights = weights.copy() 

        for col in df.columns:
            current = df.groupby(col).apply(lambda g: weights[g.index].sum(), include_groups=False)
            current /= current.sum()

            target = np.array(target_marginals[col])

            categories = sorted(df[col].unique())
            ratio = target / (current + 1e-12)

            mapping = {cat: ratio.iloc[i] for i, cat in enumerate(categories)}
            weights *= df[col].map(mapping).values

        weights /= weights.sum()

        diff = np.abs(weights - old_weights).sum()
        if diff < tol:
            break

    return weights
