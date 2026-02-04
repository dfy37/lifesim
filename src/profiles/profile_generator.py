import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass, field, fields, asdict

from utils.utils import load_jsonl_data, get_logger

@dataclass
class UserProfile:
    user_id: str
    religious: str
    employment: str
    marital: str
    race: str
    income: str
    area: str
    age: str
    gender: str
    bigfive: dict
    personality: list
    preferences: list
    preferences_value: dict
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        data = {k.lower(): v for k, v in data.items()}

        field_names = {f.name for f in fields(cls) if f.name != "extra"}
        
        known = {name: data.get(name, None) for name in field_names}
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
            ("gender", "{}"),
            ("age", "{}"),
            ("race", "{}"),
            ("marital", "{}"),
            ("religious", "user's religion is {}"),
            ("area", "usually resides in {}"),
        ]

        for attr_name, fmt in attrs:
            if attr_name in keys_to_drop:
                continue
            value = getattr(self, attr_name, None)
            if not value:
                continue
            if attr_name == "religious":
                if value == "No religion":
                    base_info_parts.append("no religious affiliation")
                else:
                    base_info_parts.append(fmt.format(value))
            else:
                base_info_parts.append(fmt.format(value))

        base_info = ", ".join(base_info_parts)

        politics_econ_parts = []
        if "income" not in keys_to_drop and self.income:
            politics_econ_parts.append(f"the income level is {self.income}")
        if "employment" not in keys_to_drop and self.employment:
            politics_econ_parts.append(f"{self.employment}")
        politics_econ_text = ", ".join(politics_econ_parts)

        personality_text = ""
        if "personality" not in keys_to_drop and self.personality:
            personality_text = "Personality traits include: " + "、".join(self.personality)

        preferences_text = ""
        if "preferences" not in keys_to_drop and self.preferences:
            preferences_text = "Preferences expressed in daily life and interactions include: " + " ".join(self.preferences)

        parts = [base_info, politics_econ_text, personality_text, preferences_text]
        return ". ".join([p for p in parts if p])

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
        
        self.id2profile = {x.user_id: x for x in self.profiles}

    def get_profile_str(self, n: int = -1):
        strs = []
        if n == -1:
            n = len(self.profiles)
        for x in self.profiles[:n]:
            strs.append({
                'user_id': x.user_id,
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
            for d in list(p.preferences_value):
                dims.add(list(d.keys())[0])
        dims = list(dims)

        users = []
        for p in self.profiles:
            pv = {}
            for x in p.preferences_value:
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
