import random
import numpy as np
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Any
import math
from dataclasses import dataclass, field, fields, asdict
from tqdm.auto import tqdm

from engine.prompts import get_event_dimensions, get_infer_goal_prompt, RERANK_PROMPT, REWRITE_PROMPT
from utils.utils import get_logger, parse_json_dict_response, load_jsonl_data

from openai import OpenAI

class DeepSeek:
    def __init__(self, api_key, base_url=None):
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.deepseek.com"
        )
    
    def chat(self, messages, n=1):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            n=n
        )
        return response.choices[0].message.content

class OfflineLifeEventEngine:
    def __init__(self, life_events) -> None:
        self.main_events = life_events
        self.event_index = 0
        self.logger = get_logger(__name__, silent=False)
        self.logger.info("OfflineLifeEventEngine initialized with %s events", self.total_events())

    def _get_event_list(self) -> list:
        if not self.main_events:
            return []
        if isinstance(self.main_events, dict):
            return self.main_events.get('events', [])
        if isinstance(self.main_events, list):
            return self.main_events
        return []

    def total_events(self) -> int:
        return len(self._get_event_list())

    def remaining_events(self) -> int:
        return max(self.total_events() - self.event_index, 0)

    def has_next_event(self) -> bool:
        has_next = self.remaining_events() > 0
        if not has_next:
            self.logger.info("No more events to generate.")
        return has_next

    def generate_event(self):
        if not self.has_next_event():
            return None
        event = self._get_event_list()[self.event_index]
        formatted_event = LifeEvent.from_dict(event, timezone=None)
        self.logger.info("Generating event %s/%s", self.event_index + 1, self.total_events())
        self.event_index += 1
        return formatted_event

class Environment:
    def __init__(self, map) -> None:
        self.map = map
        self.environment = {}
    
    def get_poi_cate(self):
        """
        - **Description**:
            - Returns a list of categories of points of interest (POI) in the map.
        
        - **Returns**:
            - A list of POI categories.
        """
        return self.map.get_poi_cate()

@dataclass
class POI_Event:
    time: str = ""
    location: str = ""
    event: str = ""
    weather: dict = ""
    life_event: str = ""
    intent: str = ""
    extra: dict = field(default_factory=dict)

    @staticmethod
    def convert_utc_to_target_zone(time_str, timezone: str = "America/New_York"):
        dt_utc = datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
        dt_target = dt_utc.astimezone(ZoneInfo(timezone))
        return dt_target.strftime("%Y-%m-%d %H:%M:%S, %A")

    @classmethod
    def from_dict(cls, data, timezone: str = None):
        standard_keys = {f.name for f in fields(cls) if f.name != "extra"}

        known = {name: data.get(name, None) for name in standard_keys}
        if known['time']:
            known['time'] = cls.convert_utc_to_target_zone(known["time"], timezone) if timezone else known["time"]
        extras = {k: v for k, v in data.items() if k not in standard_keys}
        return cls(
            **known,
            extra=extras
        )
    
    def to_dict(self):
        base = asdict(self)
        base.update(self.extra)
        base.pop("extra", None)
        return base

    def desc_time(self):
        template = "The time is {time}"
        time_str = template.format(
            time=self.time
        )
        return time_str
    
    def desc_location(self):
        template = "The location is {location}"
        location_str = template.format(
            location=self.location
        )
        return location_str

    def desc_event(self):
        template = "The scenario theme is {event}"
        event_str = template.format(
            event=self.event
        )
        return event_str

    def desc_weather(self):
        template = "The weather condition is {description}"
        weather_str = template.format(
            description=self.weather['description']
        )
        return weather_str
    
    def desc_life_event(self):
        template = "The event experienced by the user: {life_event}"
        life_event_str = template.format(
            life_event=self.life_event
        )
        return life_event_str
    
    def desc_intent(self):
        template = "The user's intent: {intent}"
        intent_str = template.format(
            intent=self.intent
        )
        return intent_str

    def desc(self, sep='\n', keys_to_drop: list = None):
        key2fun = {
            'time': self.desc_time,
            'location': self.desc_location,
            # 'event': self.desc_event,
            'weather': self.desc_weather,
            'life_event': self.desc_life_event,
            'intent': self.desc_intent,
        }
        if keys_to_drop:
            infos = [fun() for key, fun in key2fun.items() if key not in keys_to_drop]
        else:
            infos = [fun() for key, fun in key2fun.items()]
        return sep.join(infos)


@dataclass
class LifeEvent:
    time: str = ""
    location: str | None = None
    environment: str | None = None
    action: str = ""
    observation: str = ""
    inner_thought: str = ""
    extra: dict = field(default_factory=dict)

    @staticmethod
    def convert_utc_to_target_zone(time_str, timezone: str = "America/New_York"):
        return POI_Event.convert_utc_to_target_zone(time_str, timezone)

    @classmethod
    def from_dict(cls, data, timezone: str = None):
        standard_keys = {f.name for f in fields(cls) if f.name != "extra"}
        known = {name: data.get(name, None) for name in standard_keys}
        known["environment"] = known.get("environment") or data.get("weather") or ""
        known["action"] = known.get("action") or data.get("life_event") or data.get("event") or ""
        known["observation"] = known.get("observation") or data.get("intent") or ""
        known["inner_thought"] = known.get("inner_thought") or ""

        if known['time']:
            known['time'] = cls.convert_utc_to_target_zone(known["time"], timezone) if timezone else known["time"]

        extras = {k: v for k, v in data.items() if k not in standard_keys}
        return cls(**known, extra=extras)

    def to_dict(self):
        base = asdict(self)
        base.update(self.extra)
        base.pop("extra", None)
        return base

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def __getitem__(self, key):
        return self.to_dict()[key]

class OnlineLifeEventEngine:
    def __init__(self, event_sequences_path, model=None, retriever=None):
        self.events = load_jsonl_data(event_sequences_path)
        self.uid2events = {event['user_id']: event for event in self.events}
        self.id2events = {event['id']: event for event in self.events}
        self.model = model
        self.retriever = retriever
        self.logger = get_logger(__name__, silent=False)
    
    def set_event_sequence(self, sequence_id: str):
        self.event_index = 0
        self.main_events = self.id2events.get(sequence_id, None)
        self.user_id = self.main_events['user_id'] if self.main_events else None
        self.theme = self.main_events['theme'] if self.main_events else None
        self.longterm_goal = self.main_events.get('longterm_goal', '') if self.main_events else ''
        self.sequence_id = sequence_id
        self.event_dimensions = get_event_dimensions(self.theme)

    def set_user(self, user_id: str):
        """
        Set user profile
        
        - **Description**:
            - Sets the user profile for the event generation.
        
        - **Args**:
            - user_profile: A dictionary containing user profile information.
        """
        self.user_id = user_id
        self.event_index = 0
        self.main_events = self.uid2events.get(user_id, [])
    
    def set_event_index(self, index: int):
        self.event_index = index

    def get_current_user_id(self):
        return self.user_id
    
    def get_current_sequence_info(self):
        return {
            'user_id': self.user_id,
            'sequence_id': self.sequence_id,
            'theme': self.theme,
            'longterm_goal': self.longterm_goal
        }

    def generate_environment(self):
        event = self.main_events['events'][self.event_index]
        formatted_event = POI_Event.from_dict(event, timezone=None)
        event['dialogue_scene'] = '\n'.join([formatted_event.desc_time(), formatted_event.desc_location(), formatted_event.desc_weather()])
        self.event_index += 1
        return event

    def get_event_context(self, event_results):
        if not event_results:
            return "None"
        
        event_texts = []
        for i, res in enumerate(event_results):
            if 'selected_event' in res and 'trajectory_point' in res:
                event = res['selected_event']['event']
                intent = res['selected_event'].get('intent', '')
                time = res['trajectory_point'].get('time')
                location = res['trajectory_point'].get('location')
                weather = res['trajectory_point'].get('weather')
            else:
                event = res.get('life_event') or res.get('event')
                intent = res.get('intent', '')
                time = res.get('time')
                location = res.get('location')
                weather = res.get('weather')
            formatted_event = POI_Event.from_dict({
                'time': time,
                'location': location,
                'life_event': event,
                'intent': intent,
                'weather': weather
            }, timezone=None)
            event_desc = f"({i+1}) {formatted_event.desc(keys_to_drop=['event'])}"
            event_texts.append(event_desc)
        
        return "\n".join(event_texts)
    
    def generate_query_by_dimension(self, user_profile: str, event_context: str, location_desc: str, dimension: str, goal: str) -> str:
        dimension_prompt = self.event_dimensions.get(dimension, None)
        prompt = dimension_prompt.format(
            user_profile=user_profile,
            location_desc=location_desc,
            event_sequences=event_context,
            goal=goal
        )
        response = self.model.chat([{'role': 'user', 'content': prompt}])
        response = parse_json_dict_response(response, keys=['event']).get('event', None)
        self.logger.info(f"Generated query for dimension {dimension}: {response}")
        return response

    def retrieve_similar_events(self, query: str, top_k: int = 3) -> List[Dict]:
        if not query:
            return []
        similar_events = self.retriever.search(query=query, top_k=top_k)
        similar_events = [e['data'] for e in similar_events]
        return similar_events
    
    def rerank_events(self, events: List[Dict], user_profile: str,
                     location_desc: str, event_context: str, goal: str, n_keep: int = 3) -> List[Dict]:
        events_text = "\n".join([f"({i+1}) {event['event']}" for i, event in enumerate(events)])

        prompt = RERANK_PROMPT.format(
            user_profile=user_profile,
            location_desc=location_desc,
            event_sequences=event_context,
            events_text=events_text,
            goal=goal
        )

        response = self.model.chat([{'role': 'user', 'content': prompt}])
        response = parse_json_dict_response(response, keys=['ranked_events', 'has_possible_event'])
        self.logger.info(f"Rerank response: {response}")
        try:
            has_possible_event = response.get('has_possible_event', 'false')
            if isinstance(has_possible_event, str):
                has_possible_event = has_possible_event.lower() == 'true'
            elif not isinstance(has_possible_event, bool):
                has_possible_event = False
            if not has_possible_event:
                return []
            rank_indices = response.get('ranked_events', [])
            rank_indices = [int(i) - 1 for i in rank_indices]
            reranked_events = [events[i] for i in rank_indices if 0 <= i < len(events)]
            return reranked_events[:n_keep]
        except:
            return []

    def softmax_sampling(self, events: List[Dict]) -> Dict:
        if not events:
            return None

        ranks = np.arange(1, len(events) + 1)
        inverse_ranks = - ranks

        probabilities = np.exp(inverse_ranks) / np.sum(np.exp(inverse_ranks))

        selected_idx = np.random.choice(len(events), p=probabilities)
        return events[selected_idx]

    def rewrite_event(self, user_profile: str, location_desc: str, event_context: str, selected_event: dict, goal: str):
        prompt = REWRITE_PROMPT.format(
            user_profile=user_profile,
            location_desc=location_desc,
            event_sequences=event_context,
            event_text=selected_event['event'],
            intent=selected_event.get('intent', ''),
            goal=goal
        )

        response = self.model.chat([{'role': 'user', 'content': prompt}])
        response = parse_json_dict_response(response, keys=['event', 'intent'])
        self.logger.info(f"Rewritten event: {response.get('event', '')}")
        self.logger.info(f"Rewritten intent: {response.get('intent', '')}")

        if response.get('event', None):
            selected_event['event'] = response['event']
        if response.get('intent', None):
            selected_event['intent'] = response['intent']

        return selected_event

    def generate_event(self, user_profile: str = None, history_events: List[Dict] = None, goal: str = ''):
        event = self.main_events['events'][self.event_index]
        formatted_event = POI_Event.from_dict(event, timezone=None)
        event['dialogue_scene'] = '\n'.join([formatted_event.desc_time(), formatted_event.desc_location(), formatted_event.desc_weather()])
        self.event_index += 1

        if not self.model or not self.retriever or not user_profile:
            if not event.get('life_event'):
                event['life_event'] = event.get('event')
            return event

        event_context = self.get_event_context(history_events or [])
        location_desc = formatted_event.desc(keys_to_drop=['life_event', 'intent'])
        goal = goal or self.longterm_goal

        dimensions = list(self.event_dimensions.keys())
        all_candidate_events = []
        for dimension in dimensions:
            self.logger.info(f"Generating query for dimension: {dimension}")
            query = self.generate_query_by_dimension(
                user_profile,
                event_context,
                location_desc,
                dimension,
                goal=goal
            )

            similar_events = self.retrieve_similar_events(query, top_k=3)
            all_candidate_events.extend(similar_events)

        self.logger.info(f"Total candidate events: {len(all_candidate_events)}")
        reranked_events = self.rerank_events(
            all_candidate_events,
            user_profile=user_profile,
            location_desc=location_desc,
            event_context=event_context,
            goal=goal,
            n_keep=3
        )

        selected_event = self.softmax_sampling(reranked_events)
        if selected_event:
            self.logger.info(f"Selected event: {selected_event.get('event', '')}")  
            selected_event = self.rewrite_event(
                user_profile=user_profile,
                location_desc=location_desc,
                event_context=event_context,
                selected_event=selected_event,
                goal=goal
            )

            event['event'] = selected_event.get('event', event.get('event'))
            event['life_event'] = selected_event.get('event', event.get('life_event'))
            event['intent'] = selected_event.get('intent', event.get('intent'))
            event['sub_intents'] = selected_event.get('sub_intents', event.get('sub_intents', []))
        else:
            if not event.get('life_event'):
                event['life_event'] = event.get('event')

        return event

class TrajectoryEventMatcher:
    def __init__(self, event_database: List[Dict], retriever, model, theme: str, theme_tags: List[str], logger_silent: bool = False):
        """
        Initialization
        
        Args:
            event_database: irregular event base
            retriever: external retriever, used for vector search
            model: LLMs, used for generating queries and reranking
            theme_tags: ist of theme-related event tags
        """
        self.event_database = event_database
        self.retriever = retriever
        self.model = model
        self.theme = theme
        self.theme_tags = theme_tags
        self.event_dimensions = get_event_dimensions(theme)
        self.logger = get_logger(__name__, silent=logger_silent)
        
    def calculate_event_probability(self, current_time: float, start_time: float, 
                                  is_theme_location: bool, base_prob: float = 0.1, t0: int = 720) -> float:
        """
        Calculate the event occurrence probability
        
        Args:
            current_time: current timestamp
            start_time: trajectory start timestamp 
            is_theme_location: whether it is a theme-related location
            base_prob: base probability
            
        Returns:
            event occurrence probability
        """
        hours_passed = (current_time - start_time) / 3600
                  
        k = 5e-3                            
        logistic_prob = 1 / (1 + math.exp(-k * (hours_passed - t0)))
        time_prob = base_prob + logistic_prob * (1 - base_prob)

        if is_theme_location:
            time_prob *= 2
        
        return min(time_prob, 1.0)
    
    def is_theme_location(self, location: str) -> bool:
        """
        Determine whether it is a theme-related location
        
        Args:
            location_event: Location label
            
        Returns:
            Whether it is a theme-related location
        """
        return location in self.theme_tags
    
    def generate_query_by_dimension(self, user_profile: str, event_context: str, location_desc: str, dimension: str, goal: str) -> str:
        """
        Generate a query based on dimensions
        
        Args:
            user_profile: User profile
            location_desc: Detailed description of the location
            dimension: Dimension type (environment-driven, physiological-driven, cognitive-feedback-driven)
            
        Returns:
            Generated query
        """
        dimension_prompt = self.event_dimensions.get(dimension, None)
        prompt = dimension_prompt.format(
            user_profile=user_profile,
            location_desc=location_desc,
            event_sequences=event_context,
            goal=goal
        )
        response = self.model.chat([{'role': 'user', 'content': prompt}])
        response = parse_json_dict_response(response, keys=['event']).get('event', None)
        return response
    
    def retrieve_similar_events(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve similar events
        
        Args:
            query: Query text
            top_k: Return top-k similar events
            
        Returns:
            List of similar events
        """
        if not query:
            return []
        similar_events = self.retriever.search(query=query, top_k=top_k)
        similar_events = [e['data'] for e in similar_events]
        return similar_events
    
    def rerank_events(self, events: List[Dict], user_profile: str, 
                     location_desc: str, event_context: str, goal: str, n_keep: int = 3) -> List[Dict]:
        """
        Re-rank events using a large model
        
        Args:
            events: Candidate event list
            user_profile: User profile
            location_desc: Detailed description of the event
            n_keep: Number of events to keep after sampling

        Returns:
            Re-ranked event list
        """
        events_text = "\n".join([f"({i+1}) {event['event']}" for i, event in enumerate(events)])
        
        prompt = RERANK_PROMPT.format(
            user_profile=user_profile,
            location_desc=location_desc,
            event_sequences=event_context,
            events_text=events_text,
            goal=goal
        )
        
        response = self.model.chat([{'role': 'user', 'content': prompt}])
        response = parse_json_dict_response(response, keys=['ranked_events', 'has_possible_event'])
        try:
            has_possible_event = response.get('has_possible_event', 'false')
            if isinstance(has_possible_event, str):
                has_possible_event = has_possible_event.lower() == 'true'
            elif not isinstance(has_possible_event, bool):
                has_possible_event = False
            if not has_possible_event:
                return []
            rank_indices = response.get('ranked_events', [])
            rank_indices = [int(i) - 1 for i in rank_indices]
            reranked_events = [events[i] for i in rank_indices if 0 <= i < len(events)]
            return reranked_events[:n_keep]
        except:
            return []
    
    def softmax_sampling(self, events: List[Dict]) -> Dict:
        """
        Select events using softmax probability sampling
        
        Args:
            events: Re-ranked event list
            
        Returns:
            Selected event
        """
        if not events:
            return None
            
        ranks = np.arange(1, len(events) + 1)
        inverse_ranks = - ranks
        
        probabilities = np.exp(inverse_ranks) / np.sum(np.exp(inverse_ranks))
        
        selected_idx = np.random.choice(len(events), p=probabilities)
        return events[selected_idx]

    def rewrite_event(self, user_profile: str, location_desc: str, event_context: str, selected_event: dict, goal: str):
        """
        Rewrite event based on user profile and current environment
        
        Args:
            user_profile: User profile
            location_desc: Detailed description of the location
            selected_event: Selected event
            
        Returns:
            Rewritten event
        """
        prompt = REWRITE_PROMPT.format(
            user_profile=user_profile,
            location_desc=location_desc,
            event_sequences=event_context,
            event_text=selected_event['event'],
            intent=selected_event.get('intent', ''),
            goal=goal
        )
        
        response = self.model.chat([{'role': 'user', 'content': prompt}])
        response = parse_json_dict_response(response, keys=['event', 'intent'])

        if response.get('event', None):
            selected_event['event'] = response['event']
        if response.get('intent', None):
            selected_event['intent'] = response['intent']

        return selected_event
    
    def get_event_context(self, event_results):
        """
        Get the text of events the user has already experienced
        
        Args:
            event_results: List of events that have occurred
            
        Returns:
            Event text
        """
        if not event_results:
            return "None"
        
        event_texts = []
        for i, res in enumerate(event_results):
            event = res['selected_event']['event']
            intent = res['selected_event']['intent']
            time = res['trajectory_point']['time']
            location = res['trajectory_point']['location']
            weather = res['trajectory_point']['weather']
            formatted_event = POI_Event.from_dict({
                'time': time,
                'location': location,
                'life_event': event,
                'intent': intent,
                'weather': weather
            }, timezone=None)
            event_desc = f"({i+1}) {formatted_event.desc(keys_to_drop=['event'])}"
            event_texts.append(event_desc)
        
        return "\n".join(event_texts)

    def get_user_goal(self, user_profile: str, event_context: str, previous_goal: str) -> str:
        """
        Infer user's longterm goal
        
        Args:
            user_profile: User profile
            event_context: Text of events the user has already experienced
            previous_goal: Previously inferred goal
            
        Returns:
            Inferred longterm goal
        """
        template = get_infer_goal_prompt(self.theme)
        prompt = template.format(
            user_profile=user_profile,
            event_sequences=event_context,
            goal=previous_goal
        )
        
        response = self.model.chat([{'role': 'user', 'content': prompt}])
        response = parse_json_dict_response(response, keys=['goal'])
        goal = response.get('goal', previous_goal)
        return goal

    def process_trajectory(self, trajectory: List[POI_Event], user_profile: str, longterm_goal: str = '', max_n_events: int = 10, random_start_event: int = 10086) -> Tuple[List[Dict], str]:
        """
        Process the entire travel trajectory and match possible events for each location
        
        Args:
            trajectory: User travel trajectory
            user_profile: User profile
            max_n_events: Maximum number of events
            random_start_event: Randomly selected events number before summarizing the user's longterm goal
            
        Returns:
             List of possible events for each point in the trajectory
        """
        results = []
        pre_locations = set()
        pre_events = set()
        start_time = datetime.strptime(trajectory[0].time, "%Y-%m-%d %H:%M:%S, %A").timestamp()
        
        end_time = datetime.strptime(trajectory[-1].time, "%Y-%m-%d %H:%M:%S, %A").timestamp()
        t0 = (end_time - start_time) // (3600 * 10) 

        progress_bar = tqdm(range(max_n_events))

        goal = longterm_goal

        for i, location_point in enumerate(trajectory):
            if location_point.location in pre_locations:
                continue
            
            self.logger.info(f"Processing trajectory point {i+1}/{len(trajectory)}: [{location_point.time}] {location_point.location}")
            
            current_time = datetime.strptime(location_point.time, "%Y-%m-%d %H:%M:%S, %A").timestamp()
            
            is_theme = self.is_theme_location(location_point.location)
            
            event_prob = self.calculate_event_probability(current_time, start_time, is_theme, t0=t0)
            self.logger.info(f"Calculate event occurrence probability: {event_prob:.3f}, Is it a theme-related location: {is_theme}")    
            
            if random.random() < event_prob:
                dimensions = list(self.event_dimensions.keys())
                all_candidate_events = []
                
                event_context = self.get_event_context(results)

                for dimension in dimensions:
                    query = self.generate_query_by_dimension(
                        user_profile,
                        event_context, 
                        location_point.desc(keys_to_drop=['life_event', 'intent']),
                        dimension,
                        goal=goal
                    )
                    
                    similar_events = self.retrieve_similar_events(query, top_k=3)
                    all_candidate_events.extend(similar_events)
                
                if all_candidate_events:
                    all_candidate_events = [x for x in all_candidate_events if x['event'] not in pre_events]

                    reranked_events = self.rerank_events(
                        all_candidate_events,
                        user_profile=user_profile,
                        location_desc=location_point.desc(keys_to_drop=['life_event', 'intent']),
                        event_context=event_context,
                        goal=goal,
                        n_keep=1
                    )
                    
                    selected_event = self.softmax_sampling(reranked_events)
                    
                    if selected_event:
                        pre_locations.add(location_point.location)
                        pre_events.add(selected_event['event'])

                        selected_event = self.rewrite_event(
                            user_profile=user_profile, 
                            location_desc=location_point.desc(keys_to_drop=['life_event', 'intent']),
                            event_context=event_context, 
                            selected_event=selected_event,
                            goal=goal
                        )

                        result = {
                            'trajectory_point': location_point.to_dict(),
                            'event_probability': event_prob,
                            'selected_event': selected_event,
                            'candidate_events': all_candidate_events
                        }
                        
                        results.append(result)
                        self.logger.info(f"Selected event {len(results)}: {selected_event['event']}...")
                        start_time = current_time

                        progress_bar.update(1)
                        if len(results) >= random_start_event:
                            event_context = self.get_event_context(results)
                            goal = self.get_user_goal(
                                user_profile=user_profile,
                                event_context=event_context,
                                previous_goal=goal
                            )
                            self.logger.info(f"Inferred user longterm goal: {goal}")
                        
                        continue
                
            self.logger.info(f"Not occurred event")
            if len(results) >= max_n_events:
                self.logger.info(f"Sufficient events matched ({len(results)}), stop processing")
                break
        
        return results, goal
