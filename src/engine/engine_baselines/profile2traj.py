import random
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Any
import math
from dataclasses import dataclass, field, fields, asdict
from tqdm.auto import tqdm

from engine.prompts import get_event_dimensions, get_infer_goal_prompt, RERANK_PROMPT, REWRITE_PROMPT
from utils.utils import get_logger, parse_json_dict_response, load_jsonl_data

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
        template = "The weather condition is {conditions}, described as {description}. The average temperature is {temp}°C (high of {tempmax}°C, low of {tempmin}°C)."
        weather_str = template.format(
            conditions=self.weather['conditions'],
            description=self.weather['description'],
            temp=self.weather['temp'],
            tempmax=self.weather['tempmax'],
            tempmin=self.weather['tempmin']
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

    def process_trajectory(self, trajectory: List[POI_Event], user_profile: str, longterm_goal: str = '', max_n_events: int = 10, random_start_event: int = 10086) -> Tuple[List[Dict], str]:
        """
        Directly generate user life and intention sequence based on user profile and long term goal.
        An event contain time, location, weather, life_event, and intent.
        
        Args:
            trajectory: User travel trajectory
            user_profile: User profile
            max_n_events: Maximum number of events
            random_start_event: Randomly selected events number before summarizing the user's longterm goal
            
        Returns:
             List of possible events for each point in the trajectory
        """
        results = []
        start_time = datetime.strptime(trajectory[0].time, "%Y-%m-%d %H:%M:%S, %A").timestamp()
        
        end_time = datetime.strptime(trajectory[-1].time, "%Y-%m-%d %H:%M:%S, %A").timestamp()

        PROMPT = (
            "You are provided with a user profile, a long-term goal and scenario themes.\n"
            f"You need to generate {max_n_events} events the user may experience from {start_time} to {end_time}.\n"
            "For each event, you need to provide time, location, weather, life_event, and intent.\n"
            "Make sure the generated events are diverse and relevant to the user's profile and long-term goal.\n"
            "### Output Format\n"
            "Provide the results in a JSON array, where each element represents an event:\n"
            "```json\n"
            "[\n"
            "  {\n"
            "       \"time\": \"...\",\n"
            "       \"location\": \"...\",\n"
            "       \"weather\": \"...\",\n"
            "       \"life_event\": \"...\",\n"
            "       \"intent\": \"...\"\n"
            "  }\n"
            "  ...\n"
            "]\n"
            "```\n"
            "Where:\n"
            "- time: The time of the event in 'YYYY-MM-DD HH:MM:SS, Day' format.\n"
            "- location: The location of the event.\n"
            "- weather: a breif description of the weather during the event.\n"
            "- life_event: A brief description of the life event experienced by the user.\n"
            "- intent: The user's purpose in seeking assistance from the AI assistant during the event.\n"
            "### Event Examples\n"
            "Event1:\n"
            "{\n"
            "   \"time\": \"2012-07-01 10:00:00, Monday\",\n"
            "   \"location\": \"Central Park, New York City\",\n"
            "   \"weather\": \"Sunny, 25°C\",\n"
            "   \"life_event\": \"She recently feels exhausted from housework and spending time with her partner, leaving almost no time for physical activities.\",\n" 
            "   \"intent\": \"The user seeks to adjust her daily routine to include more exercise while maintaining quality time with her partner and managing household tasks.\"\n"
            "}\n"
            "Event2:\n"
            "{\n"
            "   \"time\": \"2012-03-22 18:58:00, Sunday\",\n"
            "   \"location\": \"Chinese Restaurant, New York City\",\n"
            "   \"weather\": \"Rainy, 16°C\",\n"
            "   \"life_event\": \"The user notices that his three-month-old infant, despite being in a familiar setting, suddenly becomes restless and difficult to soothe.\",\n" 
            "   \"intent\": \"The user is looking to compose a lullaby or a simple, age-appropriate tune to entertain or calm the baby.\"\n"
            "}\n"
            "### Input\n"
            "[User Profile]\n"
            f"{user_profile}\n"
            "[Long-term Goal]\n"
            f"{longterm_goal}\n"
            "[Scenario Theme]\n"
            f"{self.theme}\n"
            "[Output]\n"
        )
        results = self.model.chat([{'role': 'user', 'content': PROMPT}])
        parsed_results = parse_json_dict_response(results)
        assert len(parsed_results) >= max_n_events, "Generated events are fewer than expected."
        events = []
        for event_dict in parsed_results[:max_n_events]:
            try:
                assert "time" in event_dict, "Missing 'time' in generated event."
                assert "location" in event_dict, "Missing 'location' in generated event."
                assert "weather" in event_dict, "Missing 'weather' in generated event." 
                assert "life_event" in event_dict, "Missing 'life_event' in generated event."
                assert "intent" in event_dict, "Missing 'intent' in generated event."
                events.append(event_dict)
            except:
                continue
        
        return events, longterm_goal