import json
import os
import re
from datetime import datetime
import time
import random
from typing import Optional, Dict, Any

from utils.utils import get_logger

class ConversationSimulator:
    def __init__(self, user_profile_generator, life_event_engine, user_agent, assistant_agent, on_turn_update=None, logger_silent: bool = False):
        self.user_pg = user_profile_generator
        self.life_event_engine = life_event_engine
        self.user_agent = user_agent
        self.assistant_agent = assistant_agent
        self.dialogue_log = []
        self.on_turn_update = on_turn_update

        self.logger = get_logger(__name__, silent=logger_silent)

    def update_interface(self, role, content, episode_log, thinking=None, round_index=None, **kwargs):
        dialogue_turn = {
            "role": role,
            "content": content,
            "thinking": thinking
        }
        episode_log["dialogue"].append(dialogue_turn)
        data = {
            "step": "turn",
            "round_index": round_index,
            "dialogue": episode_log["dialogue"],
            "dropout": None
        }
        data.update(kwargs)
        self.on_turn_update(data)
        time.sleep(0.1)
        return episode_log

    def init_env(self, sequence_id):
        self.life_event_engine.set_event_sequence(sequence_id)
        user_id = self.life_event_engine.get_current_user_id()
        self.logger.info(f"👤 Initializing environment for user_id: {user_id}, sequence_id: {sequence_id}")
        self.user_profile = self.user_pg.get_profile_by_id(user_id)
        self.user_agent._build_profile(str(self.user_profile))
        self.assistant_agent._build_user_profile(self.user_profile)

    def reset(self):
        self.user_agent.reinit()
        self.assistant_agent.reinit()

    def run_simulation(self, n_events: int = 5, n_rounds: int = 20, **config):
        for i in range(n_events):
            self.logger.info(f"[The {i+1}‑th interaction scenario]")
            try:
                self.run_episode(n_rounds=n_rounds, **config)
                self.reset()
            except Exception as e:
                self.logger.exception(f"[The {i+1}-th interaction scenario] encountered an error")
                continue

    def run_episode(self, n_rounds: int = 20, user_config: Optional[Dict] = None, assistant_config: Optional[Dict] = None):
        event = self.life_event_engine.generate_event()
        self.logger.info('[Event] ' + str(event))
        if self.on_turn_update:
            self.on_turn_update({
                "step": "init",
                "event": event,
                "round": 0,
                "dialogue": [],
                "dropout": None
            })

        self.user_agent._build_environment(event)
        self.assistant_agent._build_system_prompt(event)

        episode_log = {
            "user": {
                "profile": self.user_profile.to_dict(),
                "profile_str": str(self.user_profile)
            },
            "event": event,
            "dialogue": [],
            "pre_profile": None,
        }

        user_response = ''
        assistant_response = ''

        for round_index in range(1, n_rounds + 1):
            # User action and response
            result = self.user_agent.respond(assistant_response, **user_config)
            action = result['action']
            user_response = result['response']
            self.logger.info(f'[User Action] ' +  str(action))
            if action == self.user_agent.action_space.END_CONVERSATION:
                break
            self.logger.info('[User] ' + str(user_response))

            if self.on_turn_update:
                episode_log = self.update_interface(
                    role='user', 
                    content=user_response, 
                    episode_log=episode_log, 
                    round_index=round_index
                )
            else:
                dialogue_turn = {
                    "role": 'user',
                    "memory_similarity": result.get('memory_similarity', None),
                    "emotion": result.get('emotion', None),
                    "content": user_response
                }
                episode_log["dialogue"].append(dialogue_turn)

            # Assistant response
            pre_intent, assistant_response = self.assistant_agent.respond(input=user_response, event=event, **assistant_config)
            self.logger.info('[Predicted intent by assistant] ' + str(pre_intent))
            self.logger.info('[Assistant] ' + str(assistant_response))

            if self.on_turn_update:
                episode_log = self.update_interface(
                    role='assistant', 
                    content=assistant_response, 
                    episode_log=episode_log, 
                    round_index=round_index
                )
            else:
                dialogue_turn = {
                    "role": 'assistant',
                    "pre_intent": pre_intent,
                    "content": assistant_response
                }
                episode_log["dialogue"].append(dialogue_turn)

        try:
            dimensions = list(self.user_profile.preferences_value.keys())
        except:
            dimensions = [list(x.keys())[0] for x in self.user_profile.preferences_value]
        reply = self.assistant_agent.summarize(event, dimensions, **assistant_config)
        self.logger.info('[Assistant summary] ' + str(reply))
        episode_log['pre_profile'] = reply
        self.dialogue_log.append(episode_log)

    def _format_content(self, text):
        text = text.strip().rstrip()
        quote_pairs = [
            ('"', '"'),
            ("'", "'"),
            ('“', '”'),
            ('‘', '’'),
        ]
        for left, right in quote_pairs:
            if text.startswith(left) and text.endswith(right):
                text = text[len(left):-len(right)].strip()
                break
        text = text.replace('~', ' ')
        return text

    def get_dialogue_log(self):
        return self.dialogue_log
    
    def save(self, path="./logs", filename=None):
        os.makedirs(path, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sim_log_{timestamp}.json"

        full_path = os.path.join(path, filename)

        info_to_save = {
            "event_sequence_info": self.life_event_engine.get_current_sequence_info(),
            "dialogue_log": self.dialogue_log
        }

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(info_to_save, f, ensure_ascii=False, indent=2)

        self.logger.info(f"[✓] Simulator log has been saved to {full_path}")
        
        user_full_path = os.path.join(path, 'user_' + filename)
        self.user_agent.save(user_full_path)

        assistant_full_path = os.path.join(path, 'assistant_' + filename)
        self.assistant_agent.save(assistant_full_path)