import json
import os
import re
from datetime import datetime
import time
import random
from typing import Optional, Dict, Any
from engine.event_engine import POI_Event
from profiles.profile_generator import UserProfile

from utils.utils import get_logger

class ConversationSimulator:
    def __init__(self, user_profile_generator, life_event_engine, user_agent, assistant_agent, on_turn_update=None, 
        logger_silent: bool = False, analysis_agent = None):
        self.user_pg = user_profile_generator
        self.life_event_engine = life_event_engine
        self.user_agent = user_agent
        self.assistant_agent = assistant_agent
        self.dialogue_log = []
        self.on_turn_update = on_turn_update

        self.analysis_agent = analysis_agent
        self.improvement_iterations = []

        self.logger = get_logger(__name__, silent=logger_silent)

    def update_interface_dialog(self, role, content, episode_log, round_index=None, **kwargs):
        dialogue_turn = {
            "role": role,
            "content": content,
        }
        dialogue_turn.update(kwargs)
        self.logger.info(dialogue_turn)

        episode_log["dialogue"].append(dialogue_turn)

        if not self.on_turn_update:
            return episode_log

        data = {
            "step": "turn",
            "round_index": round_index,
            "utterance": dialogue_turn,
        }
        self.on_turn_update(data)
        time.sleep(0.05)
        return episode_log

    def update_interface_analysis(self, role, result, episode_log, round_index=None, **kwargs):
        analysis_turn = {
            "role": role,
            "result": result,
        }
        analysis_turn.update(kwargs)
        episode_log["analysis"].append(analysis_turn)
        if not self.on_turn_update:
            return episode_log
        
        data = {
            "step": "analysis",
            "round_index": round_index,
            "analysis": episode_log["analysis"]
        }
        data.update(kwargs)
        self.on_turn_update(data)
        time.sleep(0.05)
        return episode_log
    
    def update_interface_dropout(self, result, episode_log, round_index=None, **kwargs):
        episode_log["dropout"] = result
        if not self.on_turn_update:
            return episode_log
        
        data = {
            "step": "dropout",
            "round_index": round_index,
            "dropout": result
        }
        data.update(kwargs)
        self.on_turn_update(data)
        time.sleep(0.05)
        return episode_log

    def init_env(self, sequence_id):
        self.dialogue_log = []
        self.life_event_engine.set_event_sequence(sequence_id)
        user_id = self.life_event_engine.get_current_user_id()
        self.logger.info(f"👤 Initializing environment for user_id: {user_id}, sequence_id: {sequence_id}")
        self.user_profile = self.user_pg.get_profile_by_id(user_id)
        self.user_agent._build_profile(str(self.user_profile))
        self.assistant_agent._build_user_profile(self.user_profile)
        self.reset()
    
    def init_env_by_custom_profile_and_events(self, profile, events):
        self.dialogue_log = []
        self.life_event_engine.set_event_sequence_by_profile_and_events(profile, events)
        # self.logger.info(f"👤 Initializing environment for user_id: {user_id}, sequence_id: {sequence_id}")
        self.user_profile = UserProfile.from_dict(profile)
        self.user_agent._build_profile(str(self.user_profile))
        self.assistant_agent._build_user_profile(self.user_profile)
        self.reset()

    def reset(self):
        self.user_agent.reinit()
        self.assistant_agent.reinit()
    
    def run_simulation(self, n_events: int = 5, n_rounds: int = 20, **config):
        for i in range(n_events):
            self.logger.info(f"[The {i+1}‑th interaction scenario]")
            try:
                _, strategy = self.run_episode(i+1, n_rounds=n_rounds, enable_turn_analysis=False, **config)
                self.reset()
                # _, strategy = self.run_episode(i+1, n_rounds=n_rounds, enable_turn_analysis=True, strategy=strategy, **config)
                # self.reset()
            except Exception as e:
                self.logger.exception(f"[The {i+1}-th interaction scenario] encountered an error")
                continue
    
    def run_episode(self, event_index: int, n_rounds: int = 20, n_advice: int = 1, 
                    user_config: Optional[Dict] = None, assistant_config: Optional[Dict] = None,
                    enable_turn_analysis: bool = False, strategy: str = ''):
        """
        修改原有的run_episode，添加enable_turn_analysis参数控制是否启用逐轮分析
        """
        if enable_turn_analysis:
            event = self.life_event_engine.get_current_event()
        else:
            event = self.life_event_engine.generate_event()
        self.logger.info('[Event] ' + str(event))
        
        if self.on_turn_update:
            if enable_turn_analysis:
                self.on_turn_update({
                    "step": "improvement_start",
                    "event": event,
                    "round": 0,
                    "event_index": event_index,
                    "dialogue": []
                })
            else:
                self.on_turn_update({
                    "step": "init",
                    "event": event,
                    "round": 0,
                    "event_index": event_index,
                    "dialogue": []
                })

        self.user_agent._build_environment(event)
        self.user_agent._build_chat_system_prompt()
        self.assistant_agent._build_system_prompt(event)

        episode_log = {
            "user": {
                "profile": self.user_profile.to_dict(),
                "profile_str": str(self.user_profile)
            },
            "event": event,
            "dialogue": [],
            "analysis": [],
            "dropout": {},
            "pre_profile": None,
            "enable_turn_analysis": enable_turn_analysis
        }

        user_response = ''
        assistant_response = ''

        for round_index in range(1, n_rounds + 1):
            # User action and response
            result = self.user_agent.respond(assistant_response, **user_config)
            action = result['action']
            user_response = result['response']
            self.logger.info(f'[User Action] ' + str(action))
            
            if action == self.user_agent.action_space.END_CONVERSATION:
                break
            
            self.logger.info('[User] ' + str(user_response))

            if self.on_turn_update:
                episode_log = self.update_interface_dialog(
                    role='user', 
                    content=user_response, 
                    episode_log=episode_log, 
                    round_index=round_index,
                    **{
                        "emotion": result.get('emotion', None),
                        "memory_similarity": result.get('memory_similarity', None)
                    }
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
            assistant_response = self.assistant_agent.respond(input=user_response, event=event, **assistant_config)
            self.logger.info('[Assistant] ' + str(assistant_response))

            if self.on_turn_update:
                episode_log = self.update_interface_dialog(
                    role='assistant', 
                    content=assistant_response,
                    episode_log=episode_log, 
                    round_index=round_index,
                    **{
                        "emotion": None,
                        "memory_similarity": None
                    }
                )
            else:
                dialogue_turn = {
                    "role": 'assistant',
                    "content": assistant_response
                }
                episode_log["dialogue"].append(dialogue_turn)

            # === 助手质量分析（仅在启用时执行） ===
            if enable_turn_analysis and self.analysis_agent:
                for _ in range(n_advice):
                    formatted_event = POI_Event.from_dict(event)
                    result = self.analysis_agent.assistant_quality_analysis(
                        user_profile=str(self.user_profile),
                        conversation_context=episode_log["dialogue"],
                        assistant_utterance=assistant_response,
                        event=formatted_event.desc(),
                        strategy=strategy
                    )
                    self.logger.info(f"[Analysis (Assistant)] ({result['flags']}) {result['advice']}")
                    
                    if result['flags']:
                        assistant_response = self.assistant_agent.revise_last_turn(result['advice'], user_response)
                        self.logger.info('[Assistant (Revised)] ' + str(assistant_response))

                    if self.on_turn_update:
                        episode_log = self.update_interface_analysis(
                            role='assistant', 
                            result=result, 
                            episode_log=episode_log, 
                            round_index=round_index
                        )
                        if result['flags']:
                            episode_log = self.update_interface_dialog(
                                role='assistant_revise', 
                                content=assistant_response, 
                                episode_log=episode_log, 
                                round_index=round_index,
                                **{
                                    "emotion": None,
                                    "memory_similarity": None
                                }
                            )
                    else:
                        analysis_turn = {
                            "role": 'assistant',
                            "result": result
                        }
                        episode_log["analysis"].append(analysis_turn)
                        if result['flags']:
                            dialogue_turn = {
                                "role": 'assistant_revise',
                                "content": assistant_response
                            }
                            episode_log["dialogue"].append(dialogue_turn)
        self.dialogue_log.append(episode_log)

        strategy = ''
        if self.analysis_agent:
            formatted_event = POI_Event.from_dict(event)
            result = self.analysis_agent.predict_dropout(
                conversation_context=episode_log["dialogue"],
                user_profile=str(self.user_profile),
                event=formatted_event.desc(),
                intents=event['sub_intents']
            )
            self.logger.info(f"[Dropout analysis] Risk: {result['risk']}\nReason: {result['reason']}\nStrategy: {result['strategy']}")
            strategy = result.get('strategy', '')
            if self.on_turn_update:
                episode_log = self.update_interface_dropout( 
                    result=result, 
                    episode_log=episode_log, 
                    round_index=round_index
                )
            else:
                episode_log["dropout"].append(result)
        
        return episode_log, strategy

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
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(info_to_save, f, ensure_ascii=False, indent=2)

            self.logger.info(f"[✓] Simulator log has been saved to {full_path}")
            
            user_full_path = os.path.join(path, 'user_' + filename)
            self.user_agent.save(user_full_path)

            assistant_full_path = os.path.join(path, 'assistant_' + filename)
            self.assistant_agent.save(assistant_full_path)

            self.user_agent.model.save(os.path.join(path, 'user_model_' + filename))
        except Exception as e:
            self.logger.error(f"[✗] Failed to save simulator log: {e}")