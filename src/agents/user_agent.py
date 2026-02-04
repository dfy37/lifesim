from collections import deque
from enum import Enum
import json
import numpy as np
import threading
import contextlib
import re
from typing import List, Optional

from agents.prompts import (
    USER_CONV_SYSTEM_PROMPT,
    USER_CONV_PROMPT,
    USER_REVISE_CONV_PROMPT,
    USER_MEMORY_PROMPT,
    USER_EMOTION_PROMPT,
    USER_ACTION_PROMPT,
    USER_BELIEF_PROMPT
)
from agents.memory import KVMemory, SimpleMemory, NullMemory
from utils.utils import parse_json_dict_response, find_closest_str_match, get_logger
from json_repair import loads as repair_json

class UserActionEnum(str, Enum):
    END_CONVERSATION = "End Conversation"
    CONTINUE = "Continue Conversation"

EMOTION_LIST = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

class UserAgent:
    _lock = threading.Lock()

    @classmethod
    @contextlib.contextmanager
    def synchronized(cls):
        with cls._lock:
            yield

    def __init__(
        self,
        model,
        retriever_config: Optional[dict] = None,
        profile: Optional[str] = None,
        alpha: float = 0.5,
        logger_silent: bool = False,
    ):
        self.model = model
        if profile:
            self.static_memory = SimpleMemory(profile)
        else:
            self.static_memory = SimpleMemory()
        if retriever_config:
            self.dynamic_memory = KVMemory(retriever_config)
        else:
            self.dynamic_memory = NullMemory()
        self.emotion_chaine = []
        self.conversations = SimpleMemory()
        self.history_messages = []
        self.messages = []
        self.action_space = UserActionEnum
        self.states = []
        self.beliefs = []

        # Memory perception coefficient
        self.alpha = alpha

        self.logger = get_logger(__name__, silent=logger_silent)
    
    def _build_profile(self, profile: str):
        self.static_memory = SimpleMemory(profile)

    def _build_environment(self, event: dict) -> None:
        """
        Environment for agents

        :param event: scenario event
        """
        self.event = event

    def _merge_beliefs(self, new_beliefs: list) -> None:
        if not isinstance(new_beliefs, list):
            return
        existing = {self._belief_key(belief) for belief in self.beliefs}
        for belief in new_beliefs:
            normalized = self._normalize_belief(belief)
            if normalized is None:
                continue
            key = self._belief_key(normalized)
            if key in existing:
                continue
            existing.add(key)
            self.beliefs.append(normalized)

    @staticmethod
    def _belief_key(belief: list) -> tuple:
        triple, description, time, utterance = belief
        return (tuple(triple), description, time, utterance)

    @staticmethod
    def _normalize_belief(belief: list | tuple) -> list | None:
        if not isinstance(belief, (list, tuple)) or len(belief) != 4:
            return None
        triple, description, time, utterance = belief
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return None
        source, relation, target = triple
        triple_norm = [
            str(source).strip(),
            str(relation).strip(),
            str(target).strip()
        ]
        if any(not value for value in triple_norm):
            return None
        description_norm = str(description).strip() if description is not None else ""
        time_norm = str(time).strip() if time is not None else None
        utterance_norm = None
        if utterance is not None and str(utterance).strip():
            try:
                utterance_norm = int(utterance)
            except (TypeError, ValueError):
                utterance_norm = None
        return [triple_norm, description_norm, time_norm, utterance_norm]

    @staticmethod
    def _parse_beliefs_response(text: str) -> list:
        if not isinstance(text, str) or not text.strip():
            return []
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        json_str = match.group(1).strip() if match else text.strip()
        try:
            data = repair_json(json_str)
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def update_belief_from_event(self, event: dict) -> dict:
        self.logger.info("[UserAgent] Start updating beliefs from event...")
        event_time = event.get("time") or event.get("timestamp")
        utterance_index = event.get("utterance") or event.get("utterance_index")
        belief_prompt = USER_BELIEF_PROMPT.format(
            profile=self.static_memory.get(),
            event=event.get("life_event") or event.get("event", ""),
            dialogue_scene=event.get("dialogue_scene", ""),
            belief_list=json.dumps(self.beliefs, ensure_ascii=False),
            event_time=event_time,
            utterance_index=utterance_index
        )
        with self.synchronized():
            belief_response = self.model.chat([{'role': 'user', 'content': belief_prompt}])
        belief_data = self._parse_beliefs_response(belief_response)
        self._merge_beliefs(belief_data)
        self.logger.info(
            "[UserAgent] Beliefs updated: "
            f"count={len(self.beliefs)}"
        )
        return belief_data

    def _build_chat_system_prompt(self) -> str:
        profile = self.static_memory.get()
        explicit_intent = [x['description'] for x in self.event['sub_intents'] if x['type'] == 'explicit']
        implicit_intent = [x['description'] for x in self.event['sub_intents'] if x['type'] == 'implicit']
        
        explicit_intent = '\n- '.join(explicit_intent).strip()
        implicit_intent = '\n- '.join(implicit_intent).strip()
        
        self.system_prompt = USER_CONV_SYSTEM_PROMPT.format(
            profile=profile, 
            dialogue_scene=self.event['dialogue_scene'],
            event=self.event['event'], 
            intent=self.event['intent'],
            explicit_intent=explicit_intent,
            implicit_intent=implicit_intent
        )

        return self.system_prompt

    def _process_dynamic_memory(self, prompt: str):
        memory_perception = ''
        S = -1

        self.logger.info("[UserAgent] Start processing dynamic memory...")
        # 检查是否需要存储动态记忆
        memory_prompt = USER_MEMORY_PROMPT.format(
            profile=self.static_memory.get(),
            event=self.event['event'], 
            intent=self.event['intent'],
            dialogue_scene=self.event['dialogue_scene'],
            conversation_context=self.conversations.get(),
            content=f"Assistant: {prompt}"
        )
        with self.synchronized():
            memory_response = self.model.chat([{'role': 'user', 'content': memory_prompt}])
        memory_data = parse_json_dict_response(memory_response, ['need_store', 'query', 'response'])
        self.logger.info(f"[UserAgent] Memory Analysis Results: need_store={memory_data.get('need_store', 'N/A')}")

        if memory_data['need_store'] and memory_data['need_store'].lower() == 'true':
            query = memory_data['query']
            response = memory_data['response']
            self.logger.info(f"[UserAgent] Start Storing Memories - Query: {query}\nResponse: {response}")
            
            if query != '-1' and response != '-1':
                text = f"Query: {query}\nResponse: {response}"
                memories = self.dynamic_memory.search(text, top_k=3)
                self.logger.info(f"[UserAgent] Found {len(memories)} related memories")
                
                if len(memories) == 0:
                    S = 0
                else:
                    S = sum([m[1] for m in memories]) / len(memories)
                
                score = self.alpha * np.log(1e-9 + S) + (1 - self.alpha) * np.log(1e-9 + 1 - S)
                self.logger.info(f"[UserAgent] Memory value score: S={S:.4f}, score={score:.4f}")
                
                if S >= 0.7:
                    memory_perception = 'The current assistant reply highly overlaps with the user’s existing memories, triggering some negative emotions (such as impatience).'
                    self.logger.info(f"[UserAgent] {memory_perception}")
                else:
                    memory_perception = 'The current assistant reply has a low degree of overlap with the user’s existing memories.'
                    self.logger.info(f"[UserAgent] {memory_perception}")
                
                self.dynamic_memory.add_key_value(query, response)
                self.logger.info("[UserAgent] Memory has been successfully stored")
            else:
                self.logger.info("[UserAgent] Skipped memory storage (query or response is -1)")
        else:
            self.logger.info("[UserAgent] Skip memory storage (no need to store)")
        return memory_perception, S

    def _analyze_emotion(self, memory_perception: str):
        emotion = ''

        self.logger.info("[UserAgent] Start emotion analysis...")
        emotion_prompt = USER_EMOTION_PROMPT.format(
            profile=self.static_memory.get(),
            event=self.event['event'], 
            intent=self.event['intent'],
            dialogue_scene=self.event['dialogue_scene'],
            conversation_context=self.conversations.get(),
            perception=memory_perception,
            emotion_options=",".join(EMOTION_LIST)
        )
        with self.synchronized():
            emotion_response = self.model.chat([{'role': 'user', 'content': emotion_prompt}])
        self.logger.info(f"[UserAgent] Emotion reasoning: {emotion_response}")
        emotion = parse_json_dict_response(emotion_response, ['emotion'])['emotion']
        
        if not emotion:
            emotion = 'neutral'
            self.logger.info("[UserAgent] Emotion analysis result: neutral (default)")
        else:
            emotion = find_closest_str_match(emotion, EMOTION_LIST)
            self.logger.info(f"[UserAgent] Current emotion: {emotion}")
        
        return emotion

    def _decide_action(self, memory_perception: str, emotion: str):
        self.logger.info("[UserAgent] Start action decision...")
        action_prompt = USER_ACTION_PROMPT.format(
            conversation_context=self.conversations.get(),
            profile=self.static_memory.get(),
            event=self.event['event'], 
            intent=self.event['intent'],
            emotion=emotion,
            perception=memory_perception,
            action_options="\n".join([f"- {action.value}" for action in self.action_space])
        )
        with self.synchronized():
            action_response = self.model.chat([{'role': 'user', 'content': action_prompt}])
        self.logger.info(f"[UserAgent] Action reasoning: {action_response}")
        action_response = parse_json_dict_response(action_response, ['action'])['action']

        action = self.select_action(action_response)
        self.logger.info(f"[UserAgent] Action decision result: {action.value}")

        return action

    def select_action(self, action_response):
        """
        Action selection

        Return : UserActionEnum.END_CONVERSATION or UserActionEnum.CONTINUE
        """
        try:
            for action in self.action_space:
                if action in action_response or action.value in action_response:
                    return action
            
            action_response_lower = action_response.lower()
            for action in self.action_space:
                if action.lower() in action_response_lower or action.value.lower() in action_response_lower:
                    return action
            
            self.logger.warning(f"Warning: Unable to parse action selection '{action_response}', defaulting to continue conversation.")
            return self.action_space.CONTINUE
                
        except Exception as e:
            self.logger.error(f"Action selection error: {e}")
            return self.action_space.CONTINUE

    def set_messages(self, messages: List[dict]):
        self.messages = messages

    def respond(self, prompt: str, use_emotion_chain: bool = False, use_dynamic_memory: bool = False) -> dict:      
        memory_perception = ''
        S = -1 
        if use_dynamic_memory and len(prompt) > 0:
            memory_perception, S = self._process_dynamic_memory(prompt)
        
        self.conversations.add(f"Assistant: {prompt}")

        emotion = ''
        if use_emotion_chain and len(prompt) > 0 and len(self.messages) > 1:
            emotion = self._analyze_emotion(memory_perception)
        else:
            emotion = "neutral"

        if len(prompt) > 0 and len(self.messages) > 4:
            action = self._decide_action(memory_perception, emotion)
        else:
            action = self.action_space.CONTINUE
        
        if action == self.action_space.END_CONVERSATION:
            self.logger.info("[UserAgent] Decide to end conversation.")
            return {
                'action': action,
                'response': ''
            }

        self.logger.info("[UserAgent] Start replying...")
        prompt = USER_CONV_PROMPT.format(
            content=f"助手回复: {prompt}" if len(prompt) > 0 else "",
            perception=f"当前记忆感知: {memory_perception}" if len(memory_perception) > 0 else "",
            emotion=f"当前用户情绪: {emotion}" if len(emotion) > 0 else "",
        ).strip()
        
        if len(self.messages) < 1:
            self.messages.append({
                'role': 'system',
                'content': self.system_prompt
            })

        if self.messages[0]['role'] != 'system':
            self.messages.insert(0, {
                'role': 'system',
                'content': self.system_prompt
            })
        
        self.messages.append({
            'role': 'user',
            'content': prompt
        })

        with self.synchronized():
            reply_content = self.model.chat(self.messages)

        self.messages.append({
            'role': 'assistant',
            'content': reply_content
        })
        self.conversations.add(f"User: {reply_content}")
        self.states.append({
            'action': action,
            'emotion': emotion,
            'memory_perception': memory_perception,
            'memory_similarity': S
        })

        result =  {
            'action': action,
            'response': reply_content,
            'memory_similarity': S,
            'emotion': emotion
        }

        return result
    
    def rollback_last_turn(self):
        assert self.messages[-1]['role'] == 'assistant', self.messages[-1]
        self.messages = self.messages[:-2]
        self.conversations.kpop(2)
    
    def revise_last_turn(self, advice: str, prompt: str):
        self.rollback_last_turn()
        self.conversations.add(f"Assistant: {prompt}")

        emotion = self.states[-1]['emotion']
        memory_perception = self.states[-1]['memory_perception']

        self.logger.info("[UserAgent] Start revising...")
        prompt = USER_REVISE_CONV_PROMPT.format(
            content=f"助手回复: {prompt}" if len(prompt) > 0 else "",
            perception=f"当前记忆感知: {memory_perception}" if len(memory_perception) > 0 else "",
            emotion=f"当前用户情绪: {emotion}" if len(emotion) > 0 else "",
            advice=f"请根据以下建议给出你的回复: {advice}"
        ).strip()

        self.messages.append({
            'role': 'user',
            'content': prompt
        })

        with self.synchronized():
            reply_content = self.model.chat(self.messages)

        self.messages[-1]['content'] = USER_CONV_PROMPT.format(
            content=f"助手回复: {prompt}" if len(prompt) > 0 else "",
            perception=f"当前记忆感知: {memory_perception}" if len(memory_perception) > 0 else "",
            emotion=f"当前用户情绪: {emotion}" if len(emotion) > 0 else ""
        )

        self.messages.append({
            'role': 'assistant',
            'content': reply_content
        })
        self.conversations.add(f"User: {reply_content}")
        result =  {
            'action': self.states[-1]['action'],
            'response': reply_content,
            'memory_similarity': self.states[-1]['memory_similarity'],
            'emotion': emotion
        }

        return result

    def reinit(self):
        self.history_messages.append(self.messages.copy())
        self.messages = []
        self.conversations = SimpleMemory()

    def get_profile(self) -> dict:
        return self.static_memory.get()

    def get_beliefs(self) -> list:
        return list(self.beliefs)
    
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history_messages, f, ensure_ascii=False, indent=2)

        self.logger.info(f"[✓] User Agent log has been saved to {path}")
