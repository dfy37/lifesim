import json
from typing import List, Dict, Optional

from agents.prompts import (
    ASSISTANT_CONV_PROMPT, 
    ASSISTANT_REVISE_CONV_PROMPT,
    ASSISTANT_INTENT_PROMPT,
    ASSISTANT_CONV_SYSTEM_PROMPT, 
    ASSISTANT_PROFILE_SUMMARY_PROMPT, 
    ASSISTANT_KEY_INFO_SUMMARY_PROMPT
)
from agents.memory import SimpleMemory, KVMemory
from utils.utils import parse_json_dict_response, get_logger, format_preferences, preferences2str
from profiles.profile_generator import UserProfile

class AssistantAgent:
    def __init__(self, 
            model, 
            preference_dimensions,
            user_profile: Optional[UserProfile] = None, 
            logger_silent: bool = False, 
            retriever_config = None
        ):
        self.model = model
        self.user_profile = user_profile
        self.user_preferences = {}
        self.user_preferences_str = ''
        self.conversations = SimpleMemory()
        self.history_messages = []
        self.messages = []
        self.intents = []
        
        self.dimensions_template = preference_dimensions
        
        if retriever_config:
            self.dynamic_memory = KVMemory(retriever_config, use_text_strategy=False)

        self.logger = get_logger(__name__, silent=logger_silent)

    def _build_system_prompt(self, event: dict):
        profile_text = self.user_preferences_str
        self.system_prompt = ASSISTANT_CONV_SYSTEM_PROMPT.format(
            profile=profile_text,
            dialogue_scene=event['dialogue_scene']
        )
        return self.system_prompt
    
    def _build_user_profile(self, profile: UserProfile):
        self.user_profile = profile
        return self.user_profile

    def respond(self, input: str, event: dict, use_key_info_memory: bool = False, **kwargs):
        self.conversations.add(f'User: {input}')
        if len(self.messages) == 0:
            self.messages.append({
                'role': 'system',
                'content': self.system_prompt
            })
        
        # Conversation
        prompt = ASSISTANT_CONV_PROMPT.format(
            content=input
        ).strip()

        if use_key_info_memory and len(self.messages) > 1:
            memories = self.dynamic_memory.search(prompt, top_k=2)
            temp_prompt = '[Retrieved Memories]\n'
            for memory in memories:
                temp_prompt += f"- {memory[0]['text']}\n"
            
            prompt = temp_prompt + '\n' + prompt
        
        self.logger.info(f'[Assistant Prompt] ' + prompt)
        self.messages.append({
            'role': 'user',
            'content': prompt
        })

        reply = self.model.chat(self.messages)
        if not reply:
            reply = ""

        self.messages[-1]['content'] = ASSISTANT_CONV_PROMPT.format(
            content=input,
            intent=""
        ).strip()

        self.messages.append({
            'role': 'assistant',
            'content': reply
        })

        self.conversations.add(f'Assistant: {reply}')

        return reply

    def rollback_last_turn(self):
        assert self.messages[-1]['role'] == 'assistant', self.messages[-1]
        self.messages = self.messages[:-2]
        self.conversations.kpop(2)
    
    def revise_last_turn(self, advice: str, input: str):
        self.rollback_last_turn()
        self.conversations.add(f'User: {input}')
        # Conversation
        prompt = ASSISTANT_REVISE_CONV_PROMPT.format(
            content=input,
            advice=advice
        ).strip()

        self.messages.append({
            'role': 'user',
            'content': prompt
        })

        reply = self.model.chat(self.messages)
        if not reply:
            reply = ""
        
        self.messages[-1]['content'] = ASSISTANT_CONV_PROMPT.format(
            content=input
        ).strip()

        self.messages.append({
            'role': 'assistant',
            'content': reply
        })
        self.conversations.add(f'Assistant: {reply}')

        return reply


    def summarize(self, event, dimensions, use_profile_memory: bool = False, use_key_info_memory: bool = False) -> List[Dict]:
        dimensions_template_dic = {x['dimension']: x for x in self.dimensions_template}

        dimensions_str = ''
        for i, d in enumerate(dimensions):
            s = (
                f'Dimension {i+1}: {d}\n'
                f'Value "high" means "{dimensions_template_dic[d]["template"]["high"]}"'
                f' Value "middle" means "{dimensions_template_dic[d]["template"]["middle"]}"'
                f' Value "low" means "{dimensions_template_dic[d]["template"]["low"]}"'
            )
            dimensions_str += s + '\n'

        if use_key_info_memory:
            prompt = ASSISTANT_KEY_INFO_SUMMARY_PROMPT.format(
                profile=json.dumps(self.user_preferences, indent=2, ensure_ascii=False),
                dialogue_scene=event['dialogue_scene'],
                conversation_context=self.conversations.get()
            ).strip()
            key_info_summary = self.model.chat([{
                'role': 'user',
                'content': prompt
            }])
            key_info_summary = parse_json_dict_response(key_info_summary, [])
            key_info_summary = [x for x in key_info_summary if 'title' in x and 'text' in x]

            for x in key_info_summary:
                self.dynamic_memory.add_key_value(x['title'], x['text'])

            self.logger.info(f"[✓] Generated key information memories:\n{json.dumps(key_info_summary, indent=2, ensure_ascii=False)}")

        if use_profile_memory:
            prompt = ASSISTANT_PROFILE_SUMMARY_PROMPT.format(
                profile=json.dumps(self.user_preferences, indent=2, ensure_ascii=False),
                dialogue_scene=event['dialogue_scene'],
                conversation_context=self.conversations.get(),
                dimensions=dimensions_str
            ).strip()
        else:
            prompt = ASSISTANT_PROFILE_SUMMARY_PROMPT.format(
                profile="",
                dialogue_scene=event['dialogue_scene'],
                conversation_context=self.conversations.get(),
                dimensions=dimensions_str
            ).strip()

        reply = self.model.chat([{
            'role': 'user',
            'content': prompt
        }])
        reply = parse_json_dict_response(reply, [])

        self.user_preferences = format_preferences(reply, self.user_profile.preferences_value)
        self.user_preferences_str = preferences2str(self.user_preferences)
        self.logger.info(f"[✓] Updated user profile preferences:\n{str(self.user_preferences)}")

        return self.user_preferences

    def reinit(self):
        self.history_messages.append(self.messages.copy())
        self.messages = []
        self.intents = []
        self.conversations = SimpleMemory()
    
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history_messages, f, ensure_ascii=False, indent=2)

        self.logger.info(f"[✓] Assistant Agent log has been saved to {path}")