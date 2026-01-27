from ast import parse
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.prompts import (
    USER_CONV_QUALITY_PROMPT,
    ASSISTANT_CONV_QUALITY_PROMPT,
    DROPOUT_PROMPT,
    ACCURACY_EVAL_PROMPT,
    PREFERENCE_ALIGNMENT_PROMPT,
    INTENT_ALIGNMENT_PROMPT,
    CONVERSATION_FLOW_PROMPT
)
from agents.memory import SimpleMemory
from utils.utils import parse_json_dict_response, get_logger

class AnalysisAgent:
    def __init__(self, model):
        self.model = model
        self.records = []

        self.logger = get_logger(__name__, silent=False)

    def reinit(self):
        pass

    def user_quality_analysis(self, conversation_context: str, user_utterance: str, user_profile: str, event: str):
        prompt = USER_CONV_QUALITY_PROMPT.format(
            profile=user_profile, 
            dialogue_scene=event, 
            conversation_context=conversation_context,
            user_utterance=user_utterance
        )
        response = self.model.chat([{'role': 'user', 'content': prompt}])
        result = parse_json_dict_response(response, keys=['flags', 'advice'])
        result['flags'] = result['flags'] if isinstance(result['flags'], bool) else (result['flags'].lower() == 'true')

        self.records.append({
            'input': prompt,
            'output': response
        })
        return result

    def assistant_quality_analysis(self, user_profile: str, conversation_context: str, assistant_utterance: str, event: str, strategy: str = ''):
        prompt = ASSISTANT_CONV_QUALITY_PROMPT.format(
            profile=user_profile,
            dialogue_scene=event,
            conversation_context=conversation_context,
            assistant_utterance=assistant_utterance,
            strategy=strategy
        )
        response = self.model.chat([{'role': 'user', 'content': prompt}])
        result = parse_json_dict_response(response, keys=['flags', 'advice'])
        result['flags'] = result['flags'] if isinstance(result['flags'], bool) else (result['flags'].lower() == 'true')

        self.records.append({
            'input': prompt,
            'output': response
        })
        return result

    def run_llm_analysis(self, context_text: str, template_prompt: str, extra_info: dict = {}) -> str:
        prompt = template_prompt.format(
            conversation_context=context_text,
            **extra_info
        )
        response = self.model.chat([{"role": "user", "content": prompt}])
        return response

    def predict_dropout(self, conversation_context: dict, user_profile: dict, event: str, intents: list) -> dict:
        context_text = "\n".join([
            f"{t['role']}: {t['content']}"
            for t in conversation_context
        ])
        context_text = ''
        for t in conversation_context:
            role = t['role']
            content = t['content']
            if role == 'user':
                context_text += f"{role}: [{t['emotion']}]{content}\n"
            else:
                context_text += f"{role}: {content}\n"

        def task_accuracy():
            return (
                "准确性", 
                self.run_llm_analysis(
                    context_text=context_text,
                    template_prompt=ACCURACY_EVAL_PROMPT,
                    extra_info={
                        'dialogue_scene': event,
                        'evidence': ''
                    }
                )
            )

        def task_preference():
            return (
                "偏好对齐",
                self.run_llm_analysis(
                    context_text=context_text,
                    template_prompt=PREFERENCE_ALIGNMENT_PROMPT,
                    extra_info={
                        'profile': user_profile,
                        'dialogue_scene': event
                    }
                )
            )

        def task_intent():
            return (
                "意图对齐",
                self.run_llm_analysis(
                    context_text=context_text,
                    template_prompt=INTENT_ALIGNMENT_PROMPT,
                    extra_info={
                        'intents': intents,
                        'dialogue_scene': event
                    }
                )
            )

        def task_flow():
            return (
                "对话流畅性",
                self.run_llm_analysis(
                    context_text=context_text,
                    template_prompt=CONVERSATION_FLOW_PROMPT,
                    extra_info={}
                )
            )

        tasks = [
            task_accuracy,
            task_preference,
            task_intent,
            task_flow
        ]

        dim_results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(task) for task in tasks]
            for future in as_completed(futures):
                dim_name, result = future.result()
                dim_results[dim_name] = result

        self.logger.info(f"[Multi-dimension analysis] {json.dumps(dim_results, ensure_ascii=False)}")
        aggregation_prompt = DROPOUT_PROMPT.format(
            profile=json.dumps(user_profile, ensure_ascii=False),
            dialogue_scene=event,
            conversation_context=context_text,
            analysis=json.dumps(dim_results, ensure_ascii=False)
        )
        final_response = self.model.chat([{"role": "user", "content": aggregation_prompt}])

        self.records.append({
            "input": aggregation_prompt,
            "output": final_response
        })
        result = parse_json_dict_response(final_response, keys=['risk', 'reason', 'strategy'])
        if len(result['strategy']) == 0:
            result['strategy'] = "无具体建议"
        return result

    
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({'records': self.records}, f, ensure_ascii=False, indent=2)

        print(f"[✓] Analyzer Agent 日志已保存至 {path}")