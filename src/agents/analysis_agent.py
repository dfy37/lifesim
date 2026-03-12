import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.prompts import (
    USER_CONV_QUALITY_PROMPT,
    ASSISTANT_CONV_QUALITY_PROMPT,
    DROPOUT_PROMPT,
    ACCURACY_EVAL_PROMPT,
    PREFERENCE_ALIGNMENT_PROMPT,
    INTENT_ALIGNMENT_PROMPT,
    CONVERSATION_FLOW_PROMPT,
)
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
            user_utterance=user_utterance,
        )
        response = self.model.chat([{'role': 'user', 'content': prompt}])
        result = parse_json_dict_response(response, keys=['flags', 'advice'])
        result['flags'] = result['flags'] if isinstance(result['flags'], bool) else (result['flags'].lower() == 'true')
        self.records.append({'input': prompt, 'output': response})
        return result

    def assistant_quality_analysis(self, user_profile: str, conversation_context: str,
                                   assistant_utterance: str, event: str, strategy: str = ''):
        prompt = ASSISTANT_CONV_QUALITY_PROMPT.format(
            profile=user_profile,
            dialogue_scene=event,
            conversation_context=conversation_context,
            assistant_utterance=assistant_utterance,
            strategy=strategy,
        )
        response = self.model.chat([{'role': 'user', 'content': prompt}])
        result = parse_json_dict_response(response, keys=['flags', 'advice'])
        result['flags'] = result['flags'] if isinstance(result['flags'], bool) else (result['flags'].lower() == 'true')
        self.records.append({'input': prompt, 'output': response})
        return result

    def _run_llm_analysis(self, context_text: str, template_prompt: str, extra_info: dict = {}) -> str:
        prompt = template_prompt.format(conversation_context=context_text, **extra_info)
        return self.model.chat([{'role': 'user', 'content': prompt}])

    def predict_dropout(self, conversation_context: list, user_profile: dict, event: str, intents: list) -> dict:
        context_text = ''
        for t in conversation_context:
            role = t['role']
            content = t['content']
            if role == 'user':
                context_text += f"{role}: [{t.get('emotion', '')}]{content}\n"
            else:
                context_text += f"{role}: {content}\n"

        def task_accuracy():
            return ('accuracy', self._run_llm_analysis(
                context_text, ACCURACY_EVAL_PROMPT,
                {'dialogue_scene': event, 'evidence': ''}
            ))

        def task_preference():
            return ('preference_alignment', self._run_llm_analysis(
                context_text, PREFERENCE_ALIGNMENT_PROMPT,
                {'profile': user_profile, 'dialogue_scene': event}
            ))

        def task_intent():
            return ('intent_alignment', self._run_llm_analysis(
                context_text, INTENT_ALIGNMENT_PROMPT,
                {'intents': intents, 'dialogue_scene': event}
            ))

        def task_flow():
            return ('conversation_flow', self._run_llm_analysis(context_text, CONVERSATION_FLOW_PROMPT))

        dim_results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(t) for t in [task_accuracy, task_preference, task_intent, task_flow]]
            for future in as_completed(futures):
                dim_name, result = future.result()
                dim_results[dim_name] = result

        self.logger.info(f"[Multi-dimension analysis] {json.dumps(dim_results, ensure_ascii=False)}")

        aggregation_prompt = DROPOUT_PROMPT.format(
            profile=json.dumps(user_profile, ensure_ascii=False),
            dialogue_scene=event,
            conversation_context=context_text,
            analysis=json.dumps(dim_results, ensure_ascii=False),
        )
        final_response = self.model.chat([{'role': 'user', 'content': aggregation_prompt}])
        self.records.append({'input': aggregation_prompt, 'output': final_response})

        result = parse_json_dict_response(final_response, keys=['risk', 'reason', 'strategy'])
        if not result.get('strategy'):
            result['strategy'] = ''
        return result

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'records': self.records}, f, ensure_ascii=False, indent=2)
        self.logger.info(f"[✓] Analysis Agent log saved to {path}")
