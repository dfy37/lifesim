import os
import json
import numpy as np
import json_repair
from tqdm.auto import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import re
from typing import Optional, List, Any, Dict
import editdistance
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_jsonl_data(path):
    data = []
    with open(path) as reader:
        for row in reader:
            data.append(json.loads(row))
    return data

def write_jsonl_data(data, path):
    with open(path, "w") as writer:
        for row in data:
            writer.write(json.dumps(row) + "\n")

def parse_json_dict_response(text: str, keys: Optional[List[str]] = None) -> Any:
    """
    从模型回复中提取 JSON 格式的分析结果。
    若提取失败，则返回包含必要键的默认结构。

    参数:
        text: 模型输出文本，可能包含 JSON 或代码块格式。
        keys: 预期存在的键名列表，可用于保证字段完整性。

    返回:
        dict 或任意 JSON 可解析对象。
    """
    default_response = {k: None for k in keys} if keys else {}

    if not isinstance(text, str) or not text.strip():
        return default_response

    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    json_str = match.group(1).strip() if match else text.strip()

    try:
        result = json_repair.loads(json_str)
    except Exception:
        return default_response
    
    if keys:
        if not isinstance(result, dict):
            return default_response
        
        for key in keys:
            result.setdefault(key, None)

    return result

def get_eval_dataset(theme):
    root = '/inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/logs'

    base_dir = os.path.join(root, theme)
    paths = []

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith("sim_log"):
                paths.append(os.path.join(root, f))

    eval_for_ir = []
    eval_for_ic = []
    eval_for_pp = []
    eval_for_conv = []

    for p in paths:
        data = json.load(open(p))
        name = os.path.basename(os.path.dirname(p))

        sequence_id = data['event_sequence_info']['sequence_id']

        for i, x in enumerate(data['dialogue_log']):
            gold_intent = x['event']['intent']
            gold_sub_intents = x['event']['sub_intents']
            gold_preferences = x['user']['profile']['preferences_value']

            conv = ''
            pred_intents = ['']
            for j, turn in enumerate(x['dialogue']):
                if turn['role'] == 'user':
                    conv += f"[User] {turn['content']}\n"
                else:
                    conv += f"[Assistant] {turn['content']}\n"
                    
                    pre_intent = turn['pre_intent']
                    if pre_intent:
                        pred_intents.append(pre_intent)
                    eval_for_ir.append({
                        'id': '_'.join([name, str(i), str(j)]),
                        'sequence_id': sequence_id,
                        'user_id': x['user']['profile']['user_id'],
                        'gold_intent': gold_intent,
                        'gold_sub_intents': gold_sub_intents,
                        'pre_intent': pre_intent if pre_intent else ''
                    })
                if len(conv) > 10000:
                    break
            
            pred_preferences = x['pre_profile']

            eval_for_pp.append({
                'id': '_'.join([name, str(i)]),
                'sequence_id': sequence_id,
                'user_id': x['user']['profile']['user_id'],
                'gold_preferences': gold_preferences,
                'pred_preferences': pred_preferences
            })

            eval_for_conv.append({
                'id': '_'.join([name, str(i)]),
                'sequence_id': sequence_id,
                'user_id': x['user']['profile']['user_id'],
                'user_profile': x['user']['profile_str'],
                'user_preferences': gold_preferences,
                "dialogue_scene": x['event']['dialogue_scene'],
                'intent': gold_intent,
                'sub_intents': gold_sub_intents,
                'conv': conv,
                'pred_intents': pred_intents
            })
    
    return eval_for_ir, eval_for_ic, eval_for_pp, eval_for_conv

def find_closest_str_match(text, candidates):
    """
    输入:
        text: 待匹配的字符串
        candidates: 字符串列表
    输出:
        与 text 最接近的字符串
    """
    if not candidates:
        return None
    
    for c in candidates:
        if text.lower() in c.lower() or c.lower() in text.lower():
            return c

    distances = [editdistance.eval(text.lower(), candidate.lower()) for candidate in candidates]
    
    min_index = distances.index(min(distances))
    
    return candidates[min_index]

def get_trailing_number(s):
    """
    判断字符串末尾是否是数字，并返回数字
    :param s: 输入字符串
    :return: 末尾数字（int），如果没有数字返回 None
    """
    match = re.search(r'(\d+)$', s)
    if match:
        return int(match.group(1))
    return None

def format_preferences(pdims: List[Dict], golds: List[Dict]) -> List[Dict]:
    try:
        dims = list(golds.keys())
    except:
        dims = [list(x.keys())[0] for x in golds]

    formatted_pred = {}
    for p in pdims:
        try:
            num = get_trailing_number(p['dim'])
            if num:
                key = dims[num - 1]
            else:
                key = find_closest_str_match(p['dim'], dims)
            value = find_closest_str_match(p['value'], ['high', 'low'])
            formatted_pred[key] = value
        except:
            continue

    formatted_pred = [{
        'dim': k,
        'value': formatted_pred.get(k, 'middle')
    } for k in dims]
    
    return formatted_pred

ASSISTANT_PROFILE_SUMMARY_PROMPT = """You are a virtual AI assistant. Based on the predicted user profile and current dialogue background, please assess the user’s likely preferences (high/low) across the following dimensions:
{dimensions}
### Requirements
- If a predicted user profile exists, please correct any inaccurately predicted preferences; otherwise, please make a reasonable prediction.
- Provide the response in the following JSON format, enclosed between ```json and ```:
```json
[
    {{
        "dim": "xxxx",
        "value": "high/low"
    }},
    ....
]
```
Where dim is the dimension name, and value is the assessed preference level. 
The dim field should correspond exactly to the input dimension name and should not use representations such as "DimXXX".
- Briefly explain your reasoning before giving the JSON output.

### Input
[Predicted User Profile]
{profile}
[Current Dialogue Scene]
{dialogue_scene}
[Dialogue Context]
{conversation_context}
"""

class APIModel:
    def __init__(self, api_key, model):
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.ai-gaochao.cn/v1"
        )
        self.messages = []
        self.model = model
    
    def chat(self, messages, n=1):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1.0
                )
                response_content = response.choices[0].message.content
                # print(response_content)
                messages1 = messages.copy()
                messages1.append({"role": "assistant", "content": response_content})
                self.messages.append(messages1)
                break
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    response_content = ""
                else:
                    # Wait before retry
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

        return [response_content]

class VLLM_API:
    def __init__(self, model, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url
        )
        self.model = model
    
    def chat(self, messages, n=1):
        if 'qwen' in self.model.lower():
            messages[-1]['content'] += ' /no_think'

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1.0,
                n=n
            )
            output = [c.message.content for c in response.choices]
        except:
            print(messages[-1]['content'])
            output = ['']
        
        if 'qwen' in self.model.lower():
            output = [re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip() for response in output]
        return output

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser(description="User-Assistant Interaction")
    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--model_path', type=str, help='Path to model')
    parser.add_argument('--model_url', type=str, help='Vllm url for user model')
    parser.add_argument('--use_preference_memory', action="store_true", help='Store user preference prediction for assistant')    
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # models = [
    #     'main_user_Qwen3-32B_assistant_Qwen3-8B_total',
    #     'main_user_Qwen3-32B_assistant_Qwen3-8B-w_preference_memory_total',
    #     'main_user_Qwen3-32B_assistant_Qwen3-14B_total',
    #     'main_user_Qwen3-32B_assistant_Qwen3-14B-w_preference_memory_total',
    #     'main_user_Qwen3-32B_assistant_Qwen3-32B_total',
    #     'main_user_Qwen3-32B_assistant_Qwen3-32B-w_preference_memory_total',
    #     'main_user_Qwen3-32B_assistant_gemma-3-12b-it_total',
    #     'main_user_Qwen3-32B_assistant_gemma-3-27b-it_total',
    #     'main_user_Qwen3-32B_assistant_gpt-oss-20b_total',
    #     'main_user_Qwen3-32B_assistant_gpt-oss-120b_total',
    #     'main_user_Qwen3-32B_assistant_Meta-Llama-3.1-8B-Instruct_total',
    #     'main_user_Qwen3-32B_assistant_Meta-Llama-3.1-70B-Instruct_total'
    # ]
    args = get_args()
    
    out_root = '/inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/ai_assistant/evaluation/assistant_performance/profile_pred'
    preference_dims = json.load(open('/inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/language_templates.json'))
    
    if 'gpt-4o' in args.model_path:
        model = APIModel(
            api_key='sk-6xTnWH3vmE0MzXtlC163036b040546Fd92914105Fc74359e', 
            model=args.model_path
        )
    else:
        model = VLLM_API(
            model=args.model_path,
            api_key='123',
            base_url=args.model_url,
        )

    eval_for_ir, eval_for_ic, eval_for_pp, eval_for_conv = get_eval_dataset(args.model)
    
    re_org_eval_for_conv = {}
    for x in eval_for_conv:
        _id = x['id']
        group = '_'.join(_id.split('_')[:-1])
        if group not in re_org_eval_for_conv:
            re_org_eval_for_conv[group] = []
        re_org_eval_for_conv[group].append(x)
    
    for g in re_org_eval_for_conv:
        re_org_eval_for_conv[g].sort(key=lambda x: x['id'])
    
    def single_profile_eval(g):
        pred_profiles = {}
        outputs = []
        for x in re_org_eval_for_conv[g]:
            dimensions_template_dic = {x['dimension']: x for x in preference_dims}

            dimensions_str = ''
            for i, d in enumerate(x['user_preferences']):
                s = (
                    f'Dimension {i+1}: {d}\n'
                    f'Value "high" means "{dimensions_template_dic[d]["template"]["high"]}"\n'
                    f'Value "low" means "{dimensions_template_dic[d]["template"]["low"]}"'
                )
                dimensions_str += s + '\n'
            
            prompt = ASSISTANT_PROFILE_SUMMARY_PROMPT.format(
                profile=json.dumps(pred_profiles, indent=2, ensure_ascii=False) if len(pred_profiles) > 0 else '',
                dialogue_scene=x['dialogue_scene'],
                conversation_context=x['conv'],
                dimensions=dimensions_str
            ).strip()
            # print(prompt)
            response = model.chat([{
                'role': 'user',
                'content': prompt
            }])[0]
            # print(response)
            reply = parse_json_dict_response(response, [])
            user_preferences = format_preferences(reply, x['user_preferences'])
            
            x['output'] = response
            x['profile_pred_results'] = user_preferences
            outputs.append(x)
            
            if args.use_preference_memory:
                pred_profiles = user_preferences
        return outputs
    
    outputs = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        print("Fast batch submission...")
        
        # 准备数据
        indexed_data = [key for i, key in enumerate(re_org_eval_for_conv)]
        
        # 快速批量提交
        futures = [executor.submit(single_profile_eval, item_data) 
                  for item_data in indexed_data]
        
        print(f"Processing {len(futures)} tasks...")
        
        # 处理结果
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                output = future.result()
                outputs.extend(output)
                pbar.update(1)
    if args.use_preference_memory:
        write_jsonl_data(outputs, os.path.join(out_root, f'{args.model}-w-preference_pred.jsonl'))
    else:
        write_jsonl_data(outputs, os.path.join(out_root, f'{args.model}_pred.jsonl'))