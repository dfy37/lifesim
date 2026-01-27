import logging
import json
import json_repair
from datetime import datetime, timedelta, timezone
import editdistance
import re
from typing import Any, List, Dict, Optional

class BeijingFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, timezone.utc)
        beijing_dt = dt + timedelta(hours=8)
        return beijing_dt.timetuple()

def get_logger(file_name, silent: bool = False):
    logger = logging.Logger(file_name, level=logging.INFO)
    if silent:
        logger.addHandler(logging.NullHandler())
    else:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def load_jsonl_data(file_path):
    """
    Load data from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: A list of dictionaries loaded from the JSONL file.
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
    return data

def write_jsonl_data(data, file_path):
    """
    Write data to a JSONL file.

    Args:
        data (list): A list of dictionaries to write.
        file_path (str): Path to the output JSONL file.
    """
    try:
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error writing JSONL file: {e}")

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
            value = find_closest_str_match(p['value'], ['high', 'middle', 'low'])
            formatted_pred[key] = value
        except:
            continue

    formatted_pred = [{
        'dim': k,
        'value': formatted_pred.get(k, 'middle')
    } for k in dims]
    
    return formatted_pred

def preferences2str(preferences: List[Dict]) -> str:
    profile_template = json.load(open('./language_templates.json'))
    profile_template_dic = {x['dimension']: x for x in profile_template}

    preferences_str = ''
    for d in preferences:
        s = profile_template_dic[d['dim']]['template'][d['value']]
        preferences_str += s + '\n'
    return preferences_str
    