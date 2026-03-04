import json
import json_repair
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from engine.map_const import POI_CATG_DICT

DAILY_PREFERENCE_PROMPT = '''根据以下用户画像，请生成他们的日常偏好和习惯：
画像：{profile}

请描述他们的日常偏好，包括：
- 喜欢的起床和睡觉时间
- 饮食偏好和进食习惯
- 工作风格和偏好
- 喜欢的休闲活动
- 运动和健康习惯
- 社交互动偏好
请按照如下JSON格式生成一份全面的日常偏好描述：
```json
{{
    "content": "xxx"
}}
```
'''

LIFE_ANCHOR_PROMPT = '''根据以下用户画像，请识别他们的生活锚点和日常规律：
画像：{profile}

请识别他们生活中的关键锚点和规律，包括：
- 固定的工作时间或任务
- 规律的用餐时间
- 运动习惯
- 社交安排
- 个人护理习惯
- 周末活动
- 其他任何重复性的活动
请按照如下JSON格式生成一份关于他们主要生活锚点和规律的描述:
```json
{{
    "content": "xxx"
}}
```
'''

DYNAMIC_PERCEIVED_PROMPT = '''根据用户画像、偏好、生活锚点以及当前意图序列，
请评估此人下一步最有可能采取的行动及其原因。

用户画像：{profile}
- 日常偏好：{preferences}
- 生活锚点：{anchor_points}
{sequence_text}

可用的意图类型：{intent_candidates}

请分析：
- 现在的时间是什么？到目前为止已经做了哪些事情？
- 根据此人的画像和习惯，接下来最有可能的活动是什么？
- 考虑工作时间安排、用餐时间、精力状态、社交义务等因素。
排出最有可能的前三个下一步意图，并说明理由。
请按照如下JSON格式提供对下一步意图的感知可能性的详细分析：
```json
{{
    "content": "xxx"
}}
```
'''

NEXT_INTENT_PROMPT = '''基于所提供的全部信息，请为此人决定下一步的意图。  

日常偏好：{preferences}  
生活锚点：{anchor_points}  

{sequence_text}  

感知可能性分析：{perceived_likelihood}  
可用的意图类型：{intent_candidates}  

请从可用意图类型中选择最合适的下一步意图。 按照如下json格式返回：
```json
{{
    "content": "xxx"
}}
``` 
'''

NEXT_TIME_PROMPT = '''根据当前的活动序列和下一步计划的意图，请确定合适的开始和结束时间。

{sequence_text}

下一步意图：{next_intention}
日常偏好：{preferences}
生活锚点：{anchor_points}
上一次活动结束时间：{last_end_time}

请考虑：
1. 此类活动的合理持续时间
2. 与上一活动的自然衔接
3. 该人的习惯和偏好
4. 此类活动的典型时间安排

请按照如下JSON格式返回下一个意图的开始和结束时间：
```json
{{
    "start_time": "HH:MM",
    "end_time": "HH:MM"
}}
```
'''

@dataclass
class IntentionItem:
    """意图项：包含时间范围和意图描述"""
    start_time: str
    end_time: str  
    intention: str
    
    def __str__(self):
        return f'["{self.start_time}, {self.end_time}", "{self.intention}"]'

def parse_json_dict_response(text: str, keys: list = None) -> dict:
    import re
    
    """
    从模型回复中提取 JSON 格式的分析结果。
    如果提取失败，则返回一个默认结构。
    """
    try:
        # 匹配 JSON 格式的代码块
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)

        json_str = match.group(1).strip()
        result = json_repair.loads(json_str)

        # 验证必要字段存在
        for key in keys:
            if key not in result:
                result[key] = None

        return result

    except Exception as e:
        # 若失败，返回默认内容以防界面报错
        response = {k: None for k in keys}
        return response

class IntentionSequenceGenerator:
    """基于LLM和计划行为理论的人类意图序列生成器"""
    
    def __init__(self, model):
        """
        初始化生成器
        Args:
            model: LLM模型，需要有chat()方法
        """
        self.model = model
        
        # 预定义的10种意图类型
        self.intention_types = POI_CATG_DICT
        
        # 基于真实数据的日常意图数量分布 (简化版本)
        self.daily_intention_count_distribution = [6, 7, 8, 9, 10, 11, 12]
        self.daily_intention_count_weights = [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]
    
    def generate_daily_preferences(self, profile: str) -> str:
        """
        生成个人日常偏好 (α)
        Args:
            profile: 用户画像信息
        Returns:
            生成的日常偏好描述
        """
        prompt_alpha = DAILY_PREFERENCE_PROMPT.format(profile=user_profile_str)

        messages = [{'role': 'user', 'content': prompt_alpha}]
        response = self.model.chat(messages)
        # 解析JSON格式的响应
        response = parse_json_dict_response(response, keys=['content'])
        return response['content']
    
    def generate_life_anchor_points(self, profile: str) -> str:
        """
        生成生活锚点/例行程序 (β)
        Args:
            profile: 用户画像信息
        Returns:
            生成的生活锚点描述
        """
        prompt_beta = LIFE_ANCHOR_PROMPT.format(profile=user_profile_str) 
        
        messages = [{'role': 'user', 'content': prompt_beta}]
        response = self.model.chat(messages)
        # 解析JSON格式的响应
        response = parse_json_dict_response(response, keys=['content'])
        return response['content']
    
    def generate_dynamic_perceived_likelihood(
        self, 
        profile: Dict[str, Any],
        current_sequence: List[IntentionItem],
        preferences: str,
        anchor_points: str
    ) -> str:
        """
        生成下一个意图的动态感知可能性 (γi)
        Args:
            profile: 用户画像信息
            current_sequence: 当前已生成的意图序列
            preferences: 个人偏好
            anchor_points: 生活锚点
        Returns:
            下一个意图的感知可能性评估
        """
        
        # 构建当前序列的文本表示
        sequence_text = "Current intention sequence:\n"
        for item in current_sequence:
            sequence_text += f"  {item}\n"
        
        if not current_sequence:
            sequence_text += "  (Empty - starting the day)\n"
        
        prompt_gamma = DYNAMIC_PERCEIVED_PROMPT.format(
            profile=json.dumps(profile, ensure_ascii=False, indent=2),
            preferences=preferences,
            anchor_points=anchor_points,
            sequence_text=sequence_text,
            intent_candidates=', '.join(self.intention_types)
        )
        
        messages = [{'role': 'user', 'content': prompt_gamma}]
        response = self.model.chat(messages)
        # 解析JSON格式的响应
        response = parse_json_dict_response(response, keys=['content'])
        return response['content']
    
    def generate_next_intention(
        self,
        current_sequence: List[IntentionItem],
        preferences: str,
        anchor_points: str, 
        perceived_likelihood: str
    ) -> str:
        """
        基于所有信息生成下一个意图 (ei+1)
        Args:
            current_sequence: 当前意图序列
            preferences: 个人偏好
            anchor_points: 生活锚点
            perceived_likelihood: 感知可能性评估
        Returns:
            下一个意图类型
        """
        
        sequence_text = "Current sequence:\n"
        for item in current_sequence:
            sequence_text += f"  {item}\n"
            
        prompt_intention = NEXT_INTENT_PROMPT.format(
            preferences=preferences,
            anchor_points=anchor_points,
            sequence_text=sequence_text,
            perceived_likelihood=perceived_likelihood,
            intent_candidates=', '.join(self.intention_types)
        )
        
        messages = [{'role': 'user', 'content': prompt_intention}]
        response = self.model.chat(messages)
        # 解析JSON格式的响应
        response = parse_json_dict_response(response, keys=['content'])['content']
        
        # 确保返回的是有效的意图类型
        if response not in self.intention_types:
            # 如果返回的不是预定义类型，尝试匹配最相似的
            response = self._match_closest_intention(response)
            
        return response
    
    def generate_time_slot(
        self,
        current_sequence: List[IntentionItem],
        next_intention: str,
        preferences: str,
        anchor_points: str
    ) -> Tuple[str, str]:
        """
        为下一个意图生成时间段 (ti+1)
        Args:
            current_sequence: 当前序列
            next_intention: 下一个意图
            preferences: 个人偏好
            anchor_points: 生活锚点
        Returns:
            (开始时间, 结束时间)元组
        """
        
        last_end_time = "00:00"
        if current_sequence:
            last_end_time = current_sequence[-1].end_time
            
        sequence_text = "Current sequence:\n"
        for item in current_sequence:
            sequence_text += f"  {item}\n"
        
        prompt_time = NEXT_TIME_PROMPT.format(
            sequence_text=sequence_text,
            next_intention=next_intention,
            preferences=preferences,
            anchor_points=anchor_points,
            last_end_time=last_end_time
        )
        
        messages = [{'role': 'user', 'content': prompt_time}]
        response = self.model.chat(messages)
        response = parse_json_dict_response(response, keys=['start_time', 'end_time'])
        
        # 解析时间
        try:
            start_time, end_time = response['start_time'], response['end_time']
            return start_time.strip(), end_time.strip()
        except:
            # 如果解析失败，提供默认值
            return self._generate_default_time_slot(last_end_time, next_intention)
    
    def generate_intention_sequence(
        self, 
        profile: Dict[str, Any],
        in_context_examples: List[str] = None
    ) -> List[IntentionItem]:
        """
        生成完整的日常意图序列
        Args:
            profile: 用户画像信息
            in_context_examples: 上下文示例 (可选)
        Returns:
            生成的意图序列
        """
        
        print("开始生成意图序列...")
        
        # 1. 采样今日意图总数
        total_intentions = random.choices(
            self.daily_intention_count_distribution,
            weights=self.daily_intention_count_weights
        )[0]
        print(f"计划生成 {total_intentions} 个意图")
        
        # 2. 生成静态上下文：个人偏好和生活锚点
        print("生成个人偏好...")
        preferences = self.generate_daily_preferences(profile)
        print(f"个人偏好: {preferences[:100]}...")
        
        print("生成生活锚点...")
        anchor_points = self.generate_life_anchor_points(profile)  
        print(f"生活锚点: {anchor_points[:100]}...")
        
        # 3. 逐步生成意图序列
        intention_sequence = []
        
        for i in range(total_intentions):
            print(f"\n生成第 {i+1}/{total_intentions} 个意图...")
            
            # 生成动态感知可能性
            perceived_likelihood = self.generate_dynamic_perceived_likelihood(
                profile, intention_sequence, preferences, anchor_points
            )
            print(f"感知可能性分析: {perceived_likelihood[:100]}...")
            
            # 生成下一个意图
            next_intention = self.generate_next_intention(
                intention_sequence, preferences, anchor_points, perceived_likelihood
            )
            print(f"下一个意图: {next_intention}")
            
            # 生成时间段
            start_time, end_time = self.generate_time_slot(
                intention_sequence, next_intention, preferences, anchor_points
            )
            print(f"时间段: {start_time} - {end_time}")
            
            # 添加到序列
            intention_item = IntentionItem(start_time, end_time, next_intention)
            intention_sequence.append(intention_item)
        
        print("\n意图序列生成完成!")
        return intention_sequence
    
    def _match_closest_intention(self, response: str) -> str:
        """匹配最相似的预定义意图类型"""
        response_lower = response.lower()
        for intention in self.intention_types:
            if intention in response_lower or any(word in response_lower for word in intention.split()):
                return intention
        # 如果都不匹配，返回默认值
        return "handle the trivialities of life"
    
    def _generate_default_time_slot(self, last_end_time: str, intention: str) -> Tuple[str, str]:
        """生成默认时间段"""
        # 简化的默认时间生成逻辑
        hour, minute = map(int, last_end_time.split(':'))
        start_hour = hour
        start_minute = minute + 15  # 15分钟后开始
        
        if start_minute >= 60:
            start_hour += 1
            start_minute -= 60
            
        if start_hour >= 24:
            start_hour = 23
            start_minute = 45
            
        # 根据意图类型设定持续时间
        duration_map = {
            "sleep": 8*60,
            "go to work": 8*60, 
            "eat": 60,
            "do shopping": 90,
            "do sports": 120,
            "excursion": 4*60,
            "leisure or entertainment": 2*60,
            "medical treatment": 90,
            "handle the trivialities of life": 30,
            "go home": 30
        }
        
        duration = duration_map.get(intention, 60)  # 默认1小时
        end_minute = start_minute + duration
        end_hour = start_hour + end_minute // 60
        end_minute = end_minute % 60
        
        if end_hour >= 24:
            end_hour = 23
            end_minute = 59
            
        return f"{start_hour:02d}:{start_minute:02d}", f"{end_hour:02d}:{end_minute:02d}"


# 示例使用
class MockLLM:
    """模拟LLM，用于演示"""
    def chat(self, prompt):
        prompt = prompt[0]['content']
        if "日常偏好" in prompt.lower():
            return '```json\n{"content": "This person prefers early mornings, healthy eating, regular exercise, and values work-life balance."}```'
        elif "生活锚点" in prompt.lower():
            return '```json\n{"content": "Fixed work schedule 9-5, regular meal times at 7am/12pm/7pm, evening exercise routine."}```'
        elif "前三个下一步意图" in prompt.lower():
            return '```json\n{"content": "Based on current time and sequence, most likely next activities are: 1) eat (meal time), 2) go to work (if morning), 3) leisure (if evening)."}```'
        elif "最合适的下一步意图" in prompt.lower():
            return '```json\n{"content": "eat"}```'  
        elif "开始和结束时间" in prompt.lower():
            return '```json\n{"start_time": "08:00", "end_time": "08:45"}```'
        else:
            return "Generated response"

def main():
    # 创建模拟模型和生成器
    mock_model = MockLLM()
    generator = IntentionSequenceGenerator(mock_model)
    
    # 示例用户画像
    user_profile = {
        "age": 28,
        "occupation": "software_engineer", 
        "work_schedule": "9-5_weekdays",
        "living_situation": "urban_apartment",
        "health_status": "healthy",
        "social_level": "moderate",
        "income_level": "middle_class"
    }

    user_profile_str = json.dumps(user_profile, ensure_ascii=False, indent=2)
    
    # 生成意图序列
    sequence = generator.generate_intention_sequence(user_profile_str)
    
    # 输出结果
    print("\n=== 生成的日常意图序列 ===")
    for i, item in enumerate(sequence):
        print(f"{i+1}. {item}")

if __name__ == "__main__":
    main()