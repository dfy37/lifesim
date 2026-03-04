import math
import random
from typing import Optional

import json_repair
import numpy as np
from pydantic import Field

from logger import get_logger
from utils.prompt import FormatPrompt
from utils.context import DotDict
from ...memory import Memory
from ...agent.dispatcher import BlockDispatcher
from ..sharing_params import SocietyAgentBlockOutput
from .utils import clean_json_response

# Prompt templates for LLM interactions
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

PLACE_TYPE_SELECTION_PROMPT = """
As an intelligent decision system, please determine the type of place the user needs to visit based on their input requirement.
User Plan: {plan}
User requirement: {intention}
Other information: 
-------------------------
{other_info}
-------------------------
Your output must be a single selection from {poi_category} without any additional text or explanation.

Please response in json format (Do not return any other text), example:
{{
    "place_type": "shopping"
}}
"""

PLACE_SECOND_TYPE_SELECTION_PROMPT = """
As an intelligent decision system, please determine the type of place the user needs to visit based on their input requirement.
User Plan: {plan}
User requirement: {intention}
Other information: 
-------------------------
{other_info}
-------------------------

Your output must be a single selection from {poi_category} without any additional text or explanation.

Please response in json format (Do not return any other text), example:
{{
    "place_type": "shopping"
}}
"""

PLACE_ANALYSIS_PROMPT = """
As an intelligent analysis system, please determine the type of place the user needs to visit based on their input requirement.
User Plan: {plan}
User requirement: {intention}
Other information: 
-------------------------
{other_info}
-------------------------

Your output must be a single selection from {place_list} without any additional text or explanation.

Please response in json format (Do not return any other text), example:
{{
    "place_type": "home"
}}
"""

RADIUS_PROMPT = """As an intelligent decision system, please determine the maximum travel radius (in meters) based on the current emotional state.

Current weather: ${context.weather}
Current temperature: ${context.temperature}
Your current emotion: ${context.current_emotion}
Your current thought: ${context.current_thought}
Other information: 
-------------------------
${context.other_information}
-------------------------

Please analyze how these emotions would affect travel willingness and return only a single integer number between 3000-200000 representing the maximum travel radius in meters. A more positive emotional state generally leads to greater willingness to travel further.

Please response in json format (Do not return any other text), example:
{{
    "radius": 10000
}}
"""


def gravity_model(pois):
    """
    Calculate selection probabilities for POIs using a gravity model.

    The model considers both distance decay (prefer closer locations)
    and spatial density (avoid overcrowded areas). Distances are grouped
    into 1km bins up to 10km, with POIs beyond 10km in a 'more' category.

    Args:
        pois: List of POI tuples containing (poi_data, distance)

    Returns:
        List of tuples: (name, id, normalized_weight, distance)
        with selection probabilities based on gravity model
    """
    # Initialize distance bins
    pois_Dis = {f"{d}k": [] for d in range(1, 11)}
    pois_Dis["more"] = []

    # Categorize POIs into distance bins
    for poi in pois:
        classified = False
        for d in range(1, 11):
            if (d - 1) * 1000 <= poi[1] < d * 1000:
                pois_Dis[f"{d}k"].append(poi)
                classified = True
                break
        if not classified:
            pois_Dis["more"].append(poi)

    res = []
    distanceProb = []
    # Calculate weights for each POI
    for poi in pois:
        for d in range(1, 11):
            if (d - 1) * 1000 <= poi[1] < d * 1000:
                n = len(pois_Dis[f"{d}k"])
                # Calculate ring area between (d-1)km and d km
                S = math.pi * ((d * 1000) ** 2 - ((d - 1) * 1000) ** 2)
                density = n / S  # POIs per square meter
                distance = max(poi[1], 1)  # Avoid division by zero

                # Inverse square distance decay combined with density
                weight = density / (distance**2)
                res.append((poi[0]["name"], poi[0]["id"], weight, distance))
                distanceProb.append(1 / math.sqrt(distance))
                break

    # Normalize probabilities and sample
    distanceProb = np.array(distanceProb)
    distanceProb /= distanceProb.sum()

    # Randomly sample 50 candidates weighted by distance probabilities
    sample_indices = np.random.choice(len(res), size=50, p=distanceProb)
    sampled_pois = [res[i] for i in sample_indices]

    # Normalize weights for final selection
    total_weight = sum(item[2] for item in sampled_pois)
    return [
        (item[0], item[1], item[2] / total_weight, item[3]) for item in sampled_pois
    ]


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
        if keys:
            for key in keys:
                if key not in result:
                    result[key] = None

        return result

    except Exception as e:
        # 若失败，返回默认内容以防界面报错
        if keys:
            response = {k: None for k in keys}
        else:
            response = {}
        return response


class PlaceSelectionBlock:
    """
    Block for selecting destinations based on user intention.

    Implements a two-stage selection process:
    1. Select primary POI category (e.g., 'shopping')
    2. Select sub-category (e.g., 'bookstore')
    Uses LLM for decision making with fallback to random selection.

    Configurable Fields:
        search_limit: Max number of POIs to retrieve from map service
    """

    name = "PlaceSelectionBlock"
    description = "Selects destinations for unknown locations (excluding home/work)"

    def __init__(
        self,
        model,
        search_limit: int = 50
    ):
        self.typeSelectionPrompt = FormatPrompt(PLACE_TYPE_SELECTION_PROMPT)
        self.secondTypeSelectionPrompt = FormatPrompt(
            PLACE_SECOND_TYPE_SELECTION_PROMPT
        )
        self.radiusPrompt = FormatPrompt(
            RADIUS_PROMPT
        )
        self.search_limit = search_limit  # Default config value
        self.model = model

    def forward(self, environment, current_pos, context: DotDict):
        """Execute the destination selection workflow"""
        # Stage 1: Select primary POI category
        poi_cate = environment.get_poi_cate()
        self.typeSelectionPrompt.format(
            plan=context["plan_context"]["plan"],
            intention=context["current_step"]["intention"],
            poi_category=list(poi_cate.keys()),
            # TODO : 待修改
            other_info=environment.environment.get("other_information", "None"),
        )
        try:
            # LLM-based category selection
            levelOneType = self.model.chat(
                self.typeSelectionPrompt.to_dialog()
            )
            levelOneType = parse_json_dict_response(levelOneType)["place_type"] # type: ignore
            sub_category = poi_cate[levelOneType]
        except Exception as e:
            get_logger().warning(f"Level 1 selection failed: {e}")
            levelOneType = random.choice(list(poi_cate.keys()))
            sub_category = poi_cate[levelOneType]

        # Stage 2: Select sub-category
        try:
            self.secondTypeSelectionPrompt.format(
                plan=context["plan_context"]["plan"],
                intention=context["current_step"]["intention"],
                poi_category=sub_category,
                other_info=environment.environment.get(
                    "other_information", "None"
                ),
            )
            levelTwoType = self.model.chat(
                self.secondTypeSelectionPrompt.to_dialog()
            )
            levelTwoType = parse_json_dict_response(levelTwoType)["place_type"] # type: ignore
        except Exception as e:
            get_logger().warning(f"Level 2 selection failed: {e}")
            levelTwoType = random.choice(sub_category)

        # Get travel radius from LLM
        try:
            self.radiusPrompt.format(context=context)
            radius = self.model.chat(
                self.radiusPrompt.to_dialog()
            )
            radius = int(parse_json_dict_response(radius)["radius"]) # type: ignore
        except Exception as e:
            get_logger().warning(f"Radius selection failed: {e}")
            radius = 10000  # Default 10km

        # Query and select POI
        center = (current_pos["x"], current_pos["y"])
        pois = environment.map.query_pois(
            center=center,
            category_prefix=levelTwoType,
            radius=radius,
            limit=self.search_limit,
        )

        if pois:
            pois = gravity_model(pois)
            probabilities = [item[2] for item in pois]
            selected = np.random.choice(len(pois), p=probabilities)
            next_place = (pois[selected][0], pois[selected][1])
        else:  # Fallback random selection
            all_pois = environment.map.get_all_pois()
            next_place = random.choice(all_pois)
            next_place = (next_place["name"], next_place["id"])

        return next_place


class MoveBlock:
    """Block for executing mobility operations (home/work/other)"""

    name = "MoveBlock"
    description = "Executes mobility operations between locations"

    def __init__(self, model, memory):
        self.placeAnalysisPrompt = FormatPrompt(PLACE_ANALYSIS_PROMPT)
        self.model = model
        self.memory = memory

    def forward(self, environment, context: DotDict):
        """判断用户要去的地点类型"""
        place_knowledge = self.memory.status.get("location_knowledge", {})
        known_places = list(place_knowledge.keys())
        places = ["home", "workplace"] + known_places + ["other"]
        
        self.placeAnalysisPrompt.format(
            plan=context["plan_context"]["plan"],
            intention=context["current_step"]["intention"],
            place_list=places,
            other_info=environment.environment.get("other_information", "None"),
        )
        
        response = self.model.chat(
            self.placeAnalysisPrompt.to_dialog()
        )
        
        try:
            place_type = parse_json_dict_response(response, ["place_type"])["place_type"]
        except Exception:
            get_logger().warning(
                f"Place Analysis: wrong type of place, raw response: {response}"
            )
            place_type = "home"
        
        return place_type, known_places, place_knowledge

    def get_destination_id(self, place_type: str, known_places: list, place_knowledge: dict, environment, next_place=None):
        """根据地点类型获取具体的目的地ID"""
        if place_type == "home":
            # go back home
            home = self.memory.status.get("home")
            if home:
                return home["aoi_position"]["aoi_id"]
            else:
                get_logger().warning("Home location not found in memory")
                return None
                
        elif place_type == "workplace":
            # back to workplace
            work = self.memory.status.get("work")
            if work:
                return work["aoi_position"]["aoi_id"]
            else:
                get_logger().warning("Workplace location not found in memory")
                return None

        elif place_type in known_places:
            # go to known place
            the_place = place_knowledge[place_type]["id"]
            return the_place
            
        else:
            # move to other places
            if next_place is None:
                # 随机选择一个地点作为fallback
                aois = environment.map.get_all_aois()
                while True:
                    r_aoi = random.choice(aois)
                    if len(r_aoi["poi_ids"]) > 0:
                        r_poi = random.choice(r_aoi["poi_ids"])
                        break
                poi = environment.map.get_poi(r_poi)
                next_place = (poi["name"], poi["aoi_id"])
            
            return next_place[1]


class MobilityNoneBlock:
    """Placeholder block for mobility operations that don't require location changes"""
    
    name = "MobilityNoneBlock"
    description = "Handles mobility operations that don't require movement"
    
    def __init__(self, model, memory):
        self.model = model
        self.memory = memory
    
    def forward(self, environment, context: DotDict):
        """Handle non-movement mobility operations"""
        # 这里可以处理一些不需要移动的操作，比如等待、休息等
        get_logger().info("Processing mobility_none operation")
        return "no_movement_required"


class MobilityBlockParams:
    # PlaceSelection
    radius_prompt: str = Field(
        default=RADIUS_PROMPT, description="Used to determine the maximum travel radius"
    )
    search_limit: int = Field(
        default=50, description="Number of POIs to retrieve from map service"
    )


class MobilityBlockContext:
    next_place: Optional[tuple[str, int]] = Field(
        default=None, description="The next place to go"
    )


class MobilityBlock:
    """
    Main mobility coordination block.
    """

    ParamsType = MobilityBlockParams
    OutputType = SocietyAgentBlockOutput
    ContextType = MobilityBlockContext
    name = "MobilityBlock"
    description = "Used for moving like go to work, go to home, go to other places, etc."
    actions = {
        "place_selection": "Support the place selection action",
        "move": "Support the move action",
        "mobility_none": "Support other mobility operations",
    }

    def __init__(
        self,
        toolbox,
        agent_memory: Memory,
        block_params: Optional[MobilityBlockParams] = None,
    ):
        self.toolbox = toolbox
        self.agent_memory = agent_memory
        self.params = block_params or MobilityBlockParams()
        self.model = toolbox  # 假设toolbox包含了模型接口
        
        # initialize all blocks
        self.place_selection_block = PlaceSelectionBlock(
            self.model, self.params.search_limit
        )
        self.move_block = MoveBlock(self.model, agent_memory)
        self.mobility_none_block = MobilityNoneBlock(self.model, agent_memory)
        self.trigger_time = 0  # Block invocation counter
        self.token_consumption = 0  # LLM token tracker

    def generate_daily_preferences(self, profile: str) -> str:
        """
        生成个人日常偏好 (α)
        Args:
            profile: 用户画像信息
        Returns:
            生成的日常偏好描述
        """
        prompt_alpha = DAILY_PREFERENCE_PROMPT.format(profile=profile)

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
        prompt_beta = LIFE_ANCHOR_PROMPT.format(profile=profile) 
        
        messages = [{'role': 'user', 'content': prompt_beta}]
        response = self.model.chat(messages)
        # 解析JSON格式的响应
        response = parse_json_dict_response(response, keys=['content'])
        return response['content']

    def forward(self, profile, environment, current_pos, context, n_POIs=50):
        """
        Main entry point - 重构后的主要逻辑
        
        Args:
            profile: 用户画像
            environment: 环境对象
            current_pos: 当前位置
            context: 上下文信息
        """
        
        # 1. 生成静态上下文：个人偏好和生活锚点
        print("生成个人偏好...")
        preferences = self.generate_daily_preferences(profile)
        print(f"个人偏好: {preferences}")
        
        print("生成生活锚点...")
        anchor_points = self.generate_life_anchor_points(profile)  
        print(f"生活锚点: {anchor_points}")

        POIs = []
        for _ in range(n_POIs):
            # 2. 使用MoveBlock判断地点类型
            print("判断目的地类型...")
            place_type, known_places, place_knowledge = self.move_block.forward(preferences, anchor_points)
            print(f"地点类型: {place_type}")
            
            

            # 3. 根据地点类型决定下一步操作
            if place_type == "other":
                # 如果是其他地方，使用PlaceSelectionBlock进行详细选择
                print("选择具体地点...")
                next_place = self.place_selection_block.forward(environment, current_pos, context)
                print(f"选择的地点: {next_place}")
                
                # 获取目的地ID
                destination_id = self.move_block.get_destination_id(
                    place_type, known_places, place_knowledge, environment, next_place
                )
            else:
                # 如果是home、workplace或已知地点，直接获取目的地ID
                print(f"前往{place_type}...")
                destination_id = self.move_block.get_destination_id(
                    place_type, known_places, place_knowledge, environment
                )
                next_place = None
            POIs.append(destination_id)
        result = {
            "preferences": preferences,
            "anchor_points": anchor_points,
            "pois": POIs
        }
        return result