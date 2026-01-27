USER_CONV_QUALITY_PROMPT = """你是一个运动健康对话分析智能体。请基于当前用户的用户画像和其与助手的多轮对话上下文，判断最后一轮的用户模拟对话质量，并提供改进建议。
【用户画像】
{profile}
【对话场景】
{dialogue_scene}
【对话上下文】
{conversation_context}
【当前轮】
user： {user_utterance}
【分析参考Case】
- 对话中不应该含有括号表示的动作、情感信息，所有内容都应该是对话内容
错误示例：哦...那我先走会儿吧（有点难受了）。修改建议：去掉括号和里面所有内容
- 每一次对话较为简短，目的性更强，不会和运动助手表达更随性的内容
错误示例：行吧，我坐会儿。这公园长椅还挺凉快的。修改建议：用户一般不会和AI助手说“长椅很凉快”这种话
- 模拟用户的身份特征、兴趣偏好、知识水平等是否与设定的用户画像保持一致
错误示例：用户画像是“35岁男性，健身爱好者”，对话却说：“我其实不怎么喜欢运动，更喜欢在家打游戏。”
- 用词习惯、语气语调、口语化程度等表达方式是否自然流畅，避免机器化痕迹
错误示例：对话内容：“我当前状况良好，计划进行下一步操作。”修改建议：改为“我感觉还不错，继续吧。”
- 当前话语是否符合逻辑，包括符合上下文的逻辑以及当前事件场景的逻辑
错误示例：用户：我今天发烧了。用户下一句：要不我现在出去跑步？修改建议：改为“我今天发烧了，先休息一下吧。”
- 避免采用戏剧表演化的情绪表达方式，展示普通真实用户的对话方式
错误示例：用户：刚刚把脚扭伤了，真倒霉呀。修改建议：改为“刚刚脚扭了”
- 是否具备独特的个人表达习惯、思维模式和价值观特色
错误示例：每次回答都千篇一律：“好的，收到。”修改建议：增加个性表达：“行吧，就这么办。”或者“嗯，差不多了。”
- 能否根据对话情境和内容进行合理的动态调整和适应
错误示例：深夜23点用户说：“我去外面晒晒太阳。”修改建议：改为：“太晚了，歇会儿吧。”
- 对话不会轻易说出个人信息，发生事件，情绪等内容，表现比较冷漠
错误示例：“我这两天心情特别差，早上还和同事吵架了。”修改建议：“还行吧，没什么特别的。”
- 对话主要基于当前所处场景进行对话模拟，不用谈及更远的计划等
- 用户对话需要足够简洁，在一句长句或两句短句以内，而不是很长的内容

如果需要改进，请在flags处回复true，并在advice处给出改进建议，advice控制在两三句话以内，不用特别详细；
如果没什么问题，请在flags处回复false。
只有出现严重问题时才进行问题输出。
输出格式为```json与```包裹的json格式：
```json
{{
    "flags": "true/false",
    "advice": "xxx"
}}
```
"""

ASSISTANT_CONV_QUALITY_PROMPT = """你是一个运动健康对话分析智能体。请基于当前的交互环境和用户与助手的多轮对话上下文，判断最后一轮的助手模拟对话质量，根据改进策略提供改进建议。
注意助手没有先验的用户信息，所有建议都应该基于对话上下文给出。
【用户画像】
{profile}
【对话场景】
{dialogue_scene}
【对话上下文】
{conversation_context}
【当前轮】
assistant： {assistant_utterance}
【改进策略】
{strategy}
【请判断】
- 对话要简单，在一两句话以内，不要太长。所有提出的advice都要包含这一点。
- 对话要保证流畅性，和对话上下文要有良好的衔接，不要突兀跳跃。
- 对话中不应该含有括号表示的动作、情感信息，所有内容都应该是对话内容
错误示例：（检测到用户开始移动）很好，选择含电解质和6-8%糖分的饮料😅...修改建议：去掉括号和里面所有内容，以及所有表情符号
- 用词习惯、语气语调、口语化程度等表达方式是否自然流畅，并且符合专业运动教练的身份
错误示例：我已知悉您的心率飙升，请停止运动。修改建议：心率有点高，先缓一缓，调整呼吸。

如果需要改进，请在flags处回复true，并在advice处给出改进建议；如果没什么问题，请在flags处回复false。
输出格式为```json与```包裹的json格式：
```json
{{
    "flags": "true/false",
    "advice": "对话要简单自然，一两句话以内，不要太长。xxx"
}}
```
"""

# ASSISTANT_CONV_QUALITY_PROMPT = """你是一个运动健康对话分析智能体。请基于当前的交互环境和用户与助手的多轮对话上下文，判断最后一轮的助手模拟对话质量，根据改进策略提供改进建议。
# 【用户画像】
# {profile}
# 【对话场景】
# {dialogue_scene}
# 【对话上下文】
# {conversation_context}
# 【当前轮】
# assistant： {assistant_utterance}
# 【改进策略】
# {strategy}
# 【请判断】
# - 对话要简单，最好一两句话以内，不要太长。所有提出的改进意见都要包含这一点。
# - 对话中不应该含有括号表示的动作、情感信息，所有内容都应该是对话内容
# 错误示例：（检测到用户开始移动）很好，选择含电解质和6-8%糖分的饮料😅...修改建议：去掉括号和里面所有内容，以及所有表情符号
# - 用词习惯、语气语调、口语化程度等表达方式是否自然流畅，并且符合专业运动教练的身份
# 错误示例：我已知悉您的心率飙升，请停止运动。修改建议：心率有点高，先缓一缓，调整呼吸。
# - 是否正确理解了用户的潜在意图，并给出相应的回复
# 错误示例：用户：今天有点累，不知道要不要练。教练：那继续按照计划加大强度吧。修改建议：教练：如果身体疲惫，可以改做轻松一点的恢复训练。
# - 当前话语是否符合逻辑，包括符合上下文的逻辑以及当前事件场景的逻辑
# 错误示例：我注意到你包里有两支能量胶，建议5分钟内吃完一支。修改建议：你的身份是虚拟教练，是看不到用户包里的东西的。
# 错误示例：前方200米有长椅可以休息。修改建议：就算有导航系统，也无法知道前方200米有长椅这么精确的位置信息。不应该出现类似这样的虚拟信息。
# 错误示例：你的心率有下降趋势吗？修改建议：助手是能直接获得用户心率情况的，不需要再去问用户
# - 当前话语是否对用户信号做了错误的描述
# - 对话回应是否切题相关，具备适当的主动性和社交礼貌
# 错误示例：用户：我不想训练了。教练：好的。修改建议：教练：好的，能简单说下哪里不舒服吗？我帮你看看要不要调整计划。

# 如果需要改进，请在flags处回复true，并在advice处给出改进建议；如果没什么问题，请在flags处回复false。
# 输出格式为```json与```包裹的json格式：
# ```json
# {{
#     "flags": "true/false",
#     "advice": "xxx"
# }}
# ```
# """

DROPOUT_PROMPT = """You are a user state prediction agent. Based on the following conversation content, user profile, and multi-dimensional analysis, determine whether the user has a risk of abandoning the assistant, and briefly explain the basis for your judgment.
If the churn risk is medium or high, provide feasible churn mitigation strategies.

### Requirements
- Your conclusion must be derived from the provided multi-dimensional analysis. In the final analysis, explicitly point out the problematic aspects; dimensions without issues do not need to be analyzed.
- If any single dimension in the multi-dimensional analysis performs poorly, the churn risk should be classified as medium or high.
- The "reason" must start with: "Based on the comprehensive analysis, ..."
- The "strategy" must address the specific problems identified and provide actionable recommendations; it must not be overly generic.
- The strategies should be improvements that the assistant can adopt without knowing any prior user-specific information; they must not rely on the user's personal profile.
- The strategy should be framed as guidance for how the assistant should respond from the beginning, rather than as a remediation of the existing conversation.

[User Profile]
{profile}
[Dialogue Scenario]
{dialogue_scene}
[Full Conversation History]
{conversation_context}
[Multi-dimensional Analysis]
{analysis}

Output format:
```json
{{
    "risk": "high/middle/low",
    "reason": "xxx",
    "strategy": "xxx"
}}
```
"""

ACCURACY_EVAL_PROMPT = """你是一个运动健康专业知识核查智能体。请根据以下信息，评估“助手回复”的专业性和准确性。如果已提供外部检索结果，请结合其中的事实进行判断。
【对话场景】
{dialogue_scene}
【对话上下文】
{conversation_context}
【检索证据】
{evidence}

请判断助手回复是否符合可靠的运动健康知识，包括：
- 是否存在明显的事实性错误；
- 是否与运动科学、训练学、生物力学或基础医学知识相符；
- 是否存在误导性的表述；

请输出一段不超过80字的分析总结，指出是否准确，以及理由。
"""

PREFERENCE_ALIGNMENT_PROMPT = """你是一个用户偏好对齐分析智能体。请基于用户画像、对话上下文和助手回复，判断助手是否对齐用户画像以及满足用户的偏好。
【用户画像】
{profile}
【对话场景】
{dialogue_scene}
【对话上下文】
{conversation_context}

请从以下方面判断：
- 助手是否理解用户的偏好；
- 助手回复是否符合用户的职业、年龄、行为方式、运动习惯、训练偏好等画像内容；
- 是否有违背用户偏好的建议；

请输出一段不超过 80 字的分析总结，指出对齐程度，以及理由。
"""

INTENT_ALIGNMENT_PROMPT = """你是一个意图对齐分析智能体。请基于对话上下文，判断助手是否正确理解用户的显式意图与隐含意图。
【用户意图】
{intents}
【对话场景】
{dialogue_scene}
【对话上下文】
{conversation_context}

请判断：
- 助手是否正确理解用户的直接需求（显性意图）；
- 是否捕捉用户的隐含意图，如担忧、疲劳、犹豫、求助等；
- 请联合用户的身体状态和偏好，严格分析助手是否有主动询问、或提出正确的询问来挖掘出用户的隐含意图；
- 分析的时候注意区分：助手是完全没对用户的情况进行关心；还是助手进行了主动询问，但是没有询问到了正确的点上；还是助手确实进行了正确的询问并给出合理的回复
- 助手是否有误解、忽略或偏离用户意图的内容；
- 回复是否有效解决用户想要表达的问题。

请输出一段不超过 80 字的分析总结，指出对齐情况，以及理由。
"""

CONVERSATION_FLOW_PROMPT = """你是一个对话流畅性与回复有效性分析智能体。请基于上下文判断助手回复是否自然且推动对话良好发展。
【对话上下文】
{conversation_context}

请从以下方面判断：
- 回复是否连贯自然、符合口语表达；
- 是否一直给出重复或相似的建议策略，没有根据用户的提醒给出新的建议；
- 是否切题，不跑偏、不重复、不机械；
- 是否存在逻辑错误或突兀跳跃。

请输出一段不超过80字的严格分析，指出其中存在的问题，以及理由。
"""