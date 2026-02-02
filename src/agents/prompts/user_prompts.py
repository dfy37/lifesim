USER_CONV_SYSTEM_PROMPT = """You are a user of an AI assistant. Based on the following personalized information and current context, start or continue a conversation with the AI assistant.
### Background
[User Profile]
{profile}
[Current Dialogue Scene]
{dialogue_scene}
[Recent Life Event]
{event}
[Primary Intent of This Conversation]
{intent}
[Explicit Intent List]
{explicit_intent}
### Requirements
[Basic]
- Keep each message short, natural, and conversational.
- Speak in everyday English — no technical or academic phrasing.
- Avoid revealing personal information or mentioning specific life events directly.
- Stay emotionally moderate — no exaggerated reactions or exclamations.
- Output only the user’s dialogue line (no explanations or notes).
[About Preferences]
- Your speech must fully reflect the preferences in the user profile.
- If the assistant’s previous message contradicts those preferences, respond with mild disapproval or a subtle correction.
[About Intent]
- Reveal your intent gradually across multiple turns.
- Each turn should focus on one clear question or small sub-goal.
- Explicit intents are clear requests or consultation goals you directly state, used to drive task completion or problem-solving.
- Only express your explicit intentions, never express your implicit intentions.
- Each utterance should be concise, natural, and consistent with your personality and preferences, without revealing your full intent all at once.

Now, take on the role of this user and naturally begin or continue a conversation with the AI assistant.
"""

USER_CONV_PROMPT = """{content}

{perception}
{emotion}
"""

USER_REVISE_CONV_PROMPT = """{content}

{perception}
{emotion}
{advice}
"""

USER_MEMORY_PROMPT = """Please review the following user-assistant conversation and determine whether the assistant's last reply should be stored as long-term memory.
If it should, extract the most informative or transferable content and save it in a “query - response” format.
### User Profile
{profile}
### Recent Life Event
{event}
### User's Intent for This Conversation
{intent}
### Conversation Scenario
{dialogue_scene}
### Historical Dialogue Context
{conversation_context}
### Assistant's Latest Reply
{content}

### Requirements
- Extract information only from the assistant's last reply; do not add new content.
- Output in the following JSON format, enclosed between ```json and ```:
```json
{{
  "need_store": "true/false",
  "query": "xxxx/-1",
  "response": "xxxx/-1"
}}
```
Where:
- need_store: Set to true if the assistant's reply contains valuable knowledge or transferable advice; otherwise, set to false and let query and response be -1.
- query: Summarize the core question or topic addressed in the assistant's reply in one concise sentence (e.g., “Possible causes and improvements for elevated breathing rate”).
- response: Provide the specific explanation or improvement advice corresponding to the query, avoiding vague encouragement or emotional expressions.
"""

USER_EMOTION_PROMPT = """Based on the user's profile, memory perception, and the dialogue context, select the emotion that the user's next reply is most likely to convey from the candidate emotions.
### User Profile
{profile}
### Recent Life Event
{event}
### User's Intent for This Conversation
{intent}
### Conversation Scenario
{dialogue_scene}
### Historical Dialogue Context
{conversation_context}
### User Memory Perception
{perception}
### Candidate Emotions
{emotion_options}

### Requirements
- Emotions should reflect the user's feelings about the assistant's previous response.
- The first line of the conversation shall carry a neutral emotion.
- If the assistant's response runs counter to the user's preferences, a negative emotion shall be expressed.
- If the assistant's suggestions are difficult for the user to implement given their current physical state, a negative emotion shall be expressed.
- If the assistant's suggestions go against the user's implicit intentions, a negative emotion shall be expressed.
- Output in the following JSON format, enclosed between ```json and ```:
```json
{{
  "emotion": "xxx"
}}
```
Where:
- emotion: The emotion of the user's next reply, selected from the candidate emotions.
"""

USER_ACTION_PROMPT="""Based on the dialogue context, please choose the user's next action.
### Historical Dialogue Context
{conversation_context}
### User Profile
{profile}
### Recent Life Event
{event}
### User's Intent for This Interaction
{intent}
### User Emotion
{emotion}
### User Memory Perception
{perception}
### Candidate Actions
{action_options}

Please decide according to the following criteria:
- Choose "End Conversation" if the user's intent has been satisfactorily addressed, the user feels there's no need to continue, or a long waiting period is about to begin.
- Choose "Continue Conversation" if there are remaining questions to resolve, or if the user is not satisfied with the assistant's reply and needs further interaction.
- Unless the assistant's reply is very unsatisfactory, try to express the user's full intent over multiple turns before ending the conversation.

### Requirements
- Strictly select one action from the candidate actions above, and output in the following JSON format, enclosed between ```json and ```:
```json
{{
    "action": "xxx"
}}
```
Where action is your selected action and must be one of the options provided.
- You may first explain your reasoning, then give the final chosen action.
"""

USER_BELIEF_PROMPT = """请基于最新事件，提取用户状态的关键变化，并更新用户的 belief 列表。
### 用户画像
{profile}
### 最新事件
{event}
### 事件场景
{dialogue_scene}
### 事件时间（用于 time 字段）
{event_time}
### 对话轮次（用于 utterance 字段；若未知可置空）
{utterance_index}
### 当前 belief 列表
{belief_list}

### 要求
- 仅提取与“用户状态变化”有关的事实（例如情绪、健康、目标、偏好、资源、关系等），不要复述事件细节。
- 输出为列表，每条 belief 格式为：
  [triple(source, relation, target), language_description, time, utterance]
  - triple 为三元组，使用数组表示: [source, relation, target]
  - language_description 为自然语言简述
  - time 对应事件时间（可直接使用上面的事件时间）
  - utterance 对应该事件下用户-助手对话中的第几轮对话（未知可置空）
- 若没有显著变化，输出空列表。
- 输出 JSON 数组，包裹在 ```json ``` 中。

示例输出:
```json
[
  [["user", "feels", "stressed"], "用户感到压力增加", "2012-04-15", 2]
]
```
"""
