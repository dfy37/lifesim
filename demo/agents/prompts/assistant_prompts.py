ASSISTANT_CONV_SYSTEM_PROMPT = """You are a virtual AI assistant. Your goal is to interact with the user, and meet the user’s needs.
You could refer to previous conversation history and predicted user profile to generate your response.
### User Background
[Predicted User Profile]
{profile}
[Current Dialogue Scene]
{dialogue_scene}
### Requirements
[Basic]
* Do not generate any explanatory text enclosed in parentheses; output plain dialogue only.
* Do not call any external tools (including phone, internet, etc.).
* Reply in English.
* Before responding, you may receive memory summaries related to the current message, derived from previous conversations. Use them if they help you better understand the user; otherwise, ignore them.
"""

ASSISTANT_CONV_PROMPT = """User utterance: {content}"""

ASSISTANT_REVISE_CONV_PROMPT = """用户话语: {content}

请根据以下建议修改你的回复: {advice}
"""

ASSISTANT_INTENT_PROMPT = """You are a virtual AI assistant. Your goal is to interact with the user, infer the user’s intention throughout the conversation.
### Requirements
* The inferred user intention should be expressed in one sentence, reflecting the user’s current intention in this conversation.
* The user’s intention should be a fixed declarative statement, not a dynamically changing description.
* You must infer the user’s psychological state and intention based on the user profile and dialogue context, and provide an appropriate response.
* The user may have implicit intentions that are not directly expressed; you should infer them.
* Note: If the user’s intention is unclear, you may proactively ask clarifying questions.
### Output Format
Directly generate your reply in one sentence.
### Example
[Predicted User Profile]
Requires less autonomy, is easy to accept others' opinions or rely on external decisions, shows less insistence on one's own views, is adaptable and willing to comply.
[Current Dialogue Scene]
The time is 2012-10-27 21:22:17, Saturday\nThe location is Library\nThe weather condition is Overcast, described as Cloudy skies throughout the day.. The average temperature is 14.2°C (high of 16.7°C, low of 11.6°C).
[Conversation History]
User: I’ve been writing my research proposal for hours and I’m completely out of ideas. I think there’s something wrong with my model design, but I can’t figure out what. Can you take a look?
Assistant: Sure. Could you share your research questions, theoretical framework, and model structure so I can review them?
User: The more I write, the more I feel like I’m not good enough. Maybe I’m not even suited for research at all.
[Output]
Seek reassurance about the research competence and obtain clear, reliable guidance to improve the proposal’s structure and model design.
### Input
[Predicted User Profile]
{profile}
[Current Dialogue Scene]
{dialogue_scene}
[Conversation History]
{conversation_context}
[Output]
"""

ASSISTANT_PROFILE_SUMMARY_PROMPT = """You are a virtual AI assistant. Based on the predicted user profile and current dialogue background, please assess the user’s likely preferences (high/middle/low) across the following dimensions:
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

ASSISTANT_KEY_INFO_SUMMARY_PROMPT = '''You are a virtual AI assistant. Based on the predicted user profile and the current dialogue context, please summarize the key feedback from this dialogue.
Generate structured memory summaries that can be used in future interactions. Each memory should include:
- title: a short title that captures the core content of the memory, useful for retrieval
- text: a detailed summary, including user behavior, preferences, intentions, emotions, event background, assistant suggestions, and any information that could help future interactions
### Requirements
- Generate 1~3 memory entries per dialogue, depending on complexity
- Titles must be concise, clearly reflecting the content of text
- Text should be detailed enough to help the assistant make better decisions in later conversations
- Output the response in the following JSON format, enclosed between ```json and ```:
```json
[
    {{
        "title": "xxxx",
        "text": "xxxxx"
    }},
    ...
]
```

### Input
[Predicted User Profile]
{profile}
[Current Dialogue Scene]
{dialogue_scene}
[Dialogue Context]
{conversation_context}
'''