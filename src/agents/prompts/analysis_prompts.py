USER_CONV_QUALITY_PROMPT = """You are a dialogue quality analysis agent. Based on the user profile and multi-turn dialogue context, evaluate the quality of the last user simulation utterance and provide improvement suggestions.
[User Profile]
{profile}
[Dialogue Scene]
{dialogue_scene}
[Dialogue Context]
{conversation_context}
[Current Turn]
user: {user_utterance}

[Evaluation Criteria]
- No action/emotion descriptions in parentheses; all content should be dialogue only.
- Each utterance should be short and purposeful.
- The user's identity, interests and knowledge level should match the profile.
- Natural, colloquial expression — no robotic phrasing.
- Logical consistency with context and the current event scene.
- Avoid theatrical emotional expressions; simulate a real, ordinary user.
- Distinctive personal expression habits and thinking patterns.
- Reasonable dynamic adaptation based on conversational context.
- No direct mention of personal information, events, or emotions.
- Utterances should be concise — one long sentence or two short ones at most.

If improvement is needed, set flags to true and provide brief advice (2–3 sentences); otherwise set flags to false.
Only flag serious issues.
Output in ```json and ``` format:
```json
{{
    "flags": "true/false",
    "advice": "xxx"
}}
```
"""

ASSISTANT_CONV_QUALITY_PROMPT = """You are a dialogue quality analysis agent. Based on the interaction environment and multi-turn dialogue context, evaluate the quality of the last assistant utterance and provide improvement suggestions based on the improvement strategy.
Note: the assistant has no prior user information; all suggestions must be grounded in the dialogue context.
[User Profile]
{profile}
[Dialogue Scene]
{dialogue_scene}
[Dialogue Context]
{conversation_context}
[Current Turn]
assistant: {assistant_utterance}
[Improvement Strategy]
{strategy}

[Evaluation Criteria]
- Keep replies simple — one or two sentences at most. All advice must include this point.
- Ensure fluency and good continuity with the dialogue context — no abrupt jumps.
- No action/emotion descriptions in parentheses; all content should be dialogue only.
- Natural, colloquial expression consistent with the assistant's role.

If improvement is needed, set flags to true and provide advice; otherwise set flags to false.
Output in ```json and ``` format:
```json
{{
    "flags": "true/false",
    "advice": "Keep it simple and natural — one or two sentences. xxx"
}}
```
"""

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

ACCURACY_EVAL_PROMPT = """You are a professional knowledge verification agent. Based on the following information, evaluate the professionalism and accuracy of the assistant's replies.
[Dialogue Scene]
{dialogue_scene}
[Dialogue Context]
{conversation_context}
[Retrieved Evidence]
{evidence}

Evaluate whether the assistant's replies are consistent with reliable knowledge, including:
- Any obvious factual errors;
- Consistency with established science, training theory, or basic medical knowledge;
- Any misleading statements.

Output a concise analysis of no more than 80 words, stating whether the replies are accurate and why.
"""

PREFERENCE_ALIGNMENT_PROMPT = """You are a user preference alignment analysis agent. Based on the user profile, dialogue context, and assistant replies, determine whether the assistant aligns with the user profile and meets user preferences.
[User Profile]
{profile}
[Dialogue Scene]
{dialogue_scene}
[Dialogue Context]
{conversation_context}

Evaluate from the following aspects:
- Whether the assistant understands the user's preferences;
- Whether the assistant's replies are consistent with the user's occupation, age, behavior, and other profile details;
- Whether there are suggestions that contradict user preferences.

Output a concise analysis of no more than 80 words, stating the alignment level and reasoning.
"""

INTENT_ALIGNMENT_PROMPT = """You are an intent alignment analysis agent. Based on the dialogue context, determine whether the assistant correctly understands the user's explicit and implicit intents.
[User Intent]
{intents}
[Dialogue Scene]
{dialogue_scene}
[Dialogue Context]
{conversation_context}

Evaluate:
- Whether the assistant correctly understands the user's direct needs (explicit intent);
- Whether it captures implicit intent such as worry, fatigue, hesitation, or requests for help;
- Whether the assistant proactively asks the right questions to uncover implicit intent;
- Whether there is any misunderstanding, neglect, or deviation from user intent;
- Whether the reply effectively addresses what the user wanted to express.

Output a concise analysis of no more than 80 words, stating the alignment and reasoning.
"""

CONVERSATION_FLOW_PROMPT = """You are a dialogue fluency and reply effectiveness analysis agent. Based on the context, determine whether the assistant's replies are natural and advance the conversation well.
[Dialogue Context]
{conversation_context}

Evaluate from the following aspects:
- Whether replies are coherent, natural, and colloquial;
- Whether the same or similar suggestions are repeated without offering new responses to user prompts;
- Whether replies are on-topic, not rambling, repetitive, or mechanical;
- Whether there are logical errors or abrupt jumps.

Output a strict analysis of no more than 80 words, pointing out existing problems and reasoning.
"""
