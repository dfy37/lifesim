BASE_QUERY_PROMPT = '''### Requirements
{description}

- Please output your reply in the following JSON format between the ```json and ``` code fences:
```json
{{
    "event": "description of the event"
}}
```
- Based on the user profile and historical events, generate events that match the user profile, do not duplicate historical events, and contain no logical errors.

### Input
[User Profile]
{user_profile}
[User Longterm Goal]
{goal}
[User's Experienced Events]
{event_sequences}
[Current Environment]
{location_desc}
[Output]
'''

RERANK_PROMPT = '''### Requirements
You will be given 9 candidate events. Based on the user profile, longterm goal, previously experienced events, and the current situation, rank the events from most likely to least likely to occur:
- Please select the events that are most likely to happen and output the list of event numbers in descending order of likelihood, in JSON format between ```json and ```. For example: `[3,1,7,2,9,4]`:
```json
{{
    "ranked_events": [x, x, x, x, x, x, x, x, x],
    "has_possible_event": "true/false"
}}
```
- If some events are impossible under the current situation, do not include them in the candidate list.
- Determine whether there is at least one event that could possibly occur in the current environment. If so, set `"has_possible_event"` to `true`; otherwise, set it to `false`.
- `ranked_events` is the reordered list of event numbers, and `has_possible_event` indicates whether any event could possibly occur.
- Avoid including events that are essentially the same as previously experienced ones.
- You may first describe your reasoning process, then provide the JSON output. During the reasoning, consider factors such as the user's preferences, the logical coherence with longterm goal and previous events, and the suitability to the current environment.

### Input
[User Profile]
{user_profile}
[User Longterm Goal]
{goal}
[User's Experienced Events]
{event_sequences}
[Current Environment]
{location_desc}
[Candidate Events]
{events_text}
[Output]
'''

REWRITE_PROMPT = '''You will be given one candidate event and a user intent.
Your task is to revise and refine them so that both align with the user’s profile, long-term goal, and current environmental context, while maintaining internal consistency across the event sequence.
### Requirements
- Adjust details such as subject, location, weather, time, or other contextual factors to make the event realistic and coherent with the given user profile and prior events.
- Ensure the revised event does not contradict any known facts or previous settings.
- The intent should remain essentially the same in meaning but must be expressed naturally and fit the updated event context.
- The intent should represent a single conversational goal (i.e., the user’s focus within one dialogue turn), not a long-term plan.
- Rephrase both the event and intent in a distinct linguistic style from any previously seen phrasing—avoid repetition or mechanical patterns.
- Remove any placeholders or meaningless symbols (e.g., "NAME_1", "XXX", "...").
### Output Format:
Please output your final answer strictly in the following JSON structure (enclosed within ```json and ```):
{{
    "event": "Describe the content of the revised event.",
    "intent": "Describe the user’s corresponding intent under this event context."
}}
Provide your reasoning before the final answer. 
In your reasoning, consider: (1) whether the event and intent satisfy the requirements; (2) whether the intent is realistically something a human would ask an AI assistant.
### Examples:
If the event is “A young woman is jogging in the park,” but the user is male, revise it to “A young man is jogging in the park.”
If the event is “Go shopping at the city-center mall at dusk,” but the event time is 2012-04-05 10:53, revise it to “Go shopping at the city-center mall in the morning.”
If the intent is “The user feels the sun is strong and wants the assistant to give hydration advice,” but the weather is cloudy, revise it to “The user has exercised for a long time and sweated a lot, wants the assistant to give hydration advice.”
If the intent is “The user wants to learn how to study and master a sport (non-e-sports) to improve physical skills or health,” and the user’s long-term goal is “to fully master the fundamental basketball skills—including dribbling, shooting, and passing—through long-term training,” revise it to “The user wants to know how a beginner should start learning the basic skills of basketball.”

### Input
[User Profile]
{user_profile}
[User Longterm Goal]
{goal}
[User's Experienced Events]
{event_sequences}
[Current Environment]
{location_desc}
[Current Event and Intent]
Current user event: {event_text}
Current user intent: {intent}
[Output]
'''

BASE_INFER_GOAL_PROMPT = '''### Requirements
You are an intelligent reasoning model. Based on the user's profile, the sequence of previously experienced events, and the earlier predicted goal, refine your understanding of the user's long-term goal or overarching plan.
Your refined goal should:
- Be specific and actionable, describing a concrete ongoing behavior or habit the user intends to develop (e.g., “improve diet quality,” “build consistent exercise habits”).
- Avoid vague lifestyle statements such as “live healthily,” “stay positive,” or “maintain independence.”
- Reflect a clear pattern of intentional change across the events — the user is trying to do something, not merely be something.
- Stay consistent with the user’s personality and experienced events.
- Be concise (one sentence) and grounded in the observed evidence — do not fabricate details or numbers.

Please output your reasoning process first, then provide the result in the following JSON format between ```json and ```:
```json
{{
    "goal": "xxx"
}}
```
### Examples
Good examples of refined long-term goals include:
{examples}

### Input
[User Profile]
{user_profile}
[User's Experienced Events]
{event_sequences}
[Previously Predicted Goal]
{goal}
[Output]
'''