from engine.prompts.base import BASE_QUERY_PROMPT, BASE_INFER_GOAL_PROMPT

ENV_EVENT_QUERY_DESC = '''Based on the input content, generate an environment-driven event related to dietary life that could occur in this location.
Environment-driven events refer to dietary problems or accidents caused by external environmental factors, such as a crowded restaurant, inappropriate food temperature, or unclean tableware.

Example: If a person is in a café, a possible event is "The café is crowded and noisy, which affects the user’s enjoyment of food."'''

ENV_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', ENV_EVENT_QUERY_DESC
)

BIO_EVENT_QUERY_DESC = '''Based on the input content, generate a physiology-driven event description related to dietary life that could occur in this location.
Physiology-driven events refer to dietary problems caused by internal physical states, such as hunger, satiety, or allergic reactions to food.

Example: If a person is in a cafeteria, a possible event is "Having not eaten for a long time, the person feels very hungry, which affects normal dining."'''

BIO_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', BIO_EVENT_QUERY_DESC
)

COGNITIVE_EVENT_QUERY_DESC = '''Based on the input content, generate a cognition-feedback-driven event related to dietary life that could occur in this location.
Cognition-feedback-driven events refer to dietary problems based on personal cognition and psychological states, such as awareness of healthy eating, changes in eating habits, or food preferences.

Example: If a person is in a restaurant, a possible event is "Because the user has recently been paying attention to healthy eating, they choose to order only a vegetable salad and skip the main course."'''

COGNITIVE_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', COGNITIVE_EVENT_QUERY_DESC
)

def get_event_dimensions():
    return {
        "environment": ENV_EVENT_QUERY_PROMPT,
        "biological": BIO_EVENT_QUERY_PROMPT,
        "cognitive": COGNITIVE_EVENT_QUERY_PROMPT
    }
    
LONG_TERM_GOAL_EXAMPLES = '''- Find balance through long-term dietary adjustments, making body management an integrated part of daily life rather than a short-term task.
- Develop the habit of eating a varied diet over the next few years, allowing the body to naturally maintain a healthy state.
- Gradually break free from emotional eating, so food is no longer the sole source of comfort.
- Establish regular mealtimes in the coming period, avoiding skipped meals or late-night eating.
- Progressively reduce dependence on sugary drinks and sweets, allowing taste to return to its natural sensitivity.
- Limit oil intake over the next year and cultivate a lasting preference for healthier flavors.
- Maintain a high-protein diet long-term to support recovery and sustain energy.
- Gradually shift toward a primarily plant-based diet, promoting a lighter body and a more sustainable lifestyle.
'''

def get_infer_goal_prompt():
    return BASE_INFER_GOAL_PROMPT.replace(
        '{examples}', LONG_TERM_GOAL_EXAMPLES
    )