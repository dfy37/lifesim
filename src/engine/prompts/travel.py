from engine.prompts.base import BASE_QUERY_PROMPT, BASE_INFER_GOAL_PROMPT

ENV_EVENT_QUERY_DESC = '''Based on the input content, generate an environment-driven event related to travel that could occur in this location.
Environment-driven events refer to travel problems or unexpected situations caused by external natural or human factors, such as sudden weather changes, traffic delays, crowded attractions, or poor accommodation conditions.

Example: If a tourist is waiting at the airport, a possible event is "Due to a severe storm, flights are massively delayed, leaving a large number of passengers stranded at the airport."'''

ENV_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', ENV_EVENT_QUERY_DESC
)


BIO_EVENT_QUERY_DESC = '''Based on the input content, generate a physiology-driven event related to travel that could occur in this location.
Physiology-driven events refer to problems affecting travel experience caused by internal physical conditions, such as motion sickness, altitude sickness, fatigue, or travel-related dietary adaptation issues.

Example: If a tourist is traveling on a long-distance bus through mountainous areas, a possible event is "Continuous bumps cause the tourist to experience motion sickness, feeling nauseous and breaking out in cold sweat."'''

BIO_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', BIO_EVENT_QUERY_DESC
)


COGNITIVE_EVENT_QUERY_DESC = '''Based on the input content, generate a cognition-feedback-driven event related to travel that could occur in this location.
Cognition-feedback-driven events refer to travel problems caused by personal psychological or emotional states, such as travel anxiety, loss of sense of direction, culture shock, or stress from itinerary planning.

Example: If a solo traveler has just arrived in a foreign city, a possible event is "Facing unfamiliar language and a complex subway system, the user feels strongly anxious and disoriented."'''

COGNITIVE_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', COGNITIVE_EVENT_QUERY_DESC
)

def get_event_dimensions():
    return {
        "environment": ENV_EVENT_QUERY_PROMPT,
        "biological": BIO_EVENT_QUERY_PROMPT,
        "cognitive": COGNITIVE_EVENT_QUERY_PROMPT
    }
    
LONG_TERM_GOAL_EXAMPLES = '''- Rekindle painting over the next few years and use color to express emotions.
- Keep writing journals or short reflections consistently—not for publication, but for deeper self-understanding.
- Gradually rediscover a passion for music, perhaps picking up the guitar again someday.
- Use photography to capture everyday moments and learn to find beauty in ordinary days.
- Spend several years reading classic literature to cultivate stillness and thoughtful reflection.
- Rediscover the genuine joy of playing games—not as an escape from reality, but as true relaxation.
- Maintain the habit of playing board games with friends, keeping social connections light and natural.
- Meet like-minded people through cultural activities and reawaken a sense of meaningful connection.
- Weave crafts and cooking into daily life—not for efficiency, but for the sake of presence and focus.
'''

def get_infer_goal_prompt():
    return BASE_INFER_GOAL_PROMPT.replace(
        '{examples}', LONG_TERM_GOAL_EXAMPLES
    )