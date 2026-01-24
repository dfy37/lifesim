from engine.prompts.base import BASE_QUERY_PROMPT, BASE_INFER_GOAL_PROMPT

ENV_EVENT_QUERY_DESC = '''Based on the input content, generate an environment-driven event related to leisure and entertainment that could occur in this location.
Environment-driven events refer to problems or accidents in the entertainment experience caused by external environmental factors, such as overcrowded venues, sound system failures, or lighting that is too dim.

Example: If a person is in a movie theater, a possible event is "The theater is overcrowded and the seats are too cramped, leading to a poor movie-watching experience."'''

ENV_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', ENV_EVENT_QUERY_DESC
)

BIO_EVENT_QUERY_DESC = '''Based on the input content, generate a physiology-driven event related to leisure and entertainment that could occur in this location.
Physiology-driven events refer to problems in the entertainment experience caused by an individual’s physical condition, such as fatigue, dizziness, or physical discomfort.

Example: If a person is singing in a karaoke room, a possible event is "Singing for too long causes a hoarse throat, affecting the ability to continue enjoying the entertainment."'''

BIO_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', BIO_EVENT_QUERY_DESC
)

COGNITIVE_EVENT_QUERY_DESC = '''Based on the input content, generate a cognition-feedback-driven event related to leisure and entertainment that could occur in this location.
Cognition-feedback-driven events refer to problems in the entertainment experience caused by personal psychological and cognitive states, such as lack of interest, insufficient engagement, or anxiety.

Example: If a person is in an amusement park, a possible event is "Seeing that the waiting time for the roller coaster is too long, the user feels impatient and bored."'''

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