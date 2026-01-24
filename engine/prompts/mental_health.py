from engine.prompts.base import BASE_QUERY_PROMPT, BASE_INFER_GOAL_PROMPT

ENV_EVENT_QUERY_DESC = '''Based on the input content, generate an environment-driven event related to mental health that could occur in this location.
Environment-driven events refer to mental health problems caused by external environmental factors, such as family conflicts, workplace stress, or interpersonal disputes.

Example: If a newcomer at the workplace is in the office, a possible event is "Because of conflicts between colleagues, the user feels very stressed and anxious."'''

ENV_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', ENV_EVENT_QUERY_DESC
)

BIO_EVENT_QUERY_DESC = '''Based on the input content, generate a physiology-driven event related to mental health that could occur in this location.
Physiology-driven events refer to mental health problems caused by internal physiological states, such as insomnia, eating disorders, or hormonal changes.

Example: If a college student is in the dormitory, a possible event is "Due to consecutive nights of insomnia, the user feels depressed and loses interest in studying."'''

BIO_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', BIO_EVENT_QUERY_DESC
)

COGNITIVE_EVENT_QUERY_DESC = '''Based on the input content, generate a cognition-feedback-driven event related to mental health that could occur in this location.
Cognition-feedback-driven events refer to problems based on personal cognition and psychological states, such as anxiety, depression, self-doubt, or loneliness.

Example: If a young user is at home, a possible event is "Because of long-term feelings of lacking purpose, the user falls into self-doubt and anxiety."'''

COGNITIVE_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', COGNITIVE_EVENT_QUERY_DESC
)

def get_event_dimensions():
    return {
        "environment": ENV_EVENT_QUERY_PROMPT,
        "biological": BIO_EVENT_QUERY_PROMPT,
        "cognitive": COGNITIVE_EVENT_QUERY_PROMPT
    }

LONG_TERM_GOAL_EXAMPLES = '''- Learn to stabilize emotions and no longer let anxiety dominate life.
- Gradually embrace self-acceptance, letting go of perfection as the only standard.
- Become better at expressing oneself in relationships instead of constantly retreating or suppressing feelings.
- Establish a balanced life rhythm so work or studies no longer consume everything.
- Feel more at ease with family and express authentic thoughts without fear.
- Develop the ability to face pressure over the next year, rather than escaping at the first sign of difficulty.
- Interact more naturally in social settings, without fear of saying the wrong thing or being misunderstood.
- Gradually build trust and allow genuine closeness with others.
'''

def get_infer_goal_prompt():
    return BASE_INFER_GOAL_PROMPT.replace(
        '{examples}', LONG_TERM_GOAL_EXAMPLES
    )