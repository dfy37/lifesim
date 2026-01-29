from engine.prompts.base import BASE_QUERY_PROMPT, BASE_INFER_GOAL_PROMPT

ENV_EVENT_QUERY_DESC = '''Based on the input content, generate an environment-driven event related to sports and health that could occur in this location.
Environment-driven events refer to accidents or problems caused by external environmental factors.'''

ENV_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', ENV_EVENT_QUERY_DESC
)

BIO_EVENT_QUERY_DESC = '''Based on the input content, generate a physiology-driven event related to sports and health that could occur in this location.
Physiology-driven events refer to problems caused by internal physical states.'''

BIO_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', BIO_EVENT_QUERY_DESC
)

COGNITIVE_EVENT_QUERY_DESC = '''Based on the input content, generate a cognition-feedback-driven event related to sports and health that could occur in this location.
Cognition-feedback-driven events refer to problems based on personal cognition and psychological states.'''

COGNITIVE_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', COGNITIVE_EVENT_QUERY_DESC
)

def get_event_dimensions():
    return {
        "environment": ENV_EVENT_QUERY_PROMPT,
        "biological": BIO_EVENT_QUERY_PROMPT,
        "cognitive": COGNITIVE_EVENT_QUERY_PROMPT
    }

LONG_TERM_GOAL_EXAMPLES = '''- Become stronger and more resilient by next summer, without constantly suffering from back pain.
- Make exercise a natural part of daily life so that fitness improves gradually instead of relying on restrictive dieting.
- Maintain a consistent exercise routine to restore a healthy metabolism and stop worrying about gaining weight from every meal.
- Fully master the fundamental basketball skills—dribbling, shooting, and passing—through long-term training.
- Through sustained badminton practice, become proficient in both forehand and backhand shots as well as net control, instead of relying on blind swings.
- Improve endurance and footwork through continuous training so that I can complete a full tennis match without exhaustion.
- Master the technical details of both freestyle and breaststroke through long-term swimming practice.
- Achieve significant weight loss through regular exercise and proper dietary control.
'''

def get_infer_goal_prompt():
    return BASE_INFER_GOAL_PROMPT.replace(
        '{examples}', LONG_TERM_GOAL_EXAMPLES
    )