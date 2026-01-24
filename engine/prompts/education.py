from engine.prompts.base import BASE_QUERY_PROMPT, BASE_INFER_GOAL_PROMPT

ENV_EVENT_QUERY_DESC = '''Based on the input content, generate an environment-driven event related to learning and education that could occur in this location.
Environment-driven events refer to learning problems or accidents caused by external environmental factors, such as excessive classroom noise, internet disconnection, or insufficient lighting.

Example: If a college student is in the library, a possible event is "The library is very crowded and noisy, which affects the user’s ability to concentrate on studying."'''

ENV_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', ENV_EVENT_QUERY_DESC
)

BIO_EVENT_QUERY_DESC = '''Based on the input content, generate a physiology-driven event related to learning and education that could occur in this location.
Physiology-driven events refer to learning problems caused by internal physical states, such as eye strain, headache, or decreased concentration.

Example: If a middle school student is in the classroom, a possible event is "After staring at the blackboard for a long time, the student’s eyes feel dry and sore, leading to difficulty concentrating."'''

BIO_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', BIO_EVENT_QUERY_DESC
)

COGNITIVE_EVENT_QUERY_DESC = '''Based on the input content, generate a cognition-feedback-driven event related to learning and education that could occur in this location.
Cognition-feedback-driven events refer to learning problems based on personal cognition and psychological states, such as lack of motivation, low learning efficiency, or test anxiety.

Example: If a high school student is in the classroom, a possible event is "Because of poor performance in a mock exam, the user feels very anxious and self-doubtful."'''

COGNITIVE_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', COGNITIVE_EVENT_QUERY_DESC
)

def get_event_dimensions():
    return {
        "environment": ENV_EVENT_QUERY_PROMPT,
        "biological": BIO_EVENT_QUERY_PROMPT,
        "cognitive": COGNITIVE_EVENT_QUERY_PROMPT
    }

LONG_TERM_GOAL_EXAMPLES = '''- Maintain a long-term habit of learning foreign languages—not for exams, but to cross cultural boundaries.
- Make learning a consistent part of professional growth, rather than something pursued only under pressure or crisis.
- Establish a steady learning rhythm, moving away from last-minute cramming and anxiety-driven progress.
- Cultivate deep focus, resisting constant distractions from information overload, and restore learning as a meaningful, high-quality practice.
- Enrich daily life through learning new things, so it’s no longer defined solely by work and responsibilities.
- Complete a personal research project over the next few years, ensuring it delivers genuine value to society.
- Continuously train depth of thinking, moving beyond surface-level understanding when approaching issues.
'''

def get_infer_goal_prompt():
    return BASE_INFER_GOAL_PROMPT.replace(
        '{examples}', LONG_TERM_GOAL_EXAMPLES
    )