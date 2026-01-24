from engine.prompts.base import BASE_QUERY_PROMPT, BASE_INFER_GOAL_PROMPT

ENV_EVENT_QUERY_DESC = '''Based on the input content, generate an environment-driven event description related to elderly care that could occur in this location.
Environment-driven events refer to elderly care problems or accidents caused by external environmental factors, such as slippery floors leading to falls, excessive noise affecting rest, or elevator malfunctions.

Example: If an elderly person is walking in the corridor of a nursing home, a possible event is "The corridor is dimly lit, making it difficult for the elderly person to see the steps clearly and causing them to trip."'''

ENV_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', ENV_EVENT_QUERY_DESC
)

BIO_EVENT_QUERY_DESC = '''Based on the input content, generate a physiology-driven event related to elderly care that could occur in this location.
Physiology-driven events refer to elderly care problems caused by internal physical states of the elderly, such as high blood pressure, dizziness, heart discomfort, or slowed mobility.

Example: If an elderly person is dining in the cafeteria, a possible event is "During the meal, due to difficulty swallowing, the elderly person chokes and starts coughing uncontrollably."'''

BIO_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', BIO_EVENT_QUERY_DESC
)

COGNITIVE_EVENT_QUERY_DESC = '''Based on the input content, generate a cognition-feedback-driven event related to elderly care that could occur in this location.
Cognition-feedback-driven events refer to problems arising from the elderly’s cognitive and psychological states, such as feelings of loneliness, memory decline, or anxiety.

Example: If an elderly person is staying alone in their room, a possible event is "Because of spending long periods alone in the room, the elderly person feels lonely and emotionally depressed."'''

COGNITIVE_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace(
    '{description}', COGNITIVE_EVENT_QUERY_DESC
)

def get_event_dimensions():
    return {
        "environment": ENV_EVENT_QUERY_PROMPT,
        "biological": BIO_EVENT_QUERY_PROMPT,
        "cognitive": COGNITIVE_EVENT_QUERY_PROMPT
    }

LONG_TERM_GOAL_EXAMPLES = '''- Establish a health management routine tailored for older adults, preventing minor issues from developing into serious conditions.
- Support seniors in maintaining a positive mindset and preventing loneliness from quietly eroding their later years.
- Learn cognitive training techniques to help older adults sustain mental sharpness over the long term.
- Assist elders over the next few years with insurance, medical, and financial matters to provide greater security in daily life.
- Continue engaging in gentle physical activity to maintain vitality and preserve independence as aging progresses.
- Keep the mind active through simple practices like playing chess, attending lectures, and writing.
- Gradually learn to trust the care offered by doctors and family members, rather than hiding health concerns out of pride.
'''

def get_infer_goal_prompt():
    return BASE_INFER_GOAL_PROMPT.replace(
        '{examples}', LONG_TERM_GOAL_EXAMPLES
    )