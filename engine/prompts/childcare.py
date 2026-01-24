from engine.prompts.base import BASE_QUERY_PROMPT, BASE_INFER_GOAL_PROMPT

ENV_EVENT_QUERY_DESC = '''Based on the input content, generate an environment-driven event related to maternal and infant care that could occur in this location.
Environment-driven events refer to childcare issues or accidents caused by external environmental factors, such as excessive room noise affecting the baby's sleep, uncomfortable room temperature, or overly bright light.

Example: If a newborn is in the nursery, a possible event is "There is continuous noise in the nursery, such as the sound of the air conditioner running, which affects the baby's sleep quality."'''

ENV_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace('{description}', ENV_EVENT_QUERY_DESC)

BIO_EVENT_QUERY_DESC = '''Based on the input content, generate a physiology-driven event related to maternal and infant care that could occur in this location.
Physiology-driven events refer to childcare issues caused by internal physical conditions, such as the baby being hungry, the mother being fatigued, or the baby feeling unwell.

Example: If a newborn is in the crib, a possible event is "The baby keeps crying due to hunger and needs to be fed promptly."'''

BIO_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace('{description}', BIO_EVENT_QUERY_DESC)

COGNITIVE_EVENT_QUERY_DESC = '''Based on the input content, generate a cognition-feedback-driven event description related to maternal and infant care that could occur in this location.
Cognition-feedback-driven events refer to childcare issues based on personal cognition and psychological states, such as new mothers feeling anxious, lacking childcare knowledge, or worrying about the baby's development.

Example: If a new mother is in the living room, a possible event is "The mother worries about the baby's developmental progress and becomes overly concerned with every small movement the baby makes."'''

COGNITIVE_EVENT_QUERY_PROMPT = BASE_QUERY_PROMPT.replace('{description}', COGNITIVE_EVENT_QUERY_DESC)

def get_event_dimensions():
    return {
        "environment": ENV_EVENT_QUERY_PROMPT,
        "biological": BIO_EVENT_QUERY_PROMPT,
        "cognitive": COGNITIVE_EVENT_QUERY_PROMPT
    }

LONG_TERM_GOAL_EXAMPLES = '''- Help the child establish a stable daily rhythm over the next few years, supporting healthier physical well-being and sleep.
- Build trust through consistent, long-term presence, so the child feels safe sharing concerns right away when challenges arise.
- Cultivate understanding rather than rushing to control, allowing the child to feel genuinely accepted throughout the parenting journey.
- Continue growing alongside the child—so parenting becomes a path of mutual development, not just the child’s improvement.
- Support the child in discovering a genuine area of interest over the next few years and nurture their long-term growth.
- Establish a regular sleep schedule for the baby in the coming months, bringing restful nights for both the child and caregiver.
- Let mealtimes with the baby become natural and relaxed, without anxiety over every bite.
'''

def get_infer_goal_prompt():
    return BASE_INFER_GOAL_PROMPT.replace(
        '{examples}', LONG_TERM_GOAL_EXAMPLES
    )