from engine.prompts.base import (
    RERANK_PROMPT,
    REWRITE_PROMPT
)

def get_event_dimensions(theme):
    if theme == 'sport_health':
        from engine.prompts.sport_health import get_event_dimensions
        return get_event_dimensions()
    elif theme == 'education':
        from engine.prompts.education import get_event_dimensions
        return get_event_dimensions()
    elif theme == 'mental_health':
        from engine.prompts.mental_health import get_event_dimensions
        return get_event_dimensions()
    elif theme == 'travel':
        from engine.prompts.travel import get_event_dimensions
        return get_event_dimensions()
    elif theme == 'childcare':
        from engine.prompts.childcare import get_event_dimensions
        return get_event_dimensions()
    elif theme == 'dining':
        from engine.prompts.dining import get_event_dimensions
        return get_event_dimensions()
    elif theme == 'elderlycare':
        from engine.prompts.elderlycare import get_event_dimensions
        return get_event_dimensions()
    elif theme == 'entertainment':
        from engine.prompts.entertainment import get_event_dimensions
        return get_event_dimensions()
    else:
        raise ValueError(f"Unsupported theme: {theme}.")

def get_infer_goal_prompt(theme):
    if theme == 'sport_health':
        from engine.prompts.sport_health import get_infer_goal_prompt
        return get_infer_goal_prompt()
    elif theme == 'education':
        from engine.prompts.education import get_infer_goal_prompt
        return get_infer_goal_prompt()
    elif theme == 'mental_health':
        from engine.prompts.mental_health import get_infer_goal_prompt
        return get_infer_goal_prompt()
    elif theme == 'travel':
        from engine.prompts.travel import get_infer_goal_prompt
        return get_infer_goal_prompt()
    elif theme == 'childcare':
        from engine.prompts.childcare import get_infer_goal_prompt
        return get_infer_goal_prompt()
    elif theme == 'dining':
        from engine.prompts.dining import get_infer_goal_prompt
        return get_infer_goal_prompt()
    elif theme == 'elderlycare':
        from engine.prompts.elderlycare import get_infer_goal_prompt
        return get_infer_goal_prompt()
    elif theme == 'entertainment':
        from engine.prompts.entertainment import get_infer_goal_prompt
        return get_infer_goal_prompt()
    else:
        raise ValueError(f"Unsupported theme: {theme}.")