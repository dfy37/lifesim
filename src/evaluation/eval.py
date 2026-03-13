import os
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from models.cloud import DeepSeek, APIModel
from models.vllm import VLLMModel, Qwen3API


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl_data(path):
    with open(path) as f:
        return [json.loads(row) for row in f]


def write_jsonl_data(data, path):
    with open(path, 'w') as f:
        for row in data:
            f.write(json.dumps(row) + '\n')


def get_eval_dataset(theme, logs_root):
    base_dir = os.path.join(logs_root, theme)
    paths = []
    for dirpath, _dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith('sim_log'):
                paths.append(os.path.join(dirpath, f))

    eval_for_ir   = []
    eval_for_pp   = []
    eval_for_conv = []

    for p in paths:
        with open(p) as f:
            data = json.load(f)
        name = os.path.basename(os.path.dirname(p))

        sequence_id = data['event_sequence_info']['sequence_id']

        for i, x in enumerate(data['dialogue_log']):
            gold_intent      = x['event']['intent']
            gold_sub_intents = x['event']['sub_intents']
            gold_preferences = x['user']['profile']['preferences_value']

            conv         = ''
            pred_intents = ['']

            for j, turn in enumerate(x['dialogue']):
                if turn['role'] == 'user':
                    conv += f"[User] {turn['content']}\n"
                else:
                    conv += f"[Assistant] {turn['content']}\n"
                    pre_intent = turn['pre_intent']
                    if pre_intent:
                        pred_intents.append(pre_intent)
                    eval_for_ir.append({
                        'id':              '_'.join([name, str(i), str(j)]),
                        'sequence_id':     sequence_id,
                        'user_id':         x['user']['profile']['user_id'],
                        'gold_intent':     gold_intent,
                        'gold_sub_intents': gold_sub_intents,
                        'pre_intent':      pre_intent if pre_intent else '',
                    })

            eval_for_pp.append({
                'id':               '_'.join([name, str(i)]),
                'sequence_id':      sequence_id,
                'user_id':          x['user']['profile']['user_id'],
                'gold_preferences': gold_preferences,
                'pred_preferences': x['pre_profile'],
            })

            eval_for_conv.append({
                'id':              '_'.join([name, str(i)]),
                'sequence_id':     sequence_id,
                'user_id':         x['user']['profile']['user_id'],
                'user_profile':    x['user']['profile_str'],
                'user_preferences': gold_preferences,
                'dialogue_scene':  x['event']['dialogue_scene'],
                'intent':          gold_intent,
                'sub_intents':     gold_sub_intents,
                'conv':            conv,
                'pred_intents':    pred_intents,
            })

    return eval_for_ir, eval_for_pp, eval_for_conv


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_CLASS_MAP = {
    'gpt-4o':        APIModel,
    'gpt-4o-mini':   APIModel,
    'gpt_oss_120b':  VLLMModel,
    'qwen3_32b':     Qwen3API,
    'llama3_70b':    VLLMModel,
    'deepseek_chat': DeepSeek,
}


def build_model_factory(base_class, api_key, base_url=None, model_path=None):
    """Return a zero-argument callable that produces a ready-to-use model instance."""
    def factory():
        kwargs = {'api_key': api_key}
        if model_path:
            kwargs['model'] = model_path
        if base_url:
            kwargs['base_url'] = base_url
        return base_class(**kwargs)
    return factory


# ---------------------------------------------------------------------------
# Concurrent inference
# ---------------------------------------------------------------------------

def process_single_item(model, item, index):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            item['output'] = model.chat([{'role': 'user', 'content': item['input']}])
            return index, item, None
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error for item {index} after {max_retries} attempts: {e}")
                item['output'] = ''
                return index, item, str(e)
            time.sleep(2 ** attempt)
    return index, item, None


def process_data_concurrent(model_factory, data, max_workers=32):
    results    = [None] * len(data)
    error_count = 0
    lock        = threading.Lock()
    thread_local = threading.local()

    def get_model():
        if not hasattr(thread_local, 'model'):
            thread_local.model = model_factory()
        return thread_local.model

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        indexed   = [(i, item.copy()) for i, item in enumerate(data)]
        futures   = [executor.submit(process_single_item, get_model(), item, i)
                     for i, item in indexed]

        with tqdm(total=len(futures), desc='Processing') as pbar:
            for future in as_completed(futures):
                index, result, error = future.result()
                results[index] = result
                if error:
                    with lock:
                        error_count += 1
                pbar.set_postfix({'errors': error_count})
                pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Evaluation prompts & runners
# ---------------------------------------------------------------------------

IR_PROMPT = '''You are an evaluator assessing whether an AI assistant's predicted intent correctly matches the **real user intent** in a given dialogue.
You will be provided with:
* Intent checklist — a structured list representing the verified components of the user's real intent.
* Predicted intents — the assistant's inferred or generated intent statements during the whole conversation.
* Conversation — showing the interaction between the user and the assistant.

### Requirements
Your task is to evaluate how accurately the predicted intent aligns with the intent checklist.
For each checklist item, determine whether the predicted intent successfully captures that element.
Each dimension in the checklist should be scored as follows:
* 1 = The predicted intent correctly covers this item
* 0 = The predicted intent fails to capture or contradicts this item

### Output Format
Your final answer should follow this format:
```json
{{
  "Checklist item 1": 1/0,
  "Checklist item 2": 1/0,
  ...
}}
```
Before your response, provide: **Dimension-by-dimension assessment (bullet list)** — show each checklist item, a short justification, and its binary score (1/0).

### Examples
[Intent checklist]
- Wants to find strategies to improve focus while working remotely.
- Seeks emotional reassurance that losing focus is normal.
- Prefers realistic, easy-to-apply methods over abstract motivation.
[Predicted intent]
The user wants to regain productivity by finding concrete methods to stay focused when working from home.
[Output]
Concise evaluation:
The predicted intent captures the main functional goal (improving focus) and emphasizes productivity, which aligns well with the user's practical needs. However, it overlooks the emotional reassurance component — acknowledging that distraction is normal — which was part of the true intent.
Dimension-by-dimension assessment:
Wants to find strategies to improve focus while working remotely → ✅ Correctly captured. → 1
Seeks emotional reassurance that losing focus is normal → ⚠️ Missing — no emotional aspect included. → 0
Prefers realistic, easy-to-apply methods over abstract motivation → ✅ The phrase "concrete methods" fits well. → 1
```json
{{
  "Wants to find strategies to improve focus while working remotely": 1,
  "Seeks emotional reassurance that losing focus is normal": 0,
  "Prefers realistic, easy-to-apply methods over abstract motivation": 1
}}
```

### Input
[Intent checklist]
{checklist}
[Conversation]
{conversation}
[Predicted intent]
{predicted_intent}
[Output]
'''


def get_ir_rating(eval_for_conv, model_factory, max_workers=32):
    inputs = []
    for x in eval_for_conv:
        checklist_str  = '\n'.join(f"- {item['description']}" for item in x['sub_intents'])
        pred_intents   = '\n'.join(x['pred_intents'])
        x['input'] = IR_PROMPT.format(
            checklist=checklist_str,
            conversation=x['conv'],
            predicted_intent=pred_intents,
        )
        inputs.append(x)
    return process_data_concurrent(model_factory, inputs, max_workers=max_workers)


IC_PROMPT = '''You are an evaluator assessing whether an AI assistant has effectively fulfilled the user's intent in a given dialogue.
You will be provided with:
* User profile — describing the user's characteristics and communication tendencies.
* Dialogue scene — describing the environmental and contextual background.
* User intent — describing what the user wanted to achieve through the conversation.
* Conversation — showing the actual interaction between the user and the assistant.

### Requirements
Your task is to evaluate how accurately the assistant's response meets the user's intent, based on the key dimensions listed in the checklist.
Each dimension in the checklist should be scored as follows:
* 1 = The assistant correctly fulfills this intent dimension
* 0 = The assistant fails to fulfill or contradicts this intent dimension

### Output Format
Your final answer should follow this format:
```json
{{
  "Checklist item 1": 1/0,
  "Checklist item 2": 1/0,
  ...
}}
```
Before your response, provide: **Dimension-by-dimension assessment (bullet list)** — show each checklist item, a short justification, and its binary score (1/0).

### Input
[User profile]
{user_profile}
[Dialogue scene]
{dialogue_scene}
[Intent checklist]
{checklist}
[Conversation]
{conversation}
'''


def get_ic_rating(eval_for_conv, model_factory, max_workers=32):
    inputs = []
    for x in eval_for_conv:
        checklist_str = '\n'.join(f"- {item['description']}" for item in x['sub_intents'])
        x['input'] = IC_PROMPT.format(
            user_profile=x['user_profile'],
            dialogue_scene=x['dialogue_scene'],
            conversation=x['conv'],
            checklist=checklist_str,
        )
        inputs.append(x)
    return process_data_concurrent(model_factory, inputs, max_workers=max_workers)


NAT_PROMPT = '''You are an evaluator assessing the fluency and naturalness of an AI assistant's conversation with a user.
You will be provided with:
* User profile — describing the user's characteristics and communication tendencies.
* Dialogue scene — describing the situational context of the conversation.
* User intent — describing what the user wanted to achieve through the interaction.
* Conversation — showing the actual interaction between the user and the assistant.

### Requirements
Your task is to determine whether the AI assistant's responses are fluent, coherent, and natural throughout the conversation.
Analyze it from multiple relevant dimensions:
* Language is conversational, avoiding overly long, formal, or bookish expressions.
* Vocabulary is natural, everyday, and varied, avoiding repetition or overly technical terms.
* Tone and emotion match the user's preferred style, showing empathy, engagement, and responsiveness.
* Replies actively incorporate and respond to user-provided details, making the conversation feel personalized.
* Replies include proactive questions to guide the conversation, rather than only passively responding.

Your response should be structured in JSON format, enclosed in ```json and ```:
```json
{{
  "rating": 1-5
}}
```
Where:
    * 1 = Very unnatural or disfluent
    * 2 = Mostly unnatural, noticeable problems in phrasing
    * 3 = Moderately fluent but with some issues
    * 4 = Mostly natural, minor disfluency
    * 5 = Fully fluent and natural

Before your JSON output, provide:
1. A concise evaluation (2–3 sentences) summarizing the overall fluency of the assistant's replies.
2. A dimension-by-dimension assessment (as bullet points).

### Input
[User profile]
{user_profile}
[Dialogue scene]
{dialogue_scene}
[User intent]
{user_intent}
[Conversation]
{conversation}
'''


def get_nat_rating(eval_for_conv, model_factory, max_workers=32):
    inputs = []
    for x in eval_for_conv:
        inputs.append({
            'id':    x['id'],
            'conv':  x,
            'input': NAT_PROMPT.format(
                user_profile=x['user_profile'],
                dialogue_scene=x['dialogue_scene'],
                user_intent=x['intent'],
                conversation=x['conv'],
            ),
        })
    return process_data_concurrent(model_factory, inputs, max_workers=max_workers)


COH_PROMPT = '''You are an evaluator assessing the coherence and logical consistency of an AI assistant's conversation with a user.
You will be provided with:
* User profile — describing the user's characteristics and communication tendencies.
* Dialogue scene — describing the situational context of the conversation.
* User intent — describing what the user wanted to achieve through the interaction.
* Conversation — showing the actual interaction between the user and the assistant.

### Requirements
Your task is to determine whether the AI assistant's responses are coherent, logically consistent, and contextually aligned throughout the dialogue.
Analyze it from multiple relevant dimensions:
* Responses should focus on the user's main concerns, avoiding unnecessary digressions or repetitive generic advice.
* Each response should be logically consistent, avoiding contradictions or redundant statements that add no new value.
* Responses should correctly reference and integrate information from previous turns, demonstrating understanding of the context.
* Pronouns and references should be clear, avoiding ambiguity or unclear referents.
* Information should be organized coherently, with a clear logical order that is easy to follow.

Your response should be structured in JSON format, enclosed in `json and `:
```json
{{
  "rating": 1-5
}}
```
Where:
    * 1 = Completely incoherent or contradictory
    * 2 = Mostly incoherent, several logical gaps or inconsistencies
    * 3 = Partially coherent with some logical gaps or inconsistencies
    * 4 = Mostly coherent, minor logical gaps or inconsistencies
    * 5 = Fully coherent and logically consistent

Before your JSON output, provide:
1. A concise evaluation (2–3 sentences) summarizing the overall coherence of the assistant's replies.
2. A dimension-by-dimension assessment (as bullet points).

### Input
[User profile]
{user_profile}
[Dialogue scene]
{dialogue_scene}
[User intent]
{user_intent}
[Conversation]
{conversation}
'''


def get_coh_rating(eval_for_conv, model_factory, max_workers=32):
    inputs = []
    for x in eval_for_conv:
        inputs.append({
            'id':    x['id'],
            'conv':  x,
            'input': COH_PROMPT.format(
                user_profile=x['user_profile'],
                dialogue_scene=x['dialogue_scene'],
                user_intent=x['intent'],
                conversation=x['conv'],
            ),
        })
    return process_data_concurrent(model_factory, inputs, max_workers=max_workers)


PA_PROMPT = '''You are an evaluator assessing how well an AI assistant's replies align with the user's preferences.
You will be provided with:
* User preferences — a list of preference dimensions (e.g., "Need for autonomy," "Preference for emotional support") and their expected values or tendencies.
* Conversation — showing the interaction between the user and the assistant.

### Requirements
Your task is to evaluate the alignment for each preference dimension individually.
For each dimension listed in the user profile, determine whether the assistant's replies conform to that specific preference.

#### Evaluation Criteria
For each preference dimension:
* 1 = The assistant's reply clearly aligns with this preference dimension.
* 0 = The assistant's reply contradicts or fails to reflect this preference dimension.
Then, provide an overall summary at the end.

### Output Format
Your response should contain:
1. Dimension-by-Dimension Assessment — a structured list showing each preference dimension, a short justification, and its binary alignment score.
2. JSON Output — containing all dimension scores.
```json
{{
  "Preference for xxx": 1/0,
  ...
}}
```

### Examples
[User Preferences]
- Preference for emotional warmth: high
- Preference for detailed explanations: low
- Preference for direct communication: high
[Conversation]
User: I just feel like things keep piling up, and I can't catch my breath.
Assistant: That sounds really stressful. Maybe we can talk about some ways to slow down and make space for yourself.
[Output]
* Preference for emotional warmth → The assistant shows care and empathy. → **1**
* Preference for detailed explanations → The reply is brief and not overly detailed, matching the user's low-detail preference. → **1**
* Preference for direct communication → The assistant's phrasing is soft and reflective, not very direct. → **0**
* Preference for human safety → The assistant's response does not reflect a concern for the user's safety. → **0**
```json
{{
  "Preference for emotional warmth": 1,
  "Preference for detailed explanations": 1,
  "Preference for direct communication": 0,
  "Preference for human safety": 0
}}
```

### Input
[User Preferences]
{user_preferences}
[Conversation]
{conversation}
[Output]
'''


def get_pa_rating(eval_for_conv, model_factory, max_workers=32):
    inputs = []
    for x in eval_for_conv:
        user_preferences = '\n'.join(f"- {k}: {v}" for k, v in x['user_preferences'].items())
        inputs.append({
            'id':    x['id'],
            'conv':  x,
            'input': PA_PROMPT.format(
                user_preferences=user_preferences,
                conversation=x['conv'],
            ),
        })
    return process_data_concurrent(model_factory, inputs, max_workers=max_workers)


EA_PROMPT = '''You are an evaluator assessing whether an AI assistant's replies are aligned with the dialogue environment.
You will be provided with:
* Dialogue scene — describing situational context and constraints.
* Conversation — showing the actual interaction between the user and the assistant.

### Requirements
Your task is to determine whether the AI assistant's responses remain contextually consistent with the dialogue environment and are strategically reasonable.
Analyze from multiple relevant dimensions:
* Factual consistency with environment: claims about time, place, weather, availability, opening hours, travel feasibility, etc. should not contradict the provided environment.
* Temporal appropriateness: suggestions match the time of day / day of week / season (e.g., late-night options, commute time, daylight, urgency).
* Spatial/logistical plausibility: suggestions match the location and constraints (distance, transport mode, local context), avoiding impossible or unrealistic plans.
* Weather/context suitability: recommendations adapt to weather and conditions (e.g., rain/heat/cold), and propose sensible alternatives when needed.
* Constraint awareness: respects explicit constraints in the scene/environment (budget, safety, accessibility, tools available, deadlines).
* Proactive adaptation: when environment makes the intent difficult, the assistant offers adjustments, contingency plans, or asks targeted questions to resolve missing environment details (only when truly necessary).

### Rating scale (1-5)
Rate overall environment alignment for the assistant across the conversation:
* 1 = Strongly misaligned; frequent contradictions or impossible advice
* 2 = Often misaligned; multiple notable inconsistencies
* 3 = Generally aligned but with some issues or missed adaptations
* 4 = Mostly aligned; minor slips only
* 5 = Fully aligned; consistently environment-aware and strategically appropriate

Your response MUST follow this structure:
1) A concise evaluation (2–3 sentences) summarizing overall alignment.
2) A dimension-by-dimension assessment (bullet points).
3) A JSON output enclosed in ```json and ``` with:
```json
{{
  "rating": 1-5
}}
### Input
[Dialogue scene]
{dialogue_scene}
[Conversation]
{conversation}
'''


def get_env_alignment_rating(eval_for_conv, model_factory, max_workers=32):
    inputs = []
    for x in eval_for_conv:
        inputs.append({
            'id':    x['id'],
            'conv':  x,
            'input': EA_PROMPT.format(
                dialogue_scene=x['dialogue_scene'],
                conversation=x['conv'],
            ),
        })
    return process_data_concurrent(model_factory, inputs, max_workers=max_workers)


RR_PROMPT = '''You are an evaluator detecting "Rigid Reasoning" in an AI assistant across an extended dialogue.
You will be provided with:
* User profile — describing the user's characteristics and communication tendencies.
* Dialogue scene — describing the situational context of the conversation.
* User intent — describing what the user wanted to achieve through the interaction.
* Conversation — showing the actual interaction between the user and the assistant.

### Definition: Rigid Reasoning
Rigid reasoning occurs when the assistant stays anchored to an initial reasoning trajectory and fails to revise its solution strategy in response to evolving user constraints or negative feedback.
Typical symptoms:
- The user repeatedly refines requirements or rejects proposed directions, but the assistant keeps recycling the same solution space with minor rephrasing.
- The assistant does not meaningfully update its plan, assumptions, or problem framing after new constraints are introduced.
- The assistant ignores or downplays infeasibility; instead of acknowledging conflicts, it repeats earlier suggestions.
- The assistant asks generic questions that do not change the approach, or offers "variations" that are essentially the same idea.

Non-rigid (adaptive) behavior includes:
- Explicitly acknowledging new constraints/negative feedback and stating what changed.
- Reformulating the problem or exploring a materially different solution space.
- Explaining trade-offs and, when constraints conflict, proposing constraint relaxation options or declaring infeasibility.
- Introducing new strategies/tools/structures that directly address the updated requirements.

### Task
Decide whether the assistant exhibits rigid reasoning over the conversation.

### Output format
Your response MUST be structured as:
1) A concise verdict (1–2 sentences) explaining why it is rigid or not.
2) Bullet-point evidence referencing specific moments (e.g., "after the user rejected X, the assistant still suggested X-like options").
3) A JSON output enclosed in ```json and ``` with:
```json
{{
  "rigid_reasoning": 0/1
}}
Where:
1 = Rigid reasoning is present (noticeable anchoring + failure to adapt)
0 = Rigid reasoning is not present (the assistant dynamically adapts its reasoning)
Input
[User profile]
{user_profile}
[Dialogue scene]
{dialogue_scene}
[User intent]
{user_intent}
[Conversation]
{conversation}
'''


def get_rigid_reasoning_flag(eval_for_conv, model_factory, max_workers=32):
    inputs = []
    for x in eval_for_conv:
        inputs.append({
            'id':    x['id'],
            'conv':  x,
            'input': RR_PROMPT.format(
                user_profile=x['user_profile'],
                dialogue_scene=x['dialogue_scene'],
                user_intent=x['intent'],
                conversation=x['conv'],
            ),
        })
    return process_data_concurrent(model_factory, inputs, max_workers=max_workers)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METRIC_RUNNERS = {
    'ir':  (get_ir_rating,             'ir_results.jsonl'),
    'ic':  (get_ic_rating,             'ic_results.jsonl'),
    'nat': (get_nat_rating,            'nat_results.jsonl'),
    'coh': (get_coh_rating,            'coh_results.jsonl'),
    'pa':  (get_pa_rating,             'pa_results.jsonl'),
    'ea':  (get_env_alignment_rating,  'ea_results.jsonl'),
    'rr':  (get_rigid_reasoning_flag,  'rr_results.jsonl'),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run LifeSim evaluation pipeline.')
    parser.add_argument('--logs_root',   required=True, help='Root directory containing simulation log folders.')
    parser.add_argument('--themes',      nargs='+', required=True, help='Theme folder names under logs_root.')
    parser.add_argument('--output_root', required=True, help='Directory to save evaluation outputs.')
    parser.add_argument('--evaluator',   choices=list(MODEL_CLASS_MAP), default='qwen3_32b')
    parser.add_argument('--api_key',     default=os.getenv('OPENAI_API_KEY', '123'), help='API key for evaluator.')
    parser.add_argument('--base_url',    default=None, help='OpenAI-compatible base URL for evaluator.')
    parser.add_argument('--model_path',  required=True, help='Model path/name for evaluator.')
    parser.add_argument('--metrics',     nargs='+', default=['rr'],
                        choices=list(METRIC_RUNNERS), help='Metrics to run.')
    parser.add_argument('--max_workers', type=int, default=32, help='Worker threads for concurrent judging.')
    return parser.parse_args()


def run_evaluation(args):
    os.makedirs(args.output_root, exist_ok=True)
    base_class    = MODEL_CLASS_MAP[args.evaluator]
    model_factory = build_model_factory(base_class, api_key=args.api_key,
                                        base_url=args.base_url, model_path=args.model_path)

    for theme in args.themes:
        print(f'Running theme: {theme}')
        _, _, eval_for_conv = get_eval_dataset(theme, logs_root=args.logs_root)
        out_dir = os.path.join(args.output_root, theme)
        os.makedirs(out_dir, exist_ok=True)

        for metric in args.metrics:
            runner, out_name = METRIC_RUNNERS[metric]
            print(f'  -> {metric.upper()}')
            results = runner(eval_for_conv, model_factory, max_workers=args.max_workers)
            write_jsonl_data(results, os.path.join(out_dir, out_name))


if __name__ == '__main__':
    run_evaluation(parse_args())
