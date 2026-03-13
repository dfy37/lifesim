import os
import json
import argparse
import numpy as np
from collections import defaultdict

from eval import load_jsonl_data, write_jsonl_data
from utils.utils import parse_json_dict_response, find_closest_str_match


# ---------------------------------------------------------------------------
# Per-item processors
# ---------------------------------------------------------------------------

def process_ir_one(item):
    output = parse_json_dict_response(item['output'], [])
    if isinstance(output, list):
        output = output[0] if output else {}
    candidates = list(output.keys()) if isinstance(output, dict) else []

    for d in item['sub_intents']:
        k = find_closest_str_match(d['description'], candidates)
        try:
            d['ir_score'] = int(output[k])
        except Exception:
            d['ir_score'] = 0
    return item


def process_ic_one(item, ir_scores):
    output = parse_json_dict_response(item['output'], [])
    if isinstance(output, list):
        output = output[0] if output else {}
    candidates = list(output.keys()) if isinstance(output, dict) else []

    for d in item['sub_intents']:
        k = find_closest_str_match(d['description'], candidates)
        _id = f'{item["id"]}@{d["description"]}'
        if ir_scores.get(_id, 0) > 0:
            try:
                d['ic_score'] = int(output[k])
            except Exception:
                d['ic_score'] = 0
        else:
            d['ic_score'] = 0
    return item


def process_pa_one(item):
    output = parse_json_dict_response(item['output'], [])
    if isinstance(output, list):
        output = output[0] if output else {}
    candidates = list(output.keys()) if isinstance(output, dict) else []

    scores = []
    for d in item['conv']['user_preferences']:
        k = find_closest_str_match(d, candidates)
        try:
            scores.append(int(output[k]))
        except Exception:
            scores.append(0)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Metric compute functions
# ---------------------------------------------------------------------------

def compute_ir_ic(results_root, model, evaluators):
    ir_cumulative = {}         
    e_ir_scores = {}            

    for evaluator in evaluators:
        e_ir_scores[evaluator] = {}
        path = os.path.join(results_root, evaluator, model, 'ir_results.jsonl')
        for x in load_jsonl_data(path):
            x = process_ir_one(x)
            for s in x['sub_intents']:
                _id = f'{x["id"]}@{s["description"]}'
                ir_cumulative[_id] = ir_cumulative.get(_id, 0) + s['ir_score']
                e_ir_scores[evaluator][_id] = s['ir_score']

    e_ir_scores_list, i_ir_scores_list = [], []
    ref_path = os.path.join(results_root, evaluators[0], model, 'ir_results.jsonl')
    for x in load_jsonl_data(ref_path):
        explicit_ir, implicit_ir = [], []
        for s in x['sub_intents']:
            _id = f'{x["id"]}@{s["description"]}'
            avg = ir_cumulative.get(_id, 0) / len(evaluators)
            (explicit_ir if s['type'] == 'explicit' else implicit_ir).append(avg)
        if explicit_ir:
            e_ir_scores_list.append(np.mean(explicit_ir))
        if implicit_ir:
            i_ir_scores_list.append(np.mean(implicit_ir))

    ic_cumulative = {}
    for evaluator in evaluators:
        path = os.path.join(results_root, evaluator, model, 'ic_results.jsonl')
        for x in load_jsonl_data(path):
            x = process_ic_one(x, e_ir_scores[evaluator])
            for s in x['sub_intents']:
                _id = f'{x["id"]}@{s["description"]}'
                ic_cumulative[_id] = ic_cumulative.get(_id, 0) + s['ic_score']

    e_ic_scores_list, i_ic_scores_list = [], []
    ref_path = os.path.join(results_root, evaluators[0], model, 'ic_results.jsonl')
    for x in load_jsonl_data(ref_path):
        explicit_ic, implicit_ic = [], []
        for s in x['sub_intents']:
            _id = f'{x["id"]}@{s["description"]}'
            avg = ic_cumulative.get(_id, 0) / len(evaluators)
            (explicit_ic if s['type'] == 'explicit' else implicit_ic).append(avg)
        if explicit_ic:
            e_ic_scores_list.append(np.mean(explicit_ic))
        if implicit_ic:
            i_ic_scores_list.append(np.mean(implicit_ic))

    return {
        'IR_explicit': round(np.mean(e_ir_scores_list) * 100, 1) if e_ir_scores_list else 0.0,
        'IR_implicit': round(np.mean(i_ir_scores_list) * 100, 1) if i_ir_scores_list else 0.0,
        'IC_explicit': round(np.mean(e_ic_scores_list) * 100, 1) if e_ic_scores_list else 0.0,
        'IC_implicit': round(np.mean(i_ic_scores_list) * 100, 1) if i_ic_scores_list else 0.0,
    }


def compute_nat(results_root, model, evaluators):
    nat_results = defaultdict(float)
    for evaluator in evaluators:
        path = os.path.join(results_root, evaluator, model, 'nat_results.jsonl')
        for x in load_jsonl_data(path):
            output = parse_json_dict_response(x['output'], ['rating'])
            try:
                nat_results[x['id']] += float(output['rating'])
            except Exception:
                pass
    score = np.mean(list(nat_results.values())) * 20 / len(evaluators) if nat_results else 0.0
    return {'NAT': round(score, 1)}


def compute_coh(results_root, model, evaluators):
    coh_results = defaultdict(float)
    for evaluator in evaluators:
        path = os.path.join(results_root, evaluator, model, 'coh_results.jsonl')
        for x in load_jsonl_data(path):
            output = parse_json_dict_response(x['output'], ['rating'])
            try:
                coh_results[x['id']] += float(output['rating'])
            except Exception:
                pass
    score = np.mean(list(coh_results.values())) * 20 / len(evaluators) if coh_results else 0.0
    return {'COH': round(score, 1)}


def compute_pa(results_root, model, evaluators):
    pa_results = defaultdict(float)
    for evaluator in evaluators:
        path = os.path.join(results_root, evaluator, model, 'pa_results.jsonl')
        for x in load_jsonl_data(path):
            pa_results[x['id']] += process_pa_one(x)
    score = np.mean(list(pa_results.values())) * 100 / len(evaluators) if pa_results else 0.0
    return {'PA': round(score, 1)}


def compute_ea(results_root, model, evaluators):
    ea_results = defaultdict(float)
    for evaluator in evaluators:
        path = os.path.join(results_root, evaluator, model, 'ea_results.jsonl')
        for x in load_jsonl_data(path):
            output = parse_json_dict_response(x['output'], ['rating'])
            try:
                ea_results[x['id']] += float(output['rating'])
            except Exception:
                pass
    score = np.mean(list(ea_results.values())) * 20 / len(evaluators) if ea_results else 0.0
    return {'EA': round(score, 1)}


def compute_rr(results_root, model, evaluators):
    rr_results = defaultdict(float)
    for evaluator in evaluators:
        path = os.path.join(results_root, evaluator, model, 'rr_results.jsonl')
        for x in load_jsonl_data(path):
            output = parse_json_dict_response(x['output'], ['rigid_reasoning'])
            try:
                rr_results[x['id']] += float(output['rigid_reasoning'])
            except Exception:
                pass
    score = np.mean(list(rr_results.values())) * 100 / len(evaluators) if rr_results else 0.0
    return {'RR': round(score, 1)}


METRIC_RUNNERS = {
    'ir':  compute_ir_ic,
    'ic':  compute_ir_ic,
    'nat': compute_nat,
    'coh': compute_coh,
    'pa':  compute_pa,
    'ea':  compute_ea,
    'rr':  compute_rr,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute LifeSim evaluation metrics from LLM-judge outputs.'
    )
    parser.add_argument('--results_root', required=True,
                        help='Root directory of eval.py outputs. '
                             'Expected layout: {results_root}/{evaluator}/{model}/{metric}_results.jsonl')
    parser.add_argument('--models',     nargs='+', required=True,
                        help='Model folder names to score (same as --themes used in eval.py).')
    parser.add_argument('--evaluators', nargs='+', required=True,
                        help='Evaluator folder names under results_root.')
    parser.add_argument('--metrics',    nargs='+', default=list(METRIC_RUNNERS),
                        choices=list(METRIC_RUNNERS),
                        help='Metrics to compute (default: all).')
    parser.add_argument('--output_root', default=None,
                        help='If set, saves scores.json per model under this directory.')
    return parser.parse_args()


def run_metrics(args):
    if args.output_root:
        os.makedirs(args.output_root, exist_ok=True)

    for model in args.models:
        print(f'\n===== {model} =====')
        all_scores = {}
        ir_ic_computed = False

        for metric in args.metrics:
            if metric in ('ir', 'ic'):
                if ir_ic_computed:
                    continue
                scores = compute_ir_ic(args.results_root, model, args.evaluators)
                ir_ic_computed = True
            else:
                scores = METRIC_RUNNERS[metric](args.results_root, model, args.evaluators)

            all_scores.update(scores)
            for k, v in scores.items():
                print(f'  {k}: {v}')

        if args.output_root:
            out_dir = os.path.join(args.output_root, model)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'scores.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(all_scores, f, indent=2, ensure_ascii=False)
            print(f'  -> scores saved to {out_path}')


if __name__ == '__main__':
    run_metrics(parse_args())
