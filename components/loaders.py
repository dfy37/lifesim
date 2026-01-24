
import json
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_jsonl(path):
    arr = []
    with open(path) as reader:
        for line in reader:
            arr.append(json.loads(line))
    return arr

def load_events():
    cfg = load_config()
    return load_jsonl(cfg["paths"]["events_path"])

def load_users():
    cfg = load_config()
    return load_jsonl(cfg["paths"]["users_path"])
