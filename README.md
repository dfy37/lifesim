# LifeSim: Long-Horizon User Life Simulator for Personalized Assistant Evaluation

The rapid advancement of large language models (LLMs) has accelerated progress toward universal AI assistants. However, existing benchmarks for personalized assistants remain misaligned with real-world user-assistant interactions, failing to capture the complexity of external contexts and users' cognitive states.

To bridge this gap, we propose **LifeSim**, a user simulator that models user cognition through the Belief-Desire-Intention (BDI) model within physical environments for coherent life trajectories generation, and simulates intention-driven user interactive behaviors. Based on LifeSim, we introduce **LifeSim-Eval**, a comprehensive benchmark for multi-scenario, long-horizon personalized assistance.

![LifeSim Framework](resources/framework.png)

---

## Table of Contents

- [Quick Start](#quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Model Setup](#3-model-setup)
  - [4. Run Simulation](#4-run-simulation)
  - [5. Evaluation](#5-evaluation)
- [Web Demo](#web-demo)
- [Citation](#citation)

---

## Quick Start

### 1. Environment Setup

Create a conda environment and install dependencies:

```bash
conda create -n lifesim python=3.10.12
conda activate lifesim
pip install -r requirements.txt
```

> **Note:** The provided `requirements.txt` is comprehensive and includes GPU/vLLM-related packages. For a lightweight setup (e.g., API-only models without local GPU inference), you only need the core packages: `openai`, `chromadb`, `sentence-transformers`, `flask`, `flask-cors`, `pyyaml`, `tqdm`, `numpy`.

---

### 2. Data Preparation

The pipeline expects the following directory layout under `data/`:

```
data/
├── single_session/
│   ├── events.jsonl          # Event sequences 
│   └── users.jsonl           # User profiles
├── long_horizon/
│   ├── events.jsonl          # Event sequences 
│   └── users.jsonl           # User profiles
└── language_templates.json   # Preference dimension templates
```

---

### 3. Model Setup

LifeSim requires two LLM backends — a **user model** (simulates the user) and an **assistant model** (the AI system under evaluation) — plus an **embedding model** for the retrieval memory.

#### Option A: Local Models via vLLM

Launch vLLM servers for the user and assistant models separately. For example:

```bash
# Launch user model (e.g., Qwen3-32B)
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/Qwen3-32B \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 2 \
  --api-key your_api_key

# Launch assistant model (e.g., Llama-3-70B)
CUDA_VISIBLE_DEVICES=2,3 vllm serve /path/to/Llama-3-70B \
  --host 0.0.0.0 --port 8002 \
  --tensor-parallel-size 2 \
  --api-key your_api_key
```

#### Option B: Cloud API Models

No local deployment is needed. Supported model names (passed via `--assistant_model_path`):

| Provider | Model Name |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `gpt-5-mini` |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` |
| Anthropic | `claude-sonnet-4-5-20250929` |

Pass the corresponding API key via `--assistant_model_api_key`.

#### Embedding Model

An embedding model is required for the ChromaDB-based retrieval memory. We recommend [`Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (or any model compatible with `sentence-transformers`). Download it to a local path and pass it via `--retriever_model_path`.

---

### 4. Run Simulation

Use `src/main_mp.py` as the main entrypoint. Run from the repository root:

```bash
cd /path/to/lifesim

python src/main_mp.py \
  --user_model_path      /path/to/Qwen3-32B \
  --user_model_url       http://0.0.0.0:8001/v1 \
  --user_model_api_key   your_user_api_key \
  --assistant_model_path gpt-4o \
  --assistant_model_api_key your_openai_api_key \
  --retriever_model_path /path/to/Qwen3-Embedding-0.6B \
  --exp_setting          long_horizon \
  --n_events_per_sequence 10 \
  --n_threads            4 \
  --chromadb_root        ./chromadb \
  --logs_root            ./logs
```

### 5. Evaluation

Run `src/evaluation/eval.py` on the simulation logs. The evaluator uses an **LLM-as-a-judge** approach across 7 dimensions:

| Metric | Type | Description |
|---|---|---|
| `ir` | Intent Recognition | Whether the assistant correctly identifies the user's intent |
| `ic` | Intent Completion | Whether the assistant's reply fulfills each intent dimension |
| `nat` | Naturalness | Fluency and conversational naturalness (1–5) |
| `coh` | Coherence | Logical consistency and contextual continuity (1–5) |
| `pa` | Preference Alignment | Whether the reply aligns with the user's preference profile |
| `ea` | Environment Alignment | Scene/environment feasibility and constraint awareness (1–5) |
| `rr` | Rigid Reasoning | Binary flag for failure to adapt after new constraints |

```bash
python src/evaluation/eval.py \
  --logs_root   ./logs \
  --themes      main_user_Qwen3-32B_assistant_gpt-4o_long_horizon \
  --output_root ./eval_outputs \
  --evaluator   gpt-4o \
  --api_key     your_openai_api_key \
  --metrics     ir ic nat coh pa ea rr \
  --max_workers 32
```

Results are saved as JSON files under `output_root`. For post-processing and score aggregation, see `src/evaluation/eval.ipynb`.

---

## Web Demo

The web demo provides an interactive UI for two usage modes:

- **Assistant Evaluation Mode** — runs automated multi-event simulations and displays real-time conversation streams, event timelines, and churn analysis.
- **Free Chat Mode** — dynamically generates life events and lets you interact directly with the simulated user.

### Step 1: Create `config.yaml`

Create a `config.yaml` file in the directory from which you will run the server. A full annotated template:

```yaml
# Model configurations
models:
  user_model:
    model_path: /path/to/Qwen3-32B   # local path or model name
    base_url:   http://0.0.0.0:8001/v1
    api_key:    your_api_key
    vllmapi:    true                  # true = vLLM endpoint; false = unused

  assistant_model:
    model_path: gpt-4o
    base_url:   ""                    # leave empty for official API endpoints
    api_key:    your_openai_api_key
    vllmapi:    false

  analysis_model:                     # model used for churn/quality analysis
    model_path: deepseek-chat
    base_url:   ""
    api_key:    your_deepseek_api_key
    vllmapi:    false

# Data paths
paths:
  events_path:     data/long_horizon/events.jsonl
  users_path:      data/long_horizon/users.jsonl
  preference_file: data/language_templates.json
  event_pool_cfg:  null               # optional: path to event pool config JSON
                                      # required only for Free Chat online event generation

# Retrieval memory (ChromaDB + embedding model)
retriever:
  embedding_model_path: /path/to/Qwen3-Embedding-0.6B
  max_length:           512
  embedding_dim:        1024
  persist_directory:    ./chromadb
  device:               cpu           # or "cuda:0"
  logger_silent:        true

# Simulator hyperparameters
simulator:
  alpha: 0.5                          # memory perception coefficient in user agent

# Assistant models available in the web UI dropdown
assistant_models:
  - gpt-4o
  - gpt-4o-mini
  - deepseek-chat
```

### Step 2: Launch the Server

```bash
cd /path/to/lifesim/src

python flask_app.py --config /path/to/config.yaml --port 5010
```

Then open `http://localhost:5010` in your browser.

**CLI options:**

| Argument | Default | Description |
|---|---|---|
| `--config`, `-c` | `config.yaml` | Path to the YAML configuration file |
| `--host` | `0.0.0.0` | Host address to bind |
| `--port`, `-p` | `5010` | Port to listen on |
| `--debug` | `False` | Enable Flask debug mode |

---

## Citation

If you use LifeSim or LifeSim-Eval in your research, please cite:

```bibtex
@article{lifesim2025,
  title   = {LifeSim: Long-Horizon User Life Simulator for Personalized Assistant Evaluation},
  author  = {},
  journal = {},
  year    = {2025}
}
```
