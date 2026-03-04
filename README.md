# LifeSim: Event-Conditioned User–Assistant Interaction Simulation

This repository provides a research-oriented simulation framework for studying long-horizon interactions between a **user agent** and an **assistant agent** under structured life-event trajectories. The codebase is designed for controlled experimentation on preference inference, memory dynamics, and dialogue adaptation.

The implementation in `src/` is the primary experimental pipeline.

---

## 1. Research Objective

LifeSim operationalizes a sequential interaction problem in which:

- a user profile is instantiated from structured demographic and preference attributes,
- a life-event engine emits event contexts over time,
- a user agent generates responses conditioned on profile, event context, and optional memory/emotion mechanisms,
- an assistant agent predicts intent, responds, and periodically summarizes user preferences.

This setup enables evaluation of longitudinal assistant behavior under non-stationary conversational contexts.

---

## 2. Code Structure (Primary: `src/`)

```text
src/
├── main_mp.py                    # Multi-thread simulation entrypoint
├── agents/
│   ├── user_agent.py             # User behavior generation + memory/emotion/action modules
│   ├── assistant_agent.py        # Intent prediction, response, and profile summarization
│   └── memory.py                 # In-memory and vector-backed memory abstractions
├── simulation/
│   └── conversation_simulator.py # Episode/event loop and logging orchestration
├── engine/
│   └── event_engine.py           # Offline event sequence loading and event generation
├── profiles/
│   └── profile_generator.py      # UserProfile parsing and profile retrieval
├── tools/
│   └── dense_retriever.py        # Retriever used by KV memory (Chroma-backed)
├── models/                       # API/local model wrappers and model loader
└── utils/                        # Logging, JSONL I/O, response parsing utilities
```

---

## 3. Installation

Install dependencies from the source requirements file:

```bash
pip install -r src/requirements.txt
```

> Note: the provided requirements list is comprehensive and includes GPU/vLLM-related packages. For lightweight reproduction, you may create a reduced environment containing only the libraries needed by your specific model backends and retriever settings.

---

## 4. Data and Resource Assumptions

The current `src/main_mp.py` pipeline expects several resources:

1. **Event and user profile JSONL files** (sequence-level events + user IDs).
2. **Preference dimension template JSON** (used by assistant summarization).
3. **Retriever embedding model** for vector memory.
4. **LLM backends** for user and assistant models (API or vLLM endpoints).

In the released code, some default paths are hard-coded to an internal filesystem. For external deployment, replace them with your local paths before running experiments.

---

## 5. Running the Main Pipeline

### 5.1 Canonical command template

Use `src/main_mp.py` as the main entrypoint:

```bash
python src/main_mp.py \
  --user_model_path /path/to/user/model_or_name \
  --user_model_url http://0.0.0.0:8000/v1 \
  --user_model_api_key <USER_API_KEY> \
  --assistant_model_path gpt-5 \
  --assistant_model_url https://api.example.com/v1 \
  --assistant_model_api_key <ASSISTANT_API_KEY> \
  --chromadb_root /path/to/chromadb_root \
  --logs_root /path/to/logs_root \
  --n_events_per_sequence 10 \
  --n_threads 4 \
  --retriever_model_path /path/to/embedding/model
```

### 5.2 Example aligned with your script style

```bash
python /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/ai_assistant/main_mp.py \
    --user_model_path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-32B \
    --user_model_url http://0.0.0.0:8000/v1 \
    --assistant_model_path gpt-5 \
    --assistant_model_url https://api.ai-gaochao.cn/v1 \
    --assistant_model_api_key <REDACTED_API_KEY> \
    --chromadb_root /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/chromadb \
    --logs_root /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/exp_results/logs_long_session \
    --n_events_per_sequence 10 \
    --n_threads 4 \
    --retriever_model_path /inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B
```

> Security note: do **not** commit real API keys into scripts or public READMEs.

---

## 6. CLI Arguments (`src/main_mp.py`)

| Argument | Type | Description |
|---|---|---|
| `--user_model_path` | `str` | User model name/path; basename is used to determine model type. |
| `--user_model_url` | `str` | User model API endpoint (typically OpenAI-compatible vLLM URL). |
| `--user_model_api_key` | `str` | API key for user model backend (default: `123`). |
| `--assistant_model_path` | `str` | Assistant model name/path. |
| `--assistant_model_url` | `str` | Assistant model API endpoint. |
| `--assistant_model_api_key` | `str` | API key for assistant backend (default: `123`). |
| `--use_preference_memory` | flag | Enables assistant profile memory during summarization. |
| `--chromadb_root` | `str` | Root directory for Chroma persistence. |
| `--logs_root` | `str` | Root directory for simulation logs and model traces. |
| `--seq_ids` | `str` | Comma-separated sequence IDs; if omitted, all sequences are processed. |
| `--retriever_model_path` | `str` | Embedding model path for retriever-backed memory. |
| `--n_events_per_sequence` | `int` | Number of events simulated per sequence. |
| `--n_threads` | `int` | Number of worker threads in the simulation pool. |

---

## 7. Execution Workflow

At runtime, `src/main_mp.py` performs the following procedure:

1. Sets deterministic seeds for Python/NumPy/PyTorch.
2. Parses CLI arguments and builds experiment naming metadata.
3. Loads event sequences and user profiles.
4. Initializes user and assistant model wrappers via `models.load_model`.
5. Creates `ConversationSimulator` instances (one per sequence/thread execution).
6. For each event episode:
   - event context is generated,
   - user responds (optionally with dynamic memory and emotion chain),
   - assistant predicts intent and replies,
   - assistant summarizes inferred preference dimensions.
7. Persists logs, agent message histories, and model call traces.

---

## 8. Output Artifacts

Typical outputs under `--logs_root` include:

- `sim_log_*.json`: event metadata + dialogue traces,
- `user_sim_log_*.json` / `assistant_sim_log_*.json`: per-agent serialized histories,
- `user_model.jsonl` / `assistant_model.jsonl`: model interaction records,
- Chroma collections under `--chromadb_root` for retriever memories.

---

## 9. Optional Demo Interface

The repository also contains a `demo/` directory with a UI-oriented workflow (`demo/run.sh`, Streamlit/Flask utilities). This path is useful for interactive inspection, while `src/` should be treated as the principal experiment code for reproducible studies.

---

## 10. Reproducibility Recommendations

For publication-grade experiments, we recommend:

- fixed random seeds and explicit reporting of seed values,
- immutable snapshots of event/user datasets,
- versioned model checkpoints and backend endpoints,
- full logging of runtime configuration and commit hash,
- strict separation between development and evaluation sequence splits.

---

## 11. Citation

If you use this codebase in academic work, please cite the corresponding paper/project artifact (to be added by maintainers).
