# LifeSim: Long-Horizon User Life Simulator for Personalized Assistant Evaluation

**Feiyu Duan<sup>1</sup>, Xuanjing Huang<sup>2</sup>, Zhongyu Wei<sup>1,3*</sup>**

<sup>1</sup> School of Data Science, Fudan University, China  
<sup>2</sup> School of Computer Science, Fudan University, China  
<sup>3</sup> Shanghai Innovation Institute  

✉️ fyduan25@m.fudan.edu.cn， {xjhuang, zywei}@fudan.edu.cn

This repository provides a research-oriented simulation framework for studying long-horizon interactions between a **user agent** and an **assistant agent** under structured life-event trajectories. The codebase is designed for controlled experimentation on preference inference, memory dynamics, and dialogue adaptation.

![LifeSim Framework](resources/framework.png)

<video width="600" controls>
  <source src="resources/lifesimDemo_with_subtitle.mov" type="video/mp4">
</video>

## 1. Research Objective

LifeSim operationalizes a sequential interaction problem in which:

- a user profile is instantiated from structured demographic and preference attributes,
- a life-event engine emits event contexts over time,
- a user agent generates responses conditioned on profile, event context, and optional memory/emotion mechanisms,
- an assistant agent predicts intent, responds, and periodically summarizes user preferences.

This setup enables evaluation of longitudinal assistant behavior under non-stationary conversational contexts.

## 2. Code Structure

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

## 3. Installation

Install dependencies from the source requirements file:

```bash
pip install -r src/requirements.txt
```

> Note: the provided requirements list is comprehensive and includes GPU/vLLM-related packages. For lightweight reproduction, you may create a reduced environment containing only the libraries needed by your specific model backends and retriever settings.

## 4. Data and Resource Assumptions

The current `src/main_mp.py` pipeline expects several resources:

1. **Event and user profile JSONL files** (sequence-level events + user IDs).
2. **Preference dimension template JSON** (used by assistant summarization).
3. **Retriever embedding model** for vector memory.
4. **LLM backends** for user and assistant models (API or vLLM endpoints).

In the released code, some default paths are hard-coded to an internal filesystem. For external deployment, replace them with your local paths before running experiments.

## 5. Running the Main Pipeline

(Events)

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

(Evaluation)

## 6. Citation

If you use this codebase in academic work, please cite the corresponding paper/project artifact (to be added by maintainers).
