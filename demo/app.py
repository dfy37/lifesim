import argparse
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import uuid
from typing import Dict, List, Optional

import yaml
from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context


DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DEMO_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from profiles.profile_generator import UserProfile  # noqa: E402
from utils.utils import load_jsonl_data  # noqa: E402

try:
    from models import load_model  # noqa: E402
    from agents.user_agent import UserAgent  # noqa: E402
except Exception:
    load_model = None
    UserAgent = None

try:
    from engine.event_engine import OnlineLifeEventEngine  # noqa: E402
except Exception:
    OnlineLifeEventEngine = None


def build_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(DEMO_DIR, "templates"),
        static_folder=os.path.join(DEMO_DIR, "static"),
    )
    app.secret_key = os.environ.get("LIFESIM_DEMO_SECRET", "lifesim-demo-secret")
    app.config["JSON_AS_ASCII"] = False
    app.config["SESSION_STORE"] = {}
    app.config["DEMO_CONFIG"] = build_runtime_config()

    register_routes(app)
    return app


def build_runtime_config() -> Dict:
    return {
        "data": {
            "events_path": os.path.join(REPO_ROOT, "data", "single_session", "events.jsonl"),
            "users_path": os.path.join(REPO_ROOT, "data", "single_session", "users.jsonl"),
            "live_trajectories_path": os.path.join(DEMO_DIR, "data", "live_trajectories.jsonl"),
            "user_names_path": os.path.join(DEMO_DIR, "data", "user_names.json"),
        },
        "chat_backend": {
            "config_path": os.path.join(REPO_ROOT, "config.yaml"),
        },
    }


def load_user_names(app: Flask) -> Dict[str, str]:
    """Return {sequence_id: display_name} mapping, cached on the app object."""
    if not hasattr(app, "_user_names"):
        path = app.config["DEMO_CONFIG"]["data"]["user_names_path"]
        try:
            with open(path, encoding="utf-8") as f:
                app._user_names = json.load(f)
        except Exception:
            app._user_names = {}
    return app._user_names


def get_demo_state(app: Flask) -> Dict:
    if "demo_session_id" not in session:
        session["demo_session_id"] = str(uuid.uuid4())
    session_id = session["demo_session_id"]
    store = app.config["SESSION_STORE"]
    if session_id not in store:
        store[session_id] = {
            # Preset mode
            "selected_sequence_id": None,
            "selected_node_index": None,
            "node_chat_histories": {},
            "node_user_agents": {},
            "node_emotion_histories": {},
            "custom_profile": None,
            # Live generation mode
            "live_sequence_id": None,
            "live_user_profile": None,
            "live_sequence_meta": None,
            "live_event_index": 0,
            "live_generated_events": [],
            "live_history_events": [],
            "live_goal": "",
            "live_node_chat_histories": {},
            "live_node_emotion_histories": {},
        }
    return store[session_id]


def load_demo_data(app: Flask) -> Dict:
    cfg = app.config["DEMO_CONFIG"]
    events = load_jsonl_data(cfg["data"]["events_path"])
    users = load_jsonl_data(cfg["data"]["users_path"])

    users_by_id = {str(user["user_id"]): user for user in users}
    events_by_sequence = {event["id"]: event for event in events}
    ordered_sequence_ids = [event["id"] for event in events]

    return {
        "events_by_sequence": events_by_sequence,
        "users_by_id": users_by_id,
        "sequence_ids": ordered_sequence_ids,
    }


def normalize_weather(weather) -> str:
    if isinstance(weather, dict):
        return weather.get("description", "")
    return weather or ""


def extract_motivation(node: Dict) -> Dict[str, List[str]]:
    explicit, implicit = [], []
    for item in node.get("sub_intents", []):
        description = item.get("description")
        if not description:
            continue
        if item.get("type") == "explicit":
            explicit.append(description)
        else:
            implicit.append(description)
    return {"explicit": explicit, "implicit": implicit}


def build_dialogue_scene(node: Dict) -> str:
    parts = []
    if node.get("time"):
        parts.append(f"The time is {node['time']}")
    if node.get("location"):
        parts.append(f"The location is {node['location']}")
    weather = normalize_weather(node.get("weather"))
    if weather:
        parts.append(f"The weather condition is {weather}")
    return "\n".join(parts)


def serialize_node(node: Dict, node_index: int) -> Dict:
    location_detail = node.get("location_detail") or {}
    return {
        "node_index": node_index,
        "time": node.get("time"),
        "location": node.get("location"),
        "location_detail": {
            "latitude": location_detail.get("latitude"),
            "longitude": location_detail.get("longitude"),
        } if location_detail else None,
        "weather": normalize_weather(node.get("weather")),
        "event": node.get("event"),
        "life_event": node.get("life_event") or node.get("event"),
        "intent": node.get("intent"),
        "sub_intents": node.get("sub_intents", []),
        "dialogue_scene": node.get("dialogue_scene") or build_dialogue_scene(node),
        "motivation": extract_motivation(node),
    }


def serialize_trajectory(app: Flask, sequence_id: str) -> Optional[Dict]:
    data = load_demo_data(app)
    sequence = data["events_by_sequence"].get(sequence_id)
    if not sequence:
        return None

    user_id = str(sequence["user_id"])
    user_profile = data["users_by_id"].get(user_id, {})
    nodes = [serialize_node(node, idx) for idx, node in enumerate(sequence.get("events", []))]

    return {
        "success": True,
        "user_profile": user_profile,
        "sequence": {
            "sequence_id": sequence["id"],
            "user_id": user_id,
            "theme": sequence.get("theme", ""),
            "longterm_goal": sequence.get("longterm_goal", ""),
        },
        "nodes": nodes,
    }


def load_chat_config(app: Flask) -> Optional[Dict]:
    config_path = app.config["DEMO_CONFIG"]["chat_backend"]["config_path"]
    if not config_path or not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def format_profile_text(profile: Dict) -> str:
    if not profile:
        return "No profile available."
    try:
        return str(UserProfile.from_dict(profile))
    except Exception:
        return json.dumps(profile, ensure_ascii=False, indent=2)


def build_timeline_summary(nodes: List[Dict], selected_index: int) -> str:
    summaries = []
    for node in nodes[: selected_index + 1]:
        summaries.append(
            f"[Node {node['node_index'] + 1}] Time: {node.get('time') or 'N/A'} | "
            f"Location: {node.get('location') or 'N/A'} | Event: {node.get('event') or 'N/A'} | "
            f"Intent: {node.get('intent') or 'N/A'}"
        )
    return "\n".join(summaries)


def build_node_messages(
    profile: Dict,
    trajectory: Dict,
    selected_node: Dict,
    history: List[Dict],
    user_message: str,
) -> List[Dict]:
    profile_text = format_profile_text(profile)
    timeline_summary = build_timeline_summary(trajectory["nodes"], selected_node["node_index"])
    motivation = selected_node.get("motivation", {})
    explicit = "\n".join(f"- {item}" for item in motivation.get("explicit", [])) or "- None"
    implicit = "\n".join(f"- {item}" for item in motivation.get("implicit", [])) or "- None"

    system_prompt = f"""
You are simulating a user inside LifeSim.

Stay fully in character as the user.
Your response must be grounded in:
- the user's profile
- the selected life node
- the current motivation at this time
- the prior trajectory context

Do not explain the simulator.
Do not role-switch into the assistant.
Reply as the user naturally would in this situation.
Keep the response concise, specific, and consistent with the life context.

[User Profile]
{profile_text}

[Trajectory Context]
{timeline_summary}

[Selected Node]
Time: {selected_node.get('time') or 'N/A'}
Location: {selected_node.get('location') or 'N/A'}
Weather: {selected_node.get('weather') or 'N/A'}
Event: {selected_node.get('event') or 'N/A'}
Intent: {selected_node.get('intent') or 'N/A'}

[Explicit Motivation]
{explicit}

[Implicit Motivation]
{implicit}
""".strip()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def get_or_create_user_agent(
    app: Flask,
    state: Dict,
    sequence_id: str,
    node_index: int,
    node: Dict,
    user_profile: Dict,
) -> Optional[object]:
    """Lazily create and cache a UserAgent for each (sequence_id, node_index) pair."""
    if not UserAgent or not load_model:
        return None

    key = f"{sequence_id}:{node_index}"
    if key in state["node_user_agents"]:
        return state["node_user_agents"][key]

    chat_cfg = load_chat_config(app)
    if not chat_cfg:
        return None

    try:
        user_model_cfg = chat_cfg["models"]["user_model"]
        model_name = os.path.basename(user_model_cfg["model_path"])
        model = load_model(
            model_name=model_name,
            api_key=user_model_cfg["api_key"],
            model_path=user_model_cfg["model_path"],
            base_url=user_model_cfg["base_url"],
            vllmapi=user_model_cfg["vllmapi"],
        )

        session_id = session.get("demo_session_id", "demo")
        safe_key = key.replace(":", "_").replace("/", "_")
        retriever_config = {
            "model_name": "",
            "collection_name": f"demo_{session_id[:8]}_{safe_key}",
            "max_length": 512,
            "embedding_dim": 384,
            "persist_directory": os.path.join(tempfile.gettempdir(), "lifesim_demo_chroma"),
            "device": "cpu",
            "use_custom_embeddings": False,
            "logger_silent": True,
        }

        profile_str = format_profile_text(user_profile)
        agent = UserAgent(
            model=model,
            retriever_config=retriever_config,
            profile=profile_str,
            alpha=0.5,
            logger_silent=True,
        )

        # Pass node dict directly; _build_environment just stores self.event = event
        agent._build_environment(node)
        agent._build_chat_system_prompt()

        state["node_user_agents"][key] = agent
        return agent
    except Exception:
        return None


def generate_fallback_reply(selected_node: Dict, user_message: str) -> str:
    motivation = selected_node.get("motivation", {})
    explicit = motivation.get("explicit") or []
    implicit = motivation.get("implicit") or []
    concern = explicit[0] if explicit else (implicit[0] if implicit else "solve the issue in front of me")
    location = selected_node.get("location") or "this place"
    time = selected_node.get("time") or "right now"
    intent = selected_node.get("intent") or "handle the current situation"
    return (
        f"I'm focused on {concern.lower()} at {location} around {time}. "
        f"Given what's happening, I mainly want help to {intent.lower()}. "
        f"On your last message, my immediate reaction is: {user_message[:120]}"
    )


def query_via_agent(agent, user_message: str) -> Dict:
    """Call UserAgent.respond() and normalize the result."""
    try:
        result = agent.respond(
            prompt=user_message,
            use_emotion_chain=True,
            use_dynamic_memory=False,
        )
        action_value = result["action"].value if hasattr(result["action"], "value") else str(result["action"])
        return {
            "mode": "agent",
            "response": result.get("response", ""),
            "emotion": result.get("emotion", "neutral"),
            "action": action_value,
        }
    except Exception:
        return None


def query_user_model(
    app: Flask,
    state: Dict,
    profile: Dict,
    trajectory: Dict,
    selected_node: Dict,
    history: List[Dict],
    user_message: str,
) -> Dict:
    sequence_id = trajectory["sequence"]["sequence_id"]
    node_index = selected_node["node_index"]

    agent = get_or_create_user_agent(app, state, sequence_id, node_index, selected_node, profile)
    if agent:
        result = query_via_agent(agent, user_message)
        if result:
            return result

    # Fallback: plain LLM call without UserAgent
    chat_cfg = load_chat_config(app)
    if chat_cfg and load_model:
        try:
            user_model_cfg = chat_cfg["models"]["user_model"]
            model_name = os.path.basename(user_model_cfg["model_path"])
            model = load_model(
                model_name=model_name,
                api_key=user_model_cfg["api_key"],
                model_path=user_model_cfg["model_path"],
                base_url=user_model_cfg["base_url"],
                vllmapi=user_model_cfg["vllmapi"],
            )
            messages = build_node_messages(profile, trajectory, selected_node, history, user_message)
            reply = model.chat(messages)
            if reply:
                return {"mode": "model", "response": reply, "emotion": "neutral", "action": "Continue Conversation"}
        except Exception:
            pass

    return {
        "mode": "fallback",
        "response": generate_fallback_reply(selected_node, user_message),
        "emotion": "neutral",
        "action": "Continue Conversation",
    }


class _EngineLogCapture:
    """Capture engine logger messages by monkey-patching logger methods."""

    def __init__(self, logger):
        self.lines: List[str] = []
        self._logger = logger
        self._orig_info    = logger.info
        self._orig_warning = getattr(logger, 'warning', None)
        self._orig_warn    = getattr(logger, 'warn', None)

        def _cap_info(msg, *a, **kw):
            self.lines.append(str(msg))
            self._orig_info(msg, *a, **kw)

        def _cap_warn(msg, *a, **kw):
            self.lines.append(f"[warn] {msg}")
            if self._orig_warning:
                self._orig_warning(msg, *a, **kw)

        logger.info    = _cap_info
        if self._orig_warning:
            logger.warning = _cap_warn
        if self._orig_warn:
            logger.warn = _cap_warn

    def restore(self):
        self._logger.info    = self._orig_info
        if self._orig_warning:
            self._logger.warning = self._orig_warning
        if self._orig_warn:
            self._logger.warn    = self._orig_warn


def load_live_data(app: Flask) -> Dict:
    cfg = app.config["DEMO_CONFIG"]
    path = cfg["data"]["live_trajectories_path"]
    if not os.path.exists(path):
        return {"live_by_id": {}, "sequence_ids": []}
    sequences = load_jsonl_data(path)
    return {
        "live_by_id": {seq["id"]: seq for seq in sequences},
        "sequence_ids": [seq["id"] for seq in sequences],
    }


def _build_retriever_index(app: Flask):
    """Build a DenseRetriever from all events in events.jsonl and return it."""
    try:
        from tools.dense_retriever import DenseRetriever  # noqa: E402
    except Exception:
        return None
    try:
        retriever = DenseRetriever(
            model_name="all-MiniLM-L6-v2",
            collection_name="lifesim_demo_events_v1",
            persist_directory=os.path.join(tempfile.gettempdir(), "lifesim_demo_retriever"),
            use_custom_embeddings=False,
            logger_silent=True,
        )
        if retriever.is_collection_empty():
            # Flatten all individual events from every sequence into a single list
            data = load_demo_data(app)
            docs = []
            for seq_id, seq in data["events_by_sequence"].items():
                for i, ev in enumerate(seq.get("events", [])):
                    text = ev.get("event") or ev.get("life_event", "")
                    if not text:
                        continue
                    docs.append({
                        "doc_id": f"{seq_id}_{i}",
                        "event": text,
                        "intent": ev.get("intent", ""),
                    })
            if docs:
                retriever.build_index(docs, text_field="event", id_field="doc_id",
                                      clear_existing=True)
        return retriever
    except Exception:
        return None


def get_live_engine(app: Flask) -> Optional[object]:
    """Return the shared OnlineLifeEventEngine with model + retriever attached."""
    if not OnlineLifeEventEngine:
        return None
    if not hasattr(app, "_live_engine"):
        cfg = app.config["DEMO_CONFIG"]
        path = cfg["data"]["events_path"]
        if not os.path.exists(path):
            app._live_engine = None
            return None
        try:
            app._live_engine = OnlineLifeEventEngine(
                event_sequences_path=path,
                model=None,
                retriever=None,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            app._live_engine = None

    engine = app._live_engine
    if engine is None:
        return None

    # Attach model (once)
    if engine.model is None:
        chat_cfg = load_chat_config(app)
        if chat_cfg and load_model:
            try:
                cfg = chat_cfg["models"]["user_model"]
                engine.model = load_model(
                    model_name=os.path.basename(cfg["model_path"]),
                    api_key=cfg["api_key"],
                    model_path=cfg["model_path"],
                    base_url=cfg["base_url"],
                    vllmapi=cfg["vllmapi"],
                )
                print(f"[live_engine] model loaded: {cfg['model_path']}")
            except Exception as e:
                print(f"[live_engine] model load failed: {e}")

    # Attach retriever (once) — built from existing events.jsonl
    if engine.retriever is None:
        engine.retriever = _build_retriever_index(app)
        if engine.retriever:
            print(f"[live_engine] retriever ready, docs={engine.retriever.get_stats().get('total_docs')}")
        else:
            print("[live_engine] retriever build failed")

    return engine


def build_live_trajectory_context(state: Dict) -> Dict:
    """Build a trajectory-shaped dict from live session state for use with query_user_model."""
    meta = state.get("live_sequence_meta") or {}
    return {
        "sequence": {
            "sequence_id": f"live:{state.get('live_sequence_id', '')}",
            "theme": meta.get("theme", ""),
            "longterm_goal": meta.get("longterm_goal", ""),
        },
        "nodes": state.get("live_generated_events", []),
        "user_profile": state.get("live_user_profile") or {},
    }


def register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/trajectory")
    def trajectory():
        data = load_demo_data(app)
        names = load_user_names(app)
        default_sequence_id = data["sequence_ids"][0] if data["sequence_ids"] else None
        traj = serialize_trajectory(app, default_sequence_id) if default_sequence_id else None
        return render_template(
            "trajectory_demo.html",
            sequence_ids=data["sequence_ids"],
            user_names=names,
            default_sequence_id=default_sequence_id,
            initial_trajectory=traj,
        )

    @app.route("/api/trajectory/<sequence_id>")
    def get_trajectory(sequence_id: str):
        traj = serialize_trajectory(app, sequence_id)
        if not traj:
            return jsonify({"success": False, "error": "Sequence not found"}), 404
        return jsonify(traj)

    @app.route("/api/select-node", methods=["POST"])
    def select_node():
        payload = request.get_json(force=True)
        sequence_id = payload.get("sequence_id")
        node_index = payload.get("node_index")
        traj = serialize_trajectory(app, sequence_id)
        if not traj:
            return jsonify({"success": False, "error": "Sequence not found"}), 404
        if not isinstance(node_index, int) or node_index < 0 or node_index >= len(traj["nodes"]):
            return jsonify({"success": False, "error": "Invalid node index"}), 400

        state = get_demo_state(app)
        state["selected_sequence_id"] = sequence_id
        state["selected_node_index"] = node_index
        state["node_chat_histories"].setdefault(f"{sequence_id}:{node_index}", [])

        return jsonify({
            "success": True,
            "selected_node": traj["nodes"][node_index],
            "chat_history": state["node_chat_histories"][f"{sequence_id}:{node_index}"],
        })

    @app.route("/api/node-chat", methods=["POST"])
    def node_chat():
        payload = request.get_json(force=True)
        sequence_id = payload.get("sequence_id")
        node_index = payload.get("node_index")
        message = (payload.get("message") or "").strip()

        if not message:
            return jsonify({"success": False, "error": "Message is required"}), 400

        traj = serialize_trajectory(app, sequence_id)
        if not traj:
            return jsonify({"success": False, "error": "Sequence not found"}), 404
        if not isinstance(node_index, int) or node_index < 0 or node_index >= len(traj["nodes"]):
            return jsonify({"success": False, "error": "Invalid node index"}), 400

        state = get_demo_state(app)
        key = f"{sequence_id}:{node_index}"
        history = state["node_chat_histories"].setdefault(key, [])
        selected_node = traj["nodes"][node_index]
        profile = state.get("custom_profile") or traj["user_profile"]

        result = query_user_model(
            app=app,
            state=state,
            profile=profile,
            trajectory=traj,
            selected_node=selected_node,
            history=history,
            user_message=message,
        )

        history.append({"role": "user", "content": message})
        if result["response"]:
            history.append({"role": "assistant", "content": result["response"]})

        emotion = result.get("emotion", "neutral")
        action = result.get("action", "Continue Conversation")
        state["node_emotion_histories"].setdefault(key, []).append(emotion)

        return jsonify({
            "success": True,
            "mode": result["mode"],
            "response": result["response"],
            "emotion": emotion,
            "action": action,
            "chat_history": history,
        })

    @app.route("/api/update-profile", methods=["POST"])
    def update_profile():
        payload = request.get_json(force=True)
        profile = payload.get("profile")  # None means reset to original
        if profile is not None and not isinstance(profile, dict):
            return jsonify({"success": False, "error": "Invalid profile"}), 400
        state = get_demo_state(app)
        state["custom_profile"] = profile
        # Clear cached agents so they are recreated with the new profile
        state["node_user_agents"] = {}
        state["node_chat_histories"] = {}
        state["node_emotion_histories"] = {}
        return jsonify({"success": True})

    @app.route("/api/clear-session", methods=["POST"])
    def clear_session():
        state = get_demo_state(app)
        state["selected_sequence_id"] = None
        state["selected_node_index"] = None
        state["node_chat_histories"] = {}
        state["node_user_agents"] = {}
        state["node_emotion_histories"] = {}
        state["custom_profile"] = None
        return jsonify({"success": True})

    # ── Live Generation routes ─────────────────────────────────────────────

    @app.route("/live")
    def live():
        data = load_demo_data(app)
        names = load_user_names(app)
        return render_template(
            "live_demo.html",
            sequence_ids=data["sequence_ids"],
            user_names=names,
        )

    @app.route("/api/live/sequences")
    def live_sequences():
        data = load_demo_data(app)
        result = []
        for seq_id in data["sequence_ids"]:
            seq = data["events_by_sequence"][seq_id]
            result.append({
                "id": seq_id,
                "theme": seq.get("theme", ""),
                "longterm_goal": seq.get("longterm_goal", ""),
                "total_points": len(seq.get("events", [])),
            })
        return jsonify({"success": True, "sequences": result})

    @app.route("/api/live/start", methods=["POST"])
    def live_start():
        payload = request.get_json(force=True)
        seq_id = payload.get("sequence_id")
        data = load_demo_data(app)
        seq = data["events_by_sequence"].get(seq_id)
        if not seq:
            return jsonify({"success": False, "error": "Sequence not found"}), 404

        user_id = str(seq["user_id"])
        user_profile = data["users_by_id"].get(user_id, {})

        state = get_demo_state(app)
        # Reset live state for new sequence
        state["live_sequence_id"] = seq_id
        state["live_user_profile"] = user_profile
        state["live_sequence_meta"] = {
            "theme": seq.get("theme", ""),
            "longterm_goal": seq.get("longterm_goal", ""),
        }
        state["live_event_index"] = 0
        state["live_generated_events"] = []
        state["live_history_events"] = []
        state["live_goal"] = seq.get("longterm_goal", "")
        state["live_node_chat_histories"] = {}
        state["live_node_emotion_histories"] = {}
        # Clear any live agents from shared agent cache
        state["node_user_agents"] = {
            k: v for k, v in state["node_user_agents"].items()
            if not k.startswith("live:")
        }

        events = seq.get("events", [])
        first_location = None
        if events:
            loc = events[0].get("location_detail", {}) or {}
            first_location = {
                "location": events[0].get("location"),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
            }

        return jsonify({
            "success": True,
            "user_profile": state["live_user_profile"],
            "sequence": {
                "sequence_id": seq_id,
                "theme": seq.get("theme", ""),
                "longterm_goal": seq.get("longterm_goal", ""),
            },
            "total_points": len(events),
            "first_location": first_location,
        })

    @app.route("/api/live/generate-event", methods=["POST"])
    def live_generate_event():
        payload = request.get_json(force=True) or {}
        state = get_demo_state(app)
        seq_id = state.get("live_sequence_id")
        if not seq_id:
            return jsonify({"success": False, "error": "No live sequence started"}), 400

        demo_data = load_demo_data(app)
        seq = demo_data["events_by_sequence"].get(seq_id)
        if not seq:
            return jsonify({"success": False, "error": "Sequence not found"}), 404

        events = seq.get("events", [])
        current_index = state["live_event_index"]
        if current_index >= len(events):
            return jsonify({"success": False, "error": "All events already generated"}), 400

        engine = get_live_engine(app)
        profile_text = format_profile_text(
            state.get("custom_profile") or state["live_user_profile"]
        )
        goal = payload.get("goal") or state.get("live_goal", "")
        history_events = list(state["live_history_events"])

        # Queue for log lines from the background generation thread
        log_q: queue.Queue = queue.Queue()
        result_holder: Dict = {}

        if engine:
            log_capture = _EngineLogCapture(engine.logger)

            def _push_log(msg):
                log_q.put({"type": "log", "line": msg})

            # Override capture to also push to queue for real-time streaming
            _orig_cap_info = engine.logger.info
            def _stream_info(msg, *a, **kw):
                _push_log(str(msg))
                _orig_cap_info(msg, *a, **kw)
            engine.logger.info = _stream_info

            def run_gen():
                try:
                    engine.set_event_sequence(seq_id)
                    engine.set_event_index(current_index)
                    result_holder["event"] = engine.generate_event(
                        user_profile=profile_text,
                        history_events=history_events,
                        goal=goal,
                    )
                    result_holder["success"] = True
                except Exception as e:
                    result_holder["error"] = str(e)
                    result_holder["success"] = False
                finally:
                    log_capture.restore()
                    log_q.put({"type": "done"})

            threading.Thread(target=run_gen, daemon=True).start()
        else:
            result_holder["event"] = dict(events[current_index])
            if not result_holder["event"].get("life_event"):
                result_holder["event"]["life_event"] = result_holder["event"].get("event", "")
            result_holder["success"] = True
            log_q.put({"type": "log", "line": "[warn] no engine available — using preset event"})
            log_q.put({"type": "done"})

        def generate():
            # Stream log lines in real time until generation is done
            while True:
                item = log_q.get()
                if item["type"] == "done":
                    break
                yield f"data: {json.dumps(item)}\n\n"

            # Update session state and send final result
            new_event = result_holder.get("event", dict(events[current_index]))
            if not new_event.get("life_event"):
                new_event["life_event"] = new_event.get("event", "")

            node_index = current_index
            state["live_event_index"] += 1
            state["live_history_events"].append(new_event)
            serialized = serialize_node(new_event, node_index)
            state["live_generated_events"].append(serialized)

            total = len(events)
            next_location = None
            if state["live_event_index"] < total:
                nxt = events[state["live_event_index"]]
                loc = nxt.get("location_detail", {}) or {}
                next_location = {
                    "location": nxt.get("location"),
                    "latitude": loc.get("latitude"),
                    "longitude": loc.get("longitude"),
                }

            result_msg = {
                "type": "result",
                "success": result_holder.get("success", False),
                "node": serialized,
                "node_index": node_index,
                "progress": {"current": state["live_event_index"], "total": total},
                "next_location": next_location,
                "done": state["live_event_index"] >= total,
            }
            if "error" in result_holder:
                result_msg["error"] = result_holder["error"]

            yield f"data: {json.dumps(result_msg)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/live/select-node", methods=["POST"])
    def live_select_node():
        payload = request.get_json(force=True)
        node_index = payload.get("node_index")
        state = get_demo_state(app)
        generated = state.get("live_generated_events", [])
        if not isinstance(node_index, int) or node_index < 0 or node_index >= len(generated):
            return jsonify({"success": False, "error": "Invalid node index"}), 400
        state["live_node_chat_histories"].setdefault(str(node_index), [])
        return jsonify({
            "success": True,
            "selected_node": generated[node_index],
            "chat_history": state["live_node_chat_histories"][str(node_index)],
        })

    @app.route("/api/live/chat", methods=["POST"])
    def live_chat():
        payload = request.get_json(force=True)
        node_index = payload.get("node_index")
        message = (payload.get("message") or "").strip()
        if not message:
            return jsonify({"success": False, "error": "Message is required"}), 400

        state = get_demo_state(app)
        generated = state.get("live_generated_events", [])
        if not isinstance(node_index, int) or node_index < 0 or node_index >= len(generated):
            return jsonify({"success": False, "error": "Invalid node index"}), 400

        selected_node = generated[node_index]
        profile = state.get("custom_profile") or state.get("live_user_profile") or {}
        traj = build_live_trajectory_context(state)
        key = str(node_index)
        history = state["live_node_chat_histories"].setdefault(key, [])

        result = query_user_model(
            app=app,
            state=state,
            profile=profile,
            trajectory=traj,
            selected_node=selected_node,
            history=history,
            user_message=message,
        )

        history.append({"role": "user", "content": message})
        if result["response"]:
            history.append({"role": "assistant", "content": result["response"]})

        emotion = result.get("emotion", "neutral")
        action = result.get("action", "Continue Conversation")
        state["live_node_emotion_histories"].setdefault(key, []).append(emotion)

        return jsonify({
            "success": True,
            "mode": result["mode"],
            "response": result["response"],
            "emotion": emotion,
            "action": action,
        })

    @app.route("/api/live/reset", methods=["POST"])
    def live_reset():
        state = get_demo_state(app)
        state["live_sequence_id"] = None
        state["live_user_profile"] = None
        state["live_sequence_meta"] = None
        state["live_event_index"] = 0
        state["live_generated_events"] = []
        state["live_history_events"] = []
        state["live_goal"] = ""
        state["live_node_chat_histories"] = {}
        state["live_node_emotion_histories"] = {}
        state["node_user_agents"] = {
            k: v for k, v in state["node_user_agents"].items()
            if not k.startswith("live:")
        }
        return jsonify({"success": True})

    @app.route("/healthz")
    def healthz():
        return Response("ok", mimetype="text/plain")


def parse_args():
    parser = argparse.ArgumentParser(description="LifeSim trajectory-first demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5020, type=int)
    parser.add_argument("--events-path", default=None)
    parser.add_argument("--users-path", default=None)
    parser.add_argument("--config", default=None, help="Optional config.yaml for real user-model chat")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app()

    if args.events_path:
        app.config["DEMO_CONFIG"]["data"]["events_path"] = args.events_path
    if args.users_path:
        app.config["DEMO_CONFIG"]["data"]["users_path"] = args.users_path
    if args.config:
        app.config["DEMO_CONFIG"]["chat_backend"]["config_path"] = args.config

    app.run(host=args.host, port=args.port, debug=False)
