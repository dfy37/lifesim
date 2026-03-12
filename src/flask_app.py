"""
LifeSim - Flask Application
A user life cycle simulator with web UI
"""
import os
import json
import yaml
import time
import uuid
import argparse
import threading
import queue
from datetime import datetime

from flask import Flask, render_template, request, jsonify, session, Response
from flask_cors import CORS

from utils.utils import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config.yaml"

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# In-memory session storage (use Redis or a database in production)
simulation_sessions = {}


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl_data(path):
    data = []
    if os.path.exists(path):
        with open(path) as f:
            for row in f:
                data.append(json.loads(row))
    return data


def save_jsonl_data(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_or_create_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']
    if session_id not in simulation_sessions:
        simulation_sessions[session_id] = {
            'timeline_events': [],
            'message_blocks': [],
            'history_messages': [],
            'event_counter': 0,
            'simulator_running': False,
            'simulation_results': [],
        }
    return simulation_sessions[session_id]


# =====================================
# Routes
# =====================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/assistant-eval')
def assistant_eval():
    sim_session = get_or_create_session()
    try:
        cfg = load_config(CONFIG_PATH)
        events = load_jsonl_data(cfg["paths"]["events_path"])
        users = load_jsonl_data(cfg["paths"]["users_path"])

        sequence_ids = [e['id'] for e in events]
        seqid2uid = {e['id']: e['user_id'] for e in events}
        seqid2eseq = {e['id']: e for e in events}
        uid2user = {u['user_id']: u for u in users}

        default_seq_id = sequence_ids[0] if sequence_ids else None
        default_user = uid2user.get(seqid2uid.get(default_seq_id)) if default_seq_id else None
        default_events = seqid2eseq.get(default_seq_id, {}).get('events', []) if default_seq_id else []

        assistant_models = cfg.get("assistant_models", ['deepseek-chat'])

        return render_template('assistant_eval.html',
                               sequence_ids=sequence_ids,
                               selected_seq_id=default_seq_id,
                               user_profile=default_user,
                               event_sequence=default_events,
                               assistant_models=assistant_models,
                               simulation_results=sim_session.get('simulation_results', []))
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return render_template('assistant_eval.html',
                               sequence_ids=[],
                               selected_seq_id=None,
                               user_profile=None,
                               event_sequence=[],
                               assistant_models=['deepseek-chat'],
                               simulation_results=[])


@app.route('/user-life')
def user_life():
    sim_session = get_or_create_session()
    try:
        cfg = load_config(CONFIG_PATH)
        events = load_jsonl_data(cfg["paths"]["events_path"])
        users = load_jsonl_data(cfg["paths"]["users_path"])

        sequence_ids = [e['id'] for e in events]
        seqid2uid = {e['id']: e['user_id'] for e in events}
        uid2user = {u['user_id']: u for u in users}

        default_seq_id = sequence_ids[0] if sequence_ids else None
        default_user = uid2user.get(seqid2uid.get(default_seq_id)) if default_seq_id else None

        return render_template('user_life.html',
                               sequence_ids=sequence_ids,
                               selected_seq_id=default_seq_id,
                               user_profile=default_user,
                               timeline_events=sim_session.get('timeline_events', []),
                               history_messages=sim_session.get('history_messages', []))
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return render_template('user_life.html',
                               sequence_ids=[],
                               selected_seq_id=None,
                               user_profile=None,
                               timeline_events=[],
                               history_messages=[])


# =====================================
# API Endpoints
# =====================================

@app.route('/api/user-profile/<sequence_id>')
def get_user_profile(sequence_id):
    try:
        cfg = load_config(CONFIG_PATH)
        events = load_jsonl_data(cfg["paths"]["events_path"])
        users = load_jsonl_data(cfg["paths"]["users_path"])

        seqid2uid = {e['id']: e['user_id'] for e in events}
        seqid2eseq = {e['id']: e for e in events}
        uid2user = {u['user_id']: u for u in users}

        user_id = seqid2uid.get(sequence_id)
        user = uid2user.get(user_id, {})
        event_seq = seqid2eseq.get(sequence_id, {}).get('events', [])

        return jsonify({'success': True, 'user_profile': user, 'event_sequence': event_seq})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/save-profile', methods=['POST'])
def save_profile():
    try:
        data = request.json
        cfg = load_config(CONFIG_PATH)
        users_path = cfg["paths"]["users_path"]

        users = load_jsonl_data(users_path)
        user_id = data.get('user_id')
        for i, user in enumerate(users):
            if user['user_id'] == user_id:
                users[i] = data
                break
        save_jsonl_data(users_path, users)
        return jsonify({'success': True, 'message': 'Profile saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stream-simulation')
def stream_simulation():
    """Stream assistant evaluation simulation results via Server-Sent Events."""
    sequence_id = request.args.get('sequence_id')
    assistant_model = request.args.get('assistant_model', 'deepseek-chat')
    n_events = int(request.args.get('n_events', 2))
    n_rounds = int(request.args.get('n_rounds', 4))

    def generate():
        result_queue = queue.Queue()
        error_holder = {'error': None}
        done_event = threading.Event()

        def callback(data):
            result_queue.put(data)

        def run_simulation_thread():
            try:
                from run_simulation import build_simulator
                exp_name = f"{sequence_id}_{int(time.time())}"
                sim = build_simulator(callback, exp_name,
                                      config_path=CONFIG_PATH,
                                      assistant_model_name=assistant_model)
                sim_config = {
                    "user_config": {"use_emotion_chain": True, "use_dynamic_memory": False},
                    "assistant_config": {"use_profile_memory": False, "use_key_info_memory": False},
                }
                sim.init_env(sequence_id)
                sim.run_simulation(n_events=n_events, n_rounds=n_rounds, **sim_config)
            except Exception as e:
                error_holder['error'] = str(e)
                logger.error(f"Simulation error: {e}")
            finally:
                done_event.set()

        sim_thread = threading.Thread(target=run_simulation_thread)
        sim_thread.start()

        while not done_event.is_set() or not result_queue.empty():
            try:
                data = result_queue.get(timeout=0.1)
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            except queue.Empty:
                continue

        if error_holder['error']:
            yield f"data: {json.dumps({'step': 'error', 'error': error_holder['error']})}\n\n"
        yield f"data: {json.dumps({'step': 'complete'})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/generate-event', methods=['POST'])
def generate_event():
    """Generate next life event for free chat mode."""
    sim_session = get_or_create_session()
    try:
        data = request.json
        sequence_id = data.get('sequence_id')

        cfg = load_config(CONFIG_PATH)
        events = load_jsonl_data(cfg["paths"]["events_path"])
        users = load_jsonl_data(cfg["paths"]["users_path"])

        seqid2uid = {e['id']: e['user_id'] for e in events}
        uid2user = {u['user_id']: u for u in users}
        profile = uid2user.get(seqid2uid.get(sequence_id), {})

        from models import load_model
        from engine.event_engine import OnlineLifeEventEngine
        from profiles.profile_generator import UserProfile

        user_m_cfg = cfg["models"]["user_model"]
        user_model_name = os.path.basename(user_m_cfg["model_path"])
        retriever_cfg = cfg["retriever"]

        event_model = load_model(
            model_name=user_model_name,
            api_key=user_m_cfg["api_key"],
            model_path=user_m_cfg["model_path"],
            base_url=user_m_cfg["base_url"],
            vllmapi=user_m_cfg["vllmapi"],
        )

        event_retriever_cfg = cfg.get("event_retriever", retriever_cfg)
        event_pool_cfg_path = cfg["paths"].get("event_pool_cfg")
        theme = cfg.get("theme") or '_'.join(sequence_id.replace('NYC_', '').replace('TKY_', '').split('_')[:-1])

        event_retriever = None
        event_database = []
        if event_retriever_cfg and event_pool_cfg_path and os.path.exists(event_pool_cfg_path):
            from tools.dense_retriever import DenseRetriever
            with open(event_pool_cfg_path) as f:
                event_pool_paths = json.load(f)

            event_retriever = DenseRetriever(
                model_name=event_retriever_cfg["embedding_model_path"],
                collection_name=f"trajectory_{theme}_event_collection",
                embedding_dim=event_retriever_cfg["embedding_dim"],
                persist_directory=event_retriever_cfg["persist_directory"],
                distance_function="cosine",
                use_custom_embeddings=False,
                device=event_retriever_cfg.get("device", "cpu"),
            )
            event_pool_path = event_pool_paths.get(theme, list(event_pool_paths.values())[0])
            event_database = load_jsonl_data(event_pool_path)
            if event_retriever.is_collection_empty():
                event_retriever.build_index(event_database, text_field="event", id_field="id", batch_size=256)

        profile_str = str(UserProfile.from_dict(profile))
        event_engine = OnlineLifeEventEngine(
            cfg["paths"]["events_path"], model=event_model, retriever=event_retriever
        )
        event_engine.set_event_sequence(sequence_id)
        event_engine.set_event_index(sim_session['event_counter'])

        event = event_engine.generate_event(
            user_profile=profile_str,
            history_events=sim_session['timeline_events'],
        )

        sim_session['timeline_events'].append(event)
        sim_session['event_counter'] += 1
        sim_session['history_messages'] = []

        return jsonify({'success': True, 'event': event, 'event_index': sim_session['event_counter']})
    except Exception as e:
        logger.error(f"Event generation error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages in free chat mode."""
    sim_session = get_or_create_session()
    try:
        data = request.json
        message = data.get('message')
        sequence_id = data.get('sequence_id')

        if not message:
            return jsonify({'success': False, 'error': 'No message provided'})

        cfg = load_config(CONFIG_PATH)
        events = load_jsonl_data(cfg["paths"]["events_path"])
        users = load_jsonl_data(cfg["paths"]["users_path"])

        seqid2uid = {e['id']: e['user_id'] for e in events}
        uid2user = {u['user_id']: u for u in users}
        profile = uid2user.get(seqid2uid.get(sequence_id), {})

        from models import load_model
        from agents.user_agent import UserAgent
        from profiles.profile_generator import UserProfile

        user_m_cfg = cfg["models"]["user_model"]
        user_model_name = os.path.basename(user_m_cfg["model_path"])

        user_model = load_model(
            model_name=user_model_name,
            api_key=user_m_cfg["api_key"],
            model_path=user_m_cfg["model_path"],
            base_url=user_m_cfg["base_url"],
            vllmapi=user_m_cfg["vllmapi"],
        )

        retriever_cfg = cfg["retriever"]
        user_retriever_cfg = {
            "model_name": retriever_cfg["embedding_model_path"],
            "collection_name": "user_memory_flask_0",
            "max_length": retriever_cfg["max_length"],
            "embedding_dim": retriever_cfg["embedding_dim"],
            "persist_directory": retriever_cfg["persist_directory"],
            "device": retriever_cfg["device"],
            "logger_silent": retriever_cfg.get("logger_silent", False),
        }

        profile_str = str(UserProfile.from_dict(profile))
        user_agent = UserAgent(
            model=user_model,
            retriever_config=user_retriever_cfg,
            profile=profile_str,
            alpha=cfg["simulator"]["alpha"],
        )

        if sim_session['timeline_events']:
            user_agent._build_environment(sim_session['timeline_events'][-1])
            user_agent._build_chat_system_prompt()

        # Restore previous messages into agent state
        if sim_session['history_messages']:
            user_agent.messages = sim_session['history_messages'].copy()

        sim_session['history_messages'].append({'role': 'user', 'content': message})

        sim_config = {"use_emotion_chain": True, "use_dynamic_memory": False}
        result = user_agent.respond(message, **sim_config)
        response = result['response']

        sim_session['history_messages'].append({'role': 'assistant', 'content': response})
        # Sync back agent messages to session
        sim_session['history_messages'] = user_agent.messages.copy()

        return jsonify({'success': True, 'response': response})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    sim_session = get_or_create_session()
    sim_session.update({
        'timeline_events': [],
        'message_blocks': [],
        'history_messages': [],
        'event_counter': 0,
        'simulator_running': False,
        'simulation_results': [],
    })
    return jsonify({'success': True})


def parse_args():
    parser = argparse.ArgumentParser(description='LifeSim - Flask Application')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='Path to the configuration YAML file (default: config.yaml)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on (default: 0.0.0.0)')
    parser.add_argument('--port', '-p', type=int, default=5010,
                        help='Port to run the server on (default: 5010)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Run in debug mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    CONFIG_PATH = args.config
    logger.info(f"Using config file: {CONFIG_PATH}")
    app.run(host=args.host, port=args.port, debug=args.debug)
