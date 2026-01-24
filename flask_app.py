"""
LifeSim Demo - Flask Application
A user life cycle simulator with beautiful UI
"""
import os
import json
import yaml
import time
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, Response
from flask_cors import CORS
import queue
import threading

from utils.utils import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Global storage for simulation states (in production, use Redis or database)
simulation_sessions = {}

def load_config(config_path="config.yaml"):
    """Load configuration file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_jsonl_data(path):
    """Load JSONL format data"""
    data = []
    if os.path.exists(path):
        with open(path) as reader:
            for row in reader:
                data.append(json.loads(row))
    return data

def save_jsonl_data(path, data):
    """Save data to JSONL file"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_user_by_id(user_id, users_path):
    """Load user data by ID"""
    users = load_jsonl_data(users_path)
    for u in users:
        if u["user_id"] == user_id:
            return u
    return None

def get_or_create_session():
    """Get or create a simulation session"""
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
            'simulation_results': []
        }
    return simulation_sessions[session_id]

# =====================================
# Routes
# =====================================

@app.route('/')
def index():
    """Home page with mode selection"""
    return render_template('index.html')

@app.route('/assistant-eval')
def assistant_eval():
    """Assistant evaluation mode page"""
    sim_session = get_or_create_session()

    # Try to load config and data
    try:
        cfg = load_config('/remote-home/fyduan/secrets/config.yaml')
        events = load_jsonl_data(cfg["paths"]["events_path"])
        users = load_jsonl_data(cfg["paths"]["users_path"])

        sequence_ids = [e['id'] for e in events]
        seqid2uid = {e['id']: e['user_id'] for e in events}
        seqid2eseq = {e['id']: e for e in events}
        uid2user = {u['user_id']: u for u in users}

        # Default selection
        default_seq_id = sequence_ids[0] if sequence_ids else None
        default_user = uid2user.get(seqid2uid.get(default_seq_id)) if default_seq_id else None
        default_events = seqid2eseq.get(default_seq_id, {}).get('events', []) if default_seq_id else []

        return render_template('assistant_eval.html',
                             sequence_ids=sequence_ids,
                             selected_seq_id=default_seq_id,
                             user_profile=default_user,
                             event_sequence=default_events,
                             assistant_models=['deepseek-chat', 'gpt-5-mini', 'gpt-4o'],
                             simulation_results=sim_session.get('simulation_results', []))
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Return page with empty/demo data
        return render_template('assistant_eval.html',
                             sequence_ids=['demo_user_1', 'demo_user_2'],
                             selected_seq_id='demo_user_1',
                             user_profile={
                                 'user_id': 'demo_user_1',
                                 'gender': 'Male',
                                 'age': 30,
                                 'marital': 'Single',
                                 'area': 'New York',
                                 'income': 'Middle',
                                 'employment': 'Software Engineer',
                                 'personality': ['Introverted', 'Analytical'],
                                 'preferences': ['Technology', 'Reading']
                             },
                             event_sequence=[],
                             assistant_models=['deepseek-chat', 'gpt-5-mini', 'gpt-4o'],
                             simulation_results=[])

@app.route('/user-life')
def user_life():
    """Free chat mode page"""
    sim_session = get_or_create_session()

    try:
        cfg = load_config('/remote-home/fyduan/secrets/config.yaml')
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
                             sequence_ids=['demo_user_1', 'demo_user_2'],
                             selected_seq_id='demo_user_1',
                             user_profile={
                                 'user_id': 'demo_user_1',
                                 'gender': 'Male',
                                 'age': 30,
                                 'marital': 'Single',
                                 'area': 'New York',
                                 'income': 'Middle',
                                 'employment': 'Software Engineer',
                                 'personality': ['Introverted', 'Analytical'],
                                 'preferences': ['Technology', 'Reading']
                             },
                             timeline_events=[],
                             history_messages=[])

# =====================================
# API Endpoints
# =====================================

@app.route('/api/user-profile/<sequence_id>')
def get_user_profile(sequence_id):
    """Get user profile by sequence ID"""
    try:
        cfg = load_config('/remote-home/fyduan/secrets/config.yaml')
        events = load_jsonl_data(cfg["paths"]["events_path"])
        users = load_jsonl_data(cfg["paths"]["users_path"])

        seqid2uid = {e['id']: e['user_id'] for e in events}
        seqid2eseq = {e['id']: e for e in events}
        uid2user = {u['user_id']: u for u in users}

        user_id = seqid2uid.get(sequence_id)
        user = uid2user.get(user_id, {})
        event_seq = seqid2eseq.get(sequence_id, {}).get('events', [])

        return jsonify({
            'success': True,
            'user_profile': user,
            'event_sequence': event_seq
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save-profile', methods=['POST'])
def save_profile():
    """Save user profile"""
    try:
        data = request.json
        cfg = load_config('/remote-home/fyduan/secrets/config.yaml')
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

@app.route('/api/start-simulation', methods=['POST'])
def start_simulation():
    """Start assistant evaluation simulation (non-streaming, kept for compatibility)"""
    sim_session = get_or_create_session()

    try:
        data = request.json
        sequence_id = data.get('sequence_id')
        assistant_model = data.get('assistant_model', 'deepseek-chat')
        n_events = data.get('n_events', 2)
        n_rounds = data.get('n_rounds', 4)

        # Clear previous results
        sim_session['simulation_results'] = []
        sim_session['simulator_running'] = True

        # Import and build simulator
        from run_simulation import build_simulator

        results = []

        def callback(data):
            results.append(data)

        exp_name = f"{sequence_id}_{int(time.time())}"
        sim = build_simulator(callback, exp_name,
                            config_path='/remote-home/fyduan/secrets/config.yaml',
                            assistant_model_name=assistant_model)

        sim_config = {
            "user_config": {
                "use_emotion_chain": True,
                "use_dynamic_memory": False,
            },
            "assistant_config": {
                "use_profile_memory": False,
                "use_key_info_memory": False,
            },
        }

        sim.init_env(sequence_id)
        sim.run_simulation(n_events=n_events, n_rounds=n_rounds, **sim_config)

        sim_session['simulation_results'] = results
        sim_session['simulator_running'] = False

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        sim_session['simulator_running'] = False
        logger.error(f"Simulation error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stream-simulation')
def stream_simulation():
    """Stream assistant evaluation simulation results using Server-Sent Events"""
    sequence_id = request.args.get('sequence_id')
    assistant_model = request.args.get('assistant_model', 'deepseek-chat')
    n_events = int(request.args.get('n_events', 2))
    n_rounds = int(request.args.get('n_rounds', 4))

    def generate():
        # Use a queue to communicate between the callback and the generator
        result_queue = queue.Queue()
        error_holder = {'error': None}
        done_event = threading.Event()

        def callback(data):
            """Callback function that puts data into the queue"""
            result_queue.put(data)

        def run_simulation_thread():
            """Run simulation in a separate thread"""
            try:
                from run_simulation import build_simulator

                exp_name = f"{sequence_id}_{int(time.time())}"
                sim = build_simulator(callback, exp_name,
                                    config_path='/remote-home/fyduan/secrets/config.yaml',
                                    assistant_model_name=assistant_model)

                sim_config = {
                    "user_config": {
                        "use_emotion_chain": True,
                        "use_dynamic_memory": False,
                    },
                    "assistant_config": {
                        "use_profile_memory": False,
                        "use_key_info_memory": False,
                    },
                }

                sim.init_env(sequence_id)
                sim.run_simulation(n_events=n_events, n_rounds=n_rounds, **sim_config)

            except Exception as e:
                error_holder['error'] = str(e)
                logger.error(f"Simulation error: {e}")
            finally:
                done_event.set()

        # Start simulation in background thread
        sim_thread = threading.Thread(target=run_simulation_thread)
        sim_thread.start()

        # Yield results as they come in
        while not done_event.is_set() or not result_queue.empty():
            try:
                # Wait for data with timeout to check done_event periodically
                data = result_queue.get(timeout=0.1)
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            except queue.Empty:
                continue

        # Check for errors
        if error_holder['error']:
            yield f"data: {json.dumps({'step': 'error', 'error': error_holder['error']})}\n\n"

        # Send completion signal
        yield f"data: {json.dumps({'step': 'complete'})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/generate-event', methods=['POST'])
def generate_event():
    """Generate next life event for free chat mode"""
    sim_session = get_or_create_session()

    try:
        data = request.json
        sequence_id = data.get('sequence_id')

        cfg = load_config('/remote-home/fyduan/secrets/config.yaml')
        events = load_jsonl_data(cfg["paths"]["events_path"])
        users = load_jsonl_data(cfg["paths"]["users_path"])

        seqid2uid = {e['id']: e['user_id'] for e in events}
        uid2user = {u['user_id']: u for u in users}

        profile = uid2user.get(seqid2uid.get(sequence_id), {})

        # Import necessary modules
        from models import load_model
        from engine.event_engine import OnlineLifeEventEngine
        from tools.dense_retriever import DenseRetriever
        from profiles.profile_generator import UserProfile

        user_m_cfg = cfg["models"]["user_model"]
        user_model_name = os.path.basename(user_m_cfg["model_path"])

        event_model = load_model(
            model_name=user_model_name,
            api_key=user_m_cfg["api_key"],
            model_path=user_m_cfg["model_path"],
            base_url=user_m_cfg["base_url"],
            vllmapi=user_m_cfg["vllmapi"],
            reason=False,
        )

        EVENT_POOL_PATH = {
            'elderlycare': '/remote-home/fyduan/event_pool_en/elderlycare_events.jsonl',
            'sport_health': '/remote-home/fyduan/event_pool_en/sport_health_events.jsonl',
            'travel': '/remote-home/fyduan/event_pool_en/travel_events.jsonl',
            'mental_health': '/remote-home/fyduan/event_pool_en/mental_health_events.jsonl',
            'education': '/remote-home/fyduan/event_pool_en/education_events.jsonl',
            'childcare': '/remote-home/fyduan/event_pool_en/childcare_events.jsonl',
            'dining': '/remote-home/fyduan/event_pool_en/dining_events.jsonl',
            'entertainment': '/remote-home/fyduan/event_pool_en/entertainment_events.jsonl'
        }

        theme = '_'.join(sequence_id.split('NYC_')[-1].split('TKY_')[-1].split('_')[:-1])

        event_retriever = DenseRetriever(
            model_name="/remote-home/fyduan/MODELS/Qwen3-Embedding-0.6B",
            collection_name=f"trajectory_{theme}_event_collection",
            embedding_dim=1024,
            persist_directory="/remote-home/fyduan/exp_data/chroma_db",
            distance_function="cosine",
            use_custom_embeddings=False,
            device='cuda:6'
        )

        event_database = load_jsonl_data(EVENT_POOL_PATH.get(theme, EVENT_POOL_PATH['entertainment']))
        if event_retriever.is_collection_empty():
            event_retriever.build_index(event_database, text_field="event", id_field="id", batch_size=256)

        profile_str = str(UserProfile.from_dict(profile))
        event_engine = OnlineLifeEventEngine(cfg["paths"]["events_path"], model=event_model, retriever=event_retriever)
        event_engine.set_event_sequence(sequence_id)
        event_engine.set_event_index(sim_session['event_counter'])

        event = event_engine.generate_event(
            user_profile=profile_str,
            history_events=sim_session['timeline_events']
        )

        sim_session['timeline_events'].append(event)
        sim_session['event_counter'] += 1
        sim_session['history_messages'] = []

        return jsonify({
            'success': True,
            'event': event,
            'event_index': sim_session['event_counter']
        })
    except Exception as e:
        logger.error(f"Event generation error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages in free chat mode"""
    sim_session = get_or_create_session()

    try:
        data = request.json
        message = data.get('message')
        sequence_id = data.get('sequence_id')

        if not message:
            return jsonify({'success': False, 'error': 'No message provided'})

        cfg = load_config('/remote-home/fyduan/secrets/config.yaml')
        events = load_jsonl_data(cfg["paths"]["events_path"])
        users = load_jsonl_data(cfg["paths"]["users_path"])

        seqid2uid = {e['id']: e['user_id'] for e in events}
        uid2user = {u['user_id']: u for u in users}

        profile = uid2user.get(seqid2uid.get(sequence_id), {})

        # Import necessary modules
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
            reason=False,
        )

        retriever_cfg = cfg["retriever"]
        user_retriever_cfg = {
            "model_name": retriever_cfg["embedding_model_path"],
            "collection_name": f"user_memory_flask_0",
            "max_length": retriever_cfg["max_length"],
            "embedding_dim": retriever_cfg["embedding_dim"],
            "persist_directory": retriever_cfg["persist_directory"],
            "device": retriever_cfg["device"],
            "logger_silent": retriever_cfg.get("logger_silent", False)
        }

        profile_str = str(UserProfile.from_dict(profile))
        user_agent = UserAgent(
            model=user_model,
            retriever_config=user_retriever_cfg,
            profile=profile_str,
            alpha=cfg["simulator"]["alpha"]
        )

        # Build environment if we have events
        if sim_session['timeline_events']:
            user_agent._build_environment(sim_session['timeline_events'][-1])
            user_agent._build_chat_system_prompt()

        # Set previous messages
        user_agent.set_messages(sim_session['history_messages'].copy())

        # Add user message
        sim_session['history_messages'].append({'role': 'user', 'content': message})

        # Get response
        sim_config = {
            "use_emotion_chain": True,
            "use_dynamic_memory": False,
        }

        result = user_agent.respond(message, **sim_config)
        response = result['response']

        sim_session['history_messages'].append({'role': 'assistant', 'content': response})

        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    """Clear current session data"""
    sim_session = get_or_create_session()
    sim_session['timeline_events'] = []
    sim_session['message_blocks'] = []
    sim_session['history_messages'] = []
    sim_session['event_counter'] = 0
    sim_session['simulator_running'] = False
    sim_session['simulation_results'] = []

    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
