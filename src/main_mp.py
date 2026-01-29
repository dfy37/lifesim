import random
import threading
import concurrent.futures
import multiprocessing as mp
from tqdm.auto import tqdm
import json
import os
import shutil
import argparse
import time

from agents.user_agent import UserAgent
from agents.assistant_agent import AssistantAgent
from engine.event_engine import OfflineLifeEventEngine
from simulation.conversation_simulator import ConversationSimulator
from models import load_model
from profiles.profile_generator import UserProfileGenerator
from utils.utils import get_logger, load_jsonl_data

import chromadb
import numpy as np
import torch

chromadb.api.client.SharedSystemClient.clear_system_cache()

logger = get_logger(__name__)

EVENTS_USERS_PATH = {
    "total": {
        "events": "/inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/event_sequences/total/events_with_intent_type.jsonl",
        "users": "/inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/event_sequences/total/users.jsonl",
    },
    "long_session": {
        "events": "/inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/ai_assistant/evaluation/long_session/total_data/events.jsonl",
        "users": "/inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/ai_assistant/evaluation/long_session/total_data/users.jsonl"
    }
}

def run_single_simulation(args_tuple):
    try:
        idx, total_threads, sequence_id, kwargs = args_tuple
        
        if idx < total_threads and idx != 0:
            time.sleep(60)

        n_events = kwargs.get('n_events', 10)
        preference_dimensions = kwargs.get('preference_dimensions', None)
        event_path = kwargs.get('event_path', None)
        exp_name = kwargs.get('experiment_name', None)
        user_pg = kwargs.get('user_pg', None)
        user_model = kwargs.get('user_model', None)
        assistant_model = kwargs.get('assistant_model', None)
        config = kwargs.get('config', None)
        chromadb_path = kwargs.get('chromadb_path', './chromadb')
        logs_path = kwargs.get('logs_path', './logs')
        retriever_model_path = kwargs.get('retriever_model_path', None)
        
        logger.info(f"🚀 Start sequence {sequence_id} simulation (Thread {idx})")
        
        logger.info(f"🔧 Initializing models and agents (Thread {idx})...")
        
        cancel_logger = True 
        if idx % total_threads == 0:
            cancel_logger = False

        life_engine = OfflineLifeEventEngine(event_path)

        user_retriever_config = {
            "model_name": retriever_model_path,
            "collection_name": f"memory_collection_user_{exp_name}_{idx}",
            "max_length": 512,
            "embedding_dim": 1024,
            "persist_directory": chromadb_path,
            "device": "cpu",
            "logger_silent": cancel_logger
        }

        assistant_retriever_config = {
            "model_name": retriever_model_path,
            "collection_name": f"memory_collection_assistant_{exp_name}_{idx}",
            "max_length": 512,
            "embedding_dim": 1024,
            "persist_directory": chromadb_path,
            "device": "cpu",
            "logger_silent": cancel_logger
        }
        
        user_agent = UserAgent(user_model, user_retriever_config, profile=None, alpha=0.5, logger_silent=cancel_logger)
        assistant_agent = AssistantAgent(assistant_model, preference_dimensions=preference_dimensions, user_profile=None, logger_silent=cancel_logger, retriever_config=assistant_retriever_config)
        
        sim = ConversationSimulator(
            user_profile_generator=user_pg,
            life_event_engine=life_engine, 
            user_agent=user_agent, 
            assistant_agent=assistant_agent,
            logger_silent=cancel_logger
        )
        
        logger.info(f"✅ Model and agent initialization completed (Thread {idx})")
        
        sim.init_env(sequence_id)
        sim.run_simulation(
            n_events=n_events,
            n_rounds=3,
            **config
        )
        sim.save(path=os.path.join(logs_path, f'{exp_name}/logs_{idx}'))
        sim.user_agent.model.save(os.path.join(logs_path, f'{exp_name}/logs_{idx}/user_model.jsonl'))
        sim.assistant_agent.model.save(os.path.join(logs_path, f'{exp_name}/logs_{idx}/assistant_model.jsonl'))
        
        logger.info(f"✅ Simulation of sequence {sequence_id} completed (Thread {idx})")
        return f"Success: {sequence_id}"
        
    except Exception as e:
        idx, total_threads, sequence_id, kwargs = args_tuple
        idx = args_tuple[0] if isinstance(args_tuple, tuple) and len(args_tuple) > 0 else 'unknown'
        logger.exception(f"❌ Simulation of sequence {sequence_id} failed ! (Thread {idx}): {str(e)}")
        return f"Failed: {sequence_id} - {str(e)}"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description="User-Assistant Interaction")
    parser.add_argument('--user_model_path', type=str, help='Path to user model')
    parser.add_argument('--user_model_url', type=str, help='Vllm url for user model')
    parser.add_argument('--user_model_api_key', type=str, default='123', help='API key for user model')
    parser.add_argument('--assistant_model_path', type=str, help='Path to assistant model')
    parser.add_argument('--assistant_model_url', type=str, help='Vllm url for assistant model')
    parser.add_argument('--assistant_model_api_key', type=str, default='123', help='API key for assistant model')
    parser.add_argument('--use_preference_memory', action="store_true", help='Store user preference prediction for assistant')    
    parser.add_argument('--chromadb_root', type=str, default='./chromadb', help='Retriever store path')
    parser.add_argument('--logs_root', type=str, default='./logs', help='Logs store path')
    parser.add_argument('--seq_ids', type=str, default=None, help='Event sequences ids to tackle')
    parser.add_argument('--retriever_model_path', type=str, help='Retriever model path')
    parser.add_argument('--n_events_per_sequence', type=int, default=10, help='Number of events per sequence')
    parser.add_argument('--n_threads', type=int, default=4, help='Number of threads for simulation')
    
    args = parser.parse_args()
    return args

def main():
    set_seed(42)
    
    args = get_args()
    logger.info("Args: " + str(args))
    
    USER_MODEL_PATH=args.user_model_path
    USER_MODEL_URL=args.user_model_url
    ASSISTANT_MODEL_PATH=args.assistant_model_path
    ASSISTANT_MODEL_URL=args.assistant_model_url
    USE_PREFERENCE_MEMORY=args.use_preference_memory
    
    USER_MODEL=os.path.basename(USER_MODEL_PATH)
    ASSISTANT_MODEL=os.path.basename(ASSISTANT_MODEL_PATH)

    NAME = f"main_user_{USER_MODEL}_assistant_{ASSISTANT_MODEL}"
    if USE_PREFERENCE_MEMORY:
        NAME += '-w_preference_memory'
    THEME = "long_session"
    N_EVENTS_PER_SEQUENCE = args.n_events_per_sequence
    N_THREADS = args.n_threads
    CONFIG = {
        "user_config": {
            "use_emotion_chain": True,
            "use_dynamic_memory": True
        },
        "assistant_config": {
            "use_profile_memory": True if USE_PREFERENCE_MEMORY else False,
            "use_key_info_memory": False
        }
    }
    PREFERENCE_DIMENSIONS = json.load(open('/inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/data/language_templates.json'))
    CHROMADB_PATH=os.path.join(args.chromadb_root, NAME)
    if os.path.exists(CHROMADB_PATH):
        shutil.rmtree(CHROMADB_PATH)
    LOGS_PATH=args.logs_root
    
    logger.info("🏃‍♀️ Individual Full-Life-Cycle Simulator – Command-Line Version")
    logger.info("=" * 80)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    events = load_jsonl_data(EVENTS_USERS_PATH[THEME]['events'])
    if args.seq_ids:
        NAME += '_V2'
        sequence_ids = args.seq_ids.split(',')
    else:
        sequence_ids = [e['id'] for e in events]
        # sequence_ids = [e['id'] for e in events][:80] # TODO GPT-4o的进度

    user_pg = UserProfileGenerator(
        EVENTS_USERS_PATH[THEME]['users'],
        random_state=1
    )
    
    user_model = load_model(model_name=USER_MODEL, api_key=args.user_model_api_key, model_path=USER_MODEL_PATH, base_url=USER_MODEL_URL, vllmapi=True)
    assistant_model = load_model(model_name=ASSISTANT_MODEL, api_key=args.assistant_model_api_key, model_path=ASSISTANT_MODEL_PATH, base_url=ASSISTANT_MODEL_URL, vllmapi=True)

    logger.info("🔄 Running dialogue simulation…")
    
    # 准备参数列表 - 修正参数结构
    kwargs = {
        'n_events': N_EVENTS_PER_SEQUENCE, 
        'preference_dimensions': PREFERENCE_DIMENSIONS,
        'event_path': EVENTS_USERS_PATH[THEME]['events'], 
        'experiment_name':  NAME + '_' + THEME, 
        'user_path': EVENTS_USERS_PATH[THEME]['users'],
        'user_model': user_model,
        'assistant_model': assistant_model,
        'user_pg': user_pg,
        'config': CONFIG,
        'chromadb_path': CHROMADB_PATH,
        'logs_path': LOGS_PATH,
        'retriever_model_path': args.retriever_model_path
    }

    args_list = [(i, N_THREADS, sequence_id, kwargs) for i, sequence_id in enumerate(sequence_ids)]
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        future_to_idx = {
            executor.submit(run_single_simulation, args): args[0] 
            for args in args_list
        }
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx), 
            total=len(sequence_ids), 
            desc="Simulation progress"
        ):
            
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                logger.debug(f"Thread {idx} completed: {result}")
            except Exception as exc:
                error_msg = f"Exception in process {idx}: {str(exc)}"
                logger.exception(error_msg)
                results.append(error_msg)
    
    logger.info("🎯 Simulation results statistics:")
    success_count = sum(1 for r in results if r.startswith("Success"))
    failed_count = len(results) - success_count
    
    logger.info(f"  ✅ Success: {success_count}")
    logger.info(f"  ❌ Failure: {failed_count}")
    
    if failed_count > 0:
        logger.info("Failure detail:")
        for result in results:
            if not result.startswith("Success"):
                logger.info(f"  - {result}")
    
    logger.info("✅ Simulation completed！")

if __name__ == '__main__':
    main()