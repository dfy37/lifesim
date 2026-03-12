import random
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
    "single_session": {
        "events": "data/single_session/events.jsonl",
        "users": "data/single_session/users.jsonl",
    },
    "long_horizon": {
        "events": "data/long_horizon/events.jsonl",
        "users": "data/long_horizon/users.jsonl",
    },
}


def run_single_simulation(args_tuple):
    idx, total_threads, sequence_id, kwargs = args_tuple

    try:
        if 0 < idx < total_threads:
            time.sleep(60)

        n_events          = kwargs.get('n_events', 10)
        preference_dims   = kwargs.get('preference_dimensions')
        event_path        = kwargs.get('event_path')
        exp_name          = kwargs.get('experiment_name')
        user_pg           = kwargs.get('user_pg')
        user_model        = kwargs.get('user_model')
        assistant_model   = kwargs.get('assistant_model')
        config            = kwargs.get('config')
        chromadb_path     = kwargs.get('chromadb_path', './chromadb')
        logs_path         = kwargs.get('logs_path', './logs')
        retriever_path    = kwargs.get('retriever_model_path')

        logger.info(f"🚀 Start sequence {sequence_id} simulation (Thread {idx})")
        logger.info(f"🔧 Initializing models and agents (Thread {idx})...")

        silent = idx % total_threads != 0

        life_engine = OfflineLifeEventEngine(event_path)

        base_retriever_config = {
            "model_name": retriever_path,
            "max_length": 512,
            "embedding_dim": 1024,
            "persist_directory": chromadb_path,
            "device": "cpu",
            "logger_silent": silent,
        }
        user_retriever_config = {
            **base_retriever_config,
            "collection_name": f"memory_collection_user_{exp_name}_{idx}",
        }
        assistant_retriever_config = {
            **base_retriever_config,
            "collection_name": f"memory_collection_assistant_{exp_name}_{idx}",
        }

        user_agent = UserAgent(
            user_model, user_retriever_config,
            profile=None, alpha=0.5, logger_silent=silent,
        )
        assistant_agent = AssistantAgent(
            assistant_model,
            preference_dimensions=preference_dims,
            user_profile=None,
            logger_silent=silent,
            retriever_config=assistant_retriever_config,
        )

        sim = ConversationSimulator(
            user_profile_generator=user_pg,
            life_event_engine=life_engine,
            user_agent=user_agent,
            assistant_agent=assistant_agent,
            logger_silent=silent,
        )

        logger.info(f"✅ Model and agent initialization completed (Thread {idx})")

        sim.init_env(sequence_id)
        sim.run_simulation(n_events=n_events, n_rounds=2, **config)

        save_dir = os.path.join(logs_path, f'{exp_name}/logs_{idx}')
        sim.save(path=save_dir)
        sim.user_agent.model.save(os.path.join(save_dir, 'user_model.jsonl'))
        sim.assistant_agent.model.save(os.path.join(save_dir, 'assistant_model.jsonl'))

        logger.info(f"✅ Simulation of sequence {sequence_id} completed (Thread {idx})")
        return f"Success: {sequence_id}"

    except Exception as e:
        logger.exception(f"❌ Simulation of sequence {sequence_id} failed (Thread {idx}): {e}")
        return f"Failed: {sequence_id} - {e}"


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
    parser.add_argument('--exp_setting', type=str, default='single_session',
                        choices=['single_session', 'long_horizon'], help='Experiment setting')
    return parser.parse_args()


def main():
    set_seed(42)

    args = get_args()
    logger.info("Args: " + str(args))

    user_model_name      = os.path.basename(args.user_model_path)
    assistant_model_name = os.path.basename(args.assistant_model_path)

    exp_name = f"main_user_{user_model_name}_assistant_{assistant_model_name}"
    if args.use_preference_memory:
        exp_name += '-w_preference_memory'

    config = {
        "user_config": {
            "use_emotion_chain": True,
            "use_dynamic_memory": True,
        },
        "assistant_config": {
            "use_profile_memory": args.use_preference_memory,
            "use_key_info_memory": False,
        },
    }

    with open('data/language_templates.json') as f:
        preference_dimensions = json.load(f)

    chromadb_path = os.path.join(args.chromadb_root, exp_name)
    if os.path.exists(chromadb_path):
        shutil.rmtree(chromadb_path)

    logger.info("🏃‍♀️ Individual Full-Life-Cycle Simulator – Command-Line Version")
    logger.info("=" * 80)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    events = load_jsonl_data(EVENTS_USERS_PATH[args.exp_setting]['events'])
    if args.seq_ids:
        exp_name += '_V2'
        sequence_ids = args.seq_ids.split(',')
    else:
        sequence_ids = [e['id'] for e in events]

    user_pg = UserProfileGenerator(EVENTS_USERS_PATH[args.exp_setting]['users'], random_state=1)

    user_model = load_model(
        model_name=user_model_name,
        api_key=args.user_model_api_key,
        model_path=args.user_model_path,
        base_url=args.user_model_url,
        vllmapi=True,
    )
    assistant_model = load_model(
        model_name=assistant_model_name,
        api_key=args.assistant_model_api_key,
        model_path=args.assistant_model_path,
        base_url=args.assistant_model_url,
        vllmapi=True,
    )

    logger.info("🔄 Running dialogue simulation...")

    kwargs = {
        'n_events':              args.n_events_per_sequence,
        'preference_dimensions': preference_dimensions,
        'event_path':            EVENTS_USERS_PATH[args.exp_setting]['events'],
        'experiment_name':       exp_name + '_' + args.exp_setting,
        'user_model':            user_model,
        'assistant_model':       assistant_model,
        'user_pg':               user_pg,
        'config':                config,
        'chromadb_path':         chromadb_path,
        'logs_path':             args.logs_root,
        'retriever_model_path':  args.retriever_model_path,
    }
    args_list = [(i, args.n_threads, seq_id, kwargs) for i, seq_id in enumerate(sequence_ids)]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        future_to_idx = {executor.submit(run_single_simulation, a): a[0] for a in args_list}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                           total=len(sequence_ids), desc="Simulation progress"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                logger.debug(f"Thread {idx} completed: {result}")
            except Exception as exc:
                error_msg = f"Exception in process {idx}: {exc}"
                logger.exception(error_msg)
                results.append(error_msg)

    success_count = sum(1 for r in results if r.startswith("Success"))
    failed_count  = len(results) - success_count

    logger.info("🎯 Simulation results statistics:")
    logger.info(f"  ✅ Success: {success_count}")
    logger.info(f"  ❌ Failure: {failed_count}")
    if failed_count > 0:
        logger.info("Failure detail:")
        for result in results:
            if not result.startswith("Success"):
                logger.info(f"  - {result}")

    logger.info("✅ Simulation completed!")


if __name__ == '__main__':
    main()
