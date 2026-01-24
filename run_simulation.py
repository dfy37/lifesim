from agents.user_agent import UserAgent
from agents.assistant_agent import AssistantAgent
from agents.analysis_agent import AnalysisAgent
from engine.event_engine import OfflineLifeEventEngine
from profiles.profile_generator import UserProfileGenerator
from simulation.conversation_simulator import ConversationSimulator
from models import load_model
import json
import os
import shutil
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_retriever_config(name, idx, retriever_cfg):
    return {
        "model_name": retriever_cfg["embedding_model_path"],
        "collection_name": f"{name}_{idx}",
        "max_length": retriever_cfg["max_length"],
        "embedding_dim": retriever_cfg["embedding_dim"],
        "persist_directory": retriever_cfg["persist_directory"],
        "device": retriever_cfg["device"],
        "logger_silent": retriever_cfg.get("logger_silent", False)
    }

def build_simulator(on_turn_update, exp_name, config_path="config.yaml"):
    cfg = load_config(config_path)

    # MODEL PATHS
    user_m_cfg = cfg["models"]["user_model"]
    asst_m_cfg = cfg["models"]["assistant_model"]
    analysis_m_cfg = cfg["models"]["analysis_model"]

    user_model_name = os.path.basename(user_m_cfg["model_path"])
    assistant_model_name = os.path.basename(asst_m_cfg["model_path"])
    analysis_model_name = os.path.basename(analysis_m_cfg["model_path"])

    user_model = load_model(
        model_name=user_model_name,
        api_key=user_m_cfg["api_key"],
        model_path=user_m_cfg["model_path"],
        base_url=user_m_cfg["base_url"],
        vllmapi=user_m_cfg["vllmapi"],
        reason=False,
    )

    assistant_model = load_model(
        model_name=assistant_model_name,
        api_key=asst_m_cfg["api_key"],
        model_path=asst_m_cfg["model_path"],
        base_url=asst_m_cfg["base_url"],
        vllmapi=asst_m_cfg["vllmapi"],
        reason=False,
    )

    analysis_model = load_model(
        model_name=analysis_model_name,
        api_key=asst_m_cfg["api_key"],
        model_path=asst_m_cfg["model_path"],
        base_url=asst_m_cfg["base_url"],
        vllmapi=asst_m_cfg["vllmapi"],
        reason=False,
    )

    # Input data paths
    events_path = cfg["paths"]["events_path"]
    users_path = cfg["paths"]["users_path"]
    preference_file = cfg["paths"]["preference_file"]

    life_engine = OfflineLifeEventEngine(events_path)
    user_pg = UserProfileGenerator(users_path)

    retriever_cfg = cfg["retriever"]

    # Build retriever config for both agents
    user_retriever_cfg = build_retriever_config(
        f"user_memory_{exp_name}", 0, retriever_cfg
    )
    asst_retriever_cfg = build_retriever_config(
        f"assistant_memory_{exp_name}", 0, retriever_cfg
    )

    preference_dims = json.load(open(preference_file))

    # Initialize agents
    user_agent = UserAgent(
        model=user_model,
        retriever_config=user_retriever_cfg,
        profile=None,
        alpha=cfg["simulator"]["alpha"]
    )
    assistant_agent = AssistantAgent(
        model=assistant_model,
        preference_dimensions=preference_dims,
        user_profile=None,
        retriever_config=asst_retriever_cfg
    )
    analysis_agent = AnalysisAgent(
        model=analysis_model
    )

    simulator = ConversationSimulator(
        user_profile_generator=user_pg,
        life_event_engine=life_engine,
        user_agent=user_agent,
        assistant_agent=assistant_agent,
        on_turn_update=on_turn_update,
        analysis_agent=analysis_agent
    )

    return simulator
