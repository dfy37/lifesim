import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from simulation.conv_history_generator import ConvHistoryGenerator
from simulation.fast_conv_simulator import FastConvSimulator
from profiles.profile_generator import UserProfileGenerator
from agents.user_agent import UserAgent
from engine.event_engine import OfflineLifeEventEngine
from models import load_model
from tools.dense_retriever import DenseRetriever
from utils.utils import get_logger, load_jsonl_data, write_jsonl_data

logger = get_logger(__name__, silent=False)
@dataclass
class SampleProfile:
    user_id: str
    profile: Dict[str, Any]
    profile_str: str


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conversation histories for profiles.")
    parser.add_argument("--profiles-path", required=True, help="Path to user profiles JSONL.")
    parser.add_argument("--query-database-path", required=True, help="Path to retriever query database JSONL.")
    parser.add_argument("--output-path", required=True, help="Output JSONL path for generated histories.")

    parser.add_argument("--model-name", required=True, help="Model name used for generation.")
    parser.add_argument("--model-api-key", default=None, help="API key for model if required.")
    parser.add_argument("--model-path", default=None, help="Local model path or vLLM model name.")
    parser.add_argument("--model-url", default=None, help="Base URL for API model endpoints.")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM API-compatible interface.")

    parser.add_argument("--max-conv-turns", type=int, default=10)
    parser.add_argument("--max-events-number", type=int, default=10)
    parser.add_argument("--max-profiles", type=int, default=-1, help="Limit number of profiles. -1 means all.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for whole experiments.")

    parser.add_argument("--retriever-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--retriever-collection-name", default="trajectory_event_collection")
    parser.add_argument("--retriever-embedding-dim", type=int, default=384)
    parser.add_argument("--retriever-persist-dir", default="./chroma_db")
    parser.add_argument("--retriever-distance", default="cosine")
    parser.add_argument("--retriever-device", default="cpu")
    parser.add_argument("--retriever-use-custom-embeddings", action="store_true")
    parser.add_argument("--retriever-text-field", default="event")
    parser.add_argument("--retriever-id-field", default="id")
    parser.add_argument("--retriever-batch-size", type=int, default=128)
    parser.add_argument("--max-retrievals", type=int, default=5)

    return parser.parse_args()


def build_profiles(generator: UserProfileGenerator, max_profiles: int) -> List[SampleProfile]:
    raw_profiles = generator.get_profile_str(n=max_profiles if max_profiles != -1 else -1)
    profiles = [
        SampleProfile(
            user_id=item["uuid"],
            profile=item.get("profile", None),
            profile_str=item.get("profile_str", ""),
        )
        for item in raw_profiles
    ]
    return profiles


def ensure_retriever_index(retriever: DenseRetriever, query_database_path: str, text_field: str, id_field: str, batch_size: int) -> None:
    if not retriever.is_collection_empty():
        return
    query_database = load_jsonl_data(query_database_path)
    if not query_database:
        logger.warning("Query database is empty, skipping index build.")
        return
    logger.info("Collection is empty, building index...")
    retriever.build_index(query_database, text_field=text_field, id_field=id_field, batch_size=batch_size)


def generate_for_profile(
    profile: SampleProfile,
    event_engine: OfflineLifeEventEngine,
    conv_generator: ConvHistoryGenerator,
    max_events_number: int,
    max_conv_turns: int,
    seed: Optional[int],
) -> Optional[Dict[str, Any]]:
    if not event_engine.has_next_event():
        logger.warning("No events found for user_id=%s, skipping.", profile.user_id)
        return None

    conv_history = conv_generator.generate(
        max_events_number=min(max_events_number, event_engine.remaining_events()),
        max_conv_turns=max_conv_turns,
        seed=seed,
    )

    return {
        "user_id": profile.user_id,
        "profile": profile.profile,
        "profile_str": profile.profile_str,
        "event_count": len(conv_history),
        "conv_history": conv_history,
    }


def main(args: argparse.Namespace) -> None:
    profile_generator = UserProfileGenerator(args.profiles_path, random_state=args.seed)
    profiles = build_profiles(profile_generator, args.max_profiles)

    model = load_model(
        model_name=args.model_name,
        api_key=args.model_api_key,
        model_path=args.model_path,
        base_url=args.model_url,
        vllmapi=args.use_vllm,
    )
    fast_conv_simulator = FastConvSimulator(model=model, max_turns=args.max_conv_turns)

    retriever = DenseRetriever(
        model_name=args.retriever_model_name,
        collection_name=args.retriever_collection_name,
        embedding_dim=args.retriever_embedding_dim,
        persist_directory=args.retriever_persist_dir,
        distance_function=args.retriever_distance,
        use_custom_embeddings=args.retriever_use_custom_embeddings,
        device=args.retriever_device,
    )
    ensure_retriever_index(
        retriever,
        query_database_path=args.query_database_path,
        text_field=args.retriever_text_field,
        id_field=args.retriever_id_field,
        batch_size=args.retriever_batch_size,
    )

    results: List[Dict[str, Any]] = []
    for profile in profiles:
        event_engine = OfflineLifeEventEngine(profile.profile.life_events)
        user_agent = UserAgent(model=model, profile=profile.profile_str)
        conv_generator = ConvHistoryGenerator(
            life_event_engine=event_engine,
            user_agent=user_agent,
            fast_conv_simulator=fast_conv_simulator,
            model=model,
            retriever=retriever,
            max_retrievals=args.max_retrievals,
        )

        result = generate_for_profile(
            profile,
            event_engine,
            conv_generator,
            max_events_number=args.max_events_number,
            max_conv_turns=args.max_conv_turns,
            seed=args.seed,
        )
        if result:
            results.append(result)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    write_jsonl_data(results, args.output_path)
    logger.info("Saved %s conversation histories to %s", len(results), args.output_path)


if __name__ == "__main__":
    main(get_args())
