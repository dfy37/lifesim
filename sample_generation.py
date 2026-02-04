from src.simulation.conv_history_generator import ConvHistoryGenerator
from src.simulation.fast_conv_simulator import FastConvSimulator
from src.profiles.profile_generator import UserProfileGenerator
from src.agents.user_agent import UserAgent
from src.engine.event_engine import OfflineLifeEventEngine
from src.models import load_model
from src.tools.dense_retriever import DenseRetriever
from src.utils.utils import get_logger, load_jsonl_data

logger = get_logger(__name__, silent=False)

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-conv-turns", required=False, type=int, default=10)
    parser.add_argument("--max-events-number", required=False, type=int, default=10)
    
    args = parser.parse_args()
    
    return args

def main(args):
    profile_generator = UserProfileGenerator(...)
    profiles = profile_generator.generate_profiles()
    
    fast_conv_simulator = FastConvSimulator(max_turns=args.max_conv_turns)
    
    model = load_model(model_name=args.model_name, api_key=args.model_api_key, model_path=args.model_path, base_url=args.model_url, vllmapi=args.use_vllm)
    retriever = DenseRetriever(
        model_name="/inspire/hdd/project/socialsimulation/linjiayu-CZXS25120090/FYDUAN/MODELS/Qwen3-Embedding-0.6B",
        collection_name=f"trajectory_{args.theme}_event_collection",
        embedding_dim=1024,
        persist_directory="./chroma_db",
        distance_function="cosine",
        use_custom_embeddings=False,
        device='cpu'
    )

    if retriever.is_collection_empty():
        query_database = load_jsonl_data(args.query_database_path)
        logger.info("Collection is empty, building index...")
        retriever.build_index(query_database, text_field="event", id_field="id", batch_size=128)
    
    for p in profiles:
        # p应该是（profile, events）两个组成
        user_agent = UserAgent(p)
        life_event_engine = OfflineLifeEventEngine(p.events)
        
        conv_generator = ConvHistoryGenerator(
            life_event_engine=life_event_engine,
            user_agent=user_agent,
            fast_conv_simulator=fast_conv_simulator,
            model=model,
            retriever=retriever,
        )
        
        conv_history = conv_generator.generate(
            max_events_number=args.max_events_number
        )
        
        # TODO: 想办法保存conv_history，最好所有profile所生成的conv_history保存到一个文件里。

if __name__ == '__main__':
    args = get_args()
    main(args)