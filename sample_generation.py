from src.simulation.conv_history_generator import ConvHistoryGenerator
from src.simulation.fast_conv_simulator import FastConvSimulator
from src.profiles.profile_generator import UserProfileGenerator
from src.agents.user_agent import UserAgent
from src.engine.event_engine import OfflineLifeEventEngine

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--input-root", required=True)
    
    args = parser.parse_args()
    
    return args

def main(args):
    profile_generator = UserProfileGenerator(...)
    profiles = profile_generator.generate_profiles()
    
    fast_conv_simulator = FastConvSimulator(max_turns)
    
    for p in profiles:
        user_agent = UserAgent(p)
        life_event_engine = OfflineLifeEventEngine(p.events)
        
        conv_generator = ConvHistoryGenerator(
            user_profile_generator,
            life_event_engine,
            user_agent,
            fast_conv_simulator,
            model,
            retriever,
        )

if __name__ == '__main__':
    args = get_args()
    main(args)