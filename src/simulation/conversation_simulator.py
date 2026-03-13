import json
import os
from datetime import datetime
import time
from typing import Optional, Dict

from engine.event_engine import POI_Event
from profiles.profile_generator import UserProfile
from utils.utils import get_logger


class ConversationSimulator:
    def __init__(self, user_profile_generator, life_event_engine, user_agent, assistant_agent,
                 on_turn_update=None, logger_silent: bool = False, analysis_agent=None):
        self.user_pg = user_profile_generator
        self.life_event_engine = life_event_engine
        self.user_agent = user_agent
        self.assistant_agent = assistant_agent
        self.dialogue_log = []
        self.on_turn_update = on_turn_update
        self.analysis_agent = analysis_agent
        self.logger = get_logger(__name__, silent=logger_silent)

    # ------------------------------------------------------------------
    # Interface update helpers (used when on_turn_update is set)
    # ------------------------------------------------------------------

    def update_interface_dialog(self, role, content, episode_log, round_index=None, **kwargs):
        dialogue_turn = {'role': role, 'content': content}
        dialogue_turn.update(kwargs)
        self.logger.info(dialogue_turn)
        episode_log['dialogue'].append(dialogue_turn)

        if self.on_turn_update:
            self.on_turn_update({
                'step': 'turn',
                'round_index': round_index,
                'utterance': dialogue_turn,
            })
            time.sleep(0.05)
        return episode_log

    def update_interface_analysis(self, role, result, episode_log, round_index=None, **kwargs):
        analysis_turn = {'role': role, 'result': result}
        analysis_turn.update(kwargs)
        episode_log['analysis'].append(analysis_turn)

        if self.on_turn_update:
            data = {'step': 'analysis', 'round_index': round_index, 'analysis': episode_log['analysis']}
            data.update(kwargs)
            self.on_turn_update(data)
            time.sleep(0.05)
        return episode_log

    def update_interface_dropout(self, result, episode_log, round_index=None, **kwargs):
        episode_log['dropout'] = result
        if self.on_turn_update:
            data = {'step': 'dropout', 'round_index': round_index, 'dropout': result}
            data.update(kwargs)
            self.on_turn_update(data)
            time.sleep(0.05)
        return episode_log

    # ------------------------------------------------------------------
    # Environment initialization
    # ------------------------------------------------------------------

    def init_env(self, sequence_id):
        self.dialogue_log = []
        self.life_event_engine.set_event_sequence(sequence_id)
        user_id = self.life_event_engine.get_current_user_id()
        self.logger.info(f"👤 Initializing environment for user_id: {user_id}, sequence_id: {sequence_id}")
        self.user_profile = self.user_pg.get_profile_by_id(user_id)
        self.user_agent._build_profile(str(self.user_profile))
        self.assistant_agent._build_user_profile(self.user_profile)
        self.reset()

    def init_env_by_custom_profile_and_events(self, profile: dict, events: list):
        self.dialogue_log = []
        self.life_event_engine.set_event_sequence_by_profile_and_events(profile, events)
        self.user_profile = UserProfile.from_dict(profile)
        self.user_agent._build_profile(str(self.user_profile))
        self.assistant_agent._build_user_profile(self.user_profile)
        self.reset()

    def reset(self):
        self.user_agent.reinit()
        self.assistant_agent.reinit()

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def run_simulation(self, n_events: int = 5, n_rounds: int = 20, **config):
        strategy = ''
        for i in range(n_events):
            self.logger.info(f"[The {i+1}-th interaction scenario]")
            try:
                _, strategy = self.run_episode(
                    event_index=i + 1,
                    n_rounds=n_rounds,
                    enable_turn_analysis=False,
                    **config,
                )
                self.reset()
            except Exception:
                self.logger.exception(f"[The {i+1}-th interaction scenario] encountered an error")
                continue

    def run_episode(self, event_index: int = 1, n_rounds: int = 20, n_advice: int = 1,
                    user_config: Optional[Dict] = None, assistant_config: Optional[Dict] = None,
                    enable_turn_analysis: bool = False, strategy: str = ''):

        if enable_turn_analysis:
            event = self.life_event_engine.get_current_event()
        else:
            event = self.life_event_engine.generate_event()
        self.logger.info('[Event] ' + str(event))

        if self.on_turn_update:
            self.on_turn_update({
                'step': 'improvement_start' if enable_turn_analysis else 'init',
                'event': event,
                'round': 0,
                'event_index': event_index,
                'dialogue': [],
            })

        self.user_agent._build_environment(event)
        self.user_agent._build_chat_system_prompt()
        self.assistant_agent._build_system_prompt(event)

        episode_log = {
            'user': {
                'profile': self.user_profile.to_dict(),
                'profile_str': str(self.user_profile),
            },
            'event': event,
            'dialogue': [],
            'analysis': [],
            'dropout': {},
            'pre_profile': None,
            'enable_turn_analysis': enable_turn_analysis,
        }

        user_response = ''
        assistant_response = ''

        for round_index in range(1, n_rounds + 1):
            # --- User turn ---
            result = self.user_agent.respond(assistant_response, **user_config)
            action = result['action']
            user_response = result['response']
            self.logger.info(f'[User Action] {action}')

            if action == self.user_agent.action_space.END_CONVERSATION:
                break

            self.logger.info(f'[User] {user_response}')
            episode_log = self.update_interface_dialog(
                role='user',
                content=user_response,
                episode_log=episode_log,
                round_index=round_index,
                emotion=result.get('emotion'),
                memory_similarity=result.get('memory_similarity'),
            )

            # --- Assistant turn ---
            pre_intent, assistant_response = self.assistant_agent.respond(
                input=user_response, event=event, **assistant_config
            )
            self.logger.info(f'[Predicted intent by assistant] {pre_intent}')
            self.logger.info(f'[Assistant] {assistant_response}')

            episode_log = self.update_interface_dialog(
                role='assistant',
                content=assistant_response,
                episode_log=episode_log,
                round_index=round_index,
                pre_intent=pre_intent,
            )

            # --- Optional: turn-level analysis ---
            if enable_turn_analysis and self.analysis_agent:
                for _ in range(n_advice):
                    formatted_event = POI_Event.from_dict(event)
                    result_analysis = self.analysis_agent.assistant_quality_analysis(
                        user_profile=str(self.user_profile),
                        conversation_context=episode_log['dialogue'],
                        assistant_utterance=assistant_response,
                        event=formatted_event.desc(),
                        strategy=strategy,
                    )
                    self.logger.info(f"[Analysis (Assistant)] ({result_analysis['flags']}) {result_analysis['advice']}")

                    if result_analysis['flags']:
                        assistant_response = self.assistant_agent.revise_last_turn(
                            result_analysis['advice'], user_response
                        )
                        self.logger.info(f'[Assistant (Revised)] {assistant_response}')

                    episode_log = self.update_interface_analysis(
                        role='assistant',
                        result=result_analysis,
                        episode_log=episode_log,
                        round_index=round_index,
                    )
                    if result_analysis['flags']:
                        episode_log = self.update_interface_dialog(
                            role='assistant_revise',
                            content=assistant_response,
                            episode_log=episode_log,
                            round_index=round_index,
                        )

        # --- End-of-episode: preference summary ---
        try:
            dimensions = list(self.user_profile.preferences_value.keys())
        except Exception:
            dimensions = [list(x.keys())[0] for x in self.user_profile.preferences_value]
        reply = self.assistant_agent.summarize(event, dimensions, **assistant_config)
        self.logger.info(f'[Assistant summary] {reply}')
        episode_log['pre_profile'] = reply

        # --- Optional: dropout analysis ---
        strategy = ''
        if self.analysis_agent:
            try:
                formatted_event = POI_Event.from_dict(event)
                dropout_result = self.analysis_agent.predict_dropout(
                    conversation_context=episode_log['dialogue'],
                    user_profile=str(self.user_profile),
                    event=formatted_event.desc(),
                    intents=event.get('sub_intents', []),
                )
                self.logger.info(
                    f"[Dropout analysis] Risk: {dropout_result['risk']}\n"
                    f"Reason: {dropout_result['reason']}\n"
                    f"Strategy: {dropout_result['strategy']}"
                )
                strategy = dropout_result.get('strategy', '')
                episode_log = self.update_interface_dropout(
                    result=dropout_result,
                    episode_log=episode_log,
                    round_index=round_index,
                )
            except Exception:
                self.logger.exception("[Dropout analysis] Failed to analyze dropout")
                episode_log = self.update_interface_dropout(
                    result={'risk': 'Unknown', 'reason': 'Analysis failed.', 'strategy': ''},
                    episode_log=episode_log,
                    round_index=round_index,
                )

        self.dialogue_log.append(episode_log)
        return episode_log, strategy

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _format_content(self, text):
        text = text.strip()
        for left, right in [('"', '"'), ("'", "'"), ('"', '"'), ('\u2018', '\u2019')]:
            if text.startswith(left) and text.endswith(right):
                text = text[len(left):-len(right)].strip()
                break
        return text.replace('~', ' ')

    def get_dialogue_log(self):
        return self.dialogue_log

    def save(self, path='./logs', filename=None):
        os.makedirs(path, exist_ok=True)
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'sim_log_{timestamp}.json'

        full_path = os.path.join(path, filename)
        info_to_save = {
            'event_sequence_info': self.life_event_engine.get_current_sequence_info(),
            'dialogue_log': self.dialogue_log,
        }
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(info_to_save, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[✓] Simulator log saved to {full_path}")

            self.user_agent.save(os.path.join(path, 'user_' + filename))
            self.assistant_agent.save(os.path.join(path, 'assistant_' + filename))
            self.user_agent.model.save(os.path.join(path, 'user_model_' + filename))
        except Exception as e:
            self.logger.error(f"[✗] Failed to save simulator log: {e}")
