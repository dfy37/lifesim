import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime
import time
import os
import random
import json
import yaml
from utils.utils import get_logger
import streamlit.components.v1 as components
from run_simulation import build_simulator
from streamlit_timeline import timeline
from engine.event_engine import OnlineLifeEventEngine

logger = get_logger(__name__)

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_jsonl_data(path):
    """加载JSONL格式数据"""
    data = []
    with open(path) as reader:
        for row in reader:
            data.append(json.loads(row))
    return data

def save_jsonl_data(path, data):
    """保存数据到JSONL文件"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_user_by_id(user_id, users_path):
    """根据ID加载用户数据"""
    users = load_jsonl_data(users_path)
    for u in users:
        if u["user_id"] == user_id:
            return u
    return None

def save_user_profile(user_id, updated_profile, users_path, saved_path=None):
    """保存更新后的用户画像"""
    if not saved_path:
        saved_path = users_path
    users = load_jsonl_data(users_path)
    logger.info("Loaded users ...")
    for i, user in enumerate(users):
        if user['user_id'] == user_id:
            users[i] = updated_profile
            logger.info(f"Updated user profile for user_id: {updated_profile}")
            break
    save_jsonl_data(saved_path, users)
    return users

def save_event_sequence(sequence_id, updated_event_seq, events_path):
    """保存更新后的事件序列到原始 JSONL"""
    events = load_jsonl_data(events_path)
    logger.info("Loaded event sequences ...")
    for i, e in enumerate(events):
        if e["id"] == sequence_id:
            events[i] = updated_event_seq
            logger.info(f"Updated event sequence for id={sequence_id}")
            break
    save_jsonl_data(events_path, events)
    return events

def init_simulation():
    """初始化仿真状态"""
    if 'timeline_events' not in st.session_state:
        st.session_state.timeline_events = []
    if 'selected_event_idx' not in st.session_state:
        st.session_state.selected_event_idx = -1
    if 'message_blocks' not in st.session_state:
        st.session_state.message_blocks = []
    if 'simulator_running' not in st.session_state:
        st.session_state.simulator_running = False
    if 'event_counter' not in st.session_state:
        st.session_state.event_counter = 0
    if 'history_messages' not in st.session_state:
        st.session_state.history_messages = []

def clear_state():
    st.session_state.timeline_events = []
    st.session_state.selected_event_idx = -1
    st.session_state.message_blocks = []
    st.session_state.simulator_running = False
    st.session_state.event_counter = 0

class MapRender:
    def __init__(self, container):
        self.container = container
        with self.container:
            st.subheader("🗺️ Real-time Location")
            self.slot = st.empty()
        
    def render(self, lat, lon):
        with self.slot:
            view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=45)
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=[{"lat": lat, "lon": lon}],
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_radius=100,
            )
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state)
            st.pydeck_chart(deck, width='stretch')

class TimelineRender:
    def __init__(self, container, height=300):
        self.container = container
        self.height = height
        with self.container:
            st.subheader("📑 User Life Experience")
            self.slot = st.empty()
        # self.render()

    def _to_timelinejs_date(self, dt_like: str):
        """
        支持格式：
        - 'YYYY-MM-DD'
        - 'YYYY-MM-DD HH:MM:SS'
        - 'YYYY-MM-DD HH:MM:SS, Weekday'
        """
        clean = dt_like.split(",")[0].strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(clean, fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            raise ValueError(f"Unsupported datetime format: {dt_like}")
        return {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "second": dt.second,
        }
    
    def render(self, new_event = None):
        if new_event:
            st.session_state.timeline_events.append(new_event)
        events = st.session_state.timeline_events
        n = len(events)
        if n == 0:
            return
        # TimelineJS 需要的 JSON schema
        data = {
            "options": {
                "start_at_end": True,
            },
            "events": []
        }
        for i, ev in enumerate(events):
            t = ev.get("time")
            data["events"].append({
                "start_date": self._to_timelinejs_date(t),
                "text": {
                    "headline": f"Event {i+1}",
                    "text": ev.get("event", "")
                }
            })
        
        with self.slot:
            timeline(data, height=self.height)  # 不接收任何返回值 -> 只展示

class MainAreaRender:
    def __init__(self, container):
        self.container = container
    
    def render(self, data):
        step = data['step']
            
        if step == 'init':
            with self.container:
                # with st.expander(label=f"The {data['event_index']}th interaction scene", expanded=True):
                event_box = st.container()
                message_box = st.container()
                self.add_event(event_box, data['event'])
            
            st.session_state.message_blocks.append(message_box)
        
        elif step == 'turn':
            last = data["utterance"]
            message_box = st.session_state.message_blocks[-1]
            with message_box:
                self.add_message(
                    last["role"], 
                    last["content"], 
                    last.get('emotion', '')
                )
                
    def add_event(self, event_box, event):
        """事件信息始终在顶部显示"""
        explicit_intents = [x['description'] + '\n' for x in event.get('sub_intents', [{'description': '', 'type': ''}]) if x['type'] == 'explicit']
        explicit_intents = '- '.join(explicit_intents)
        implicit_intents = [x['description'] + '\n' for x in event.get('sub_intents', [{'description': '', 'type': ''}]) if x['type'] == 'implicit']
        implicit_intents = '- '.join(implicit_intents)
        event_box.info(
            f"📍 **[Time]**: {event.get('time','无')}\n\n"
            f"📍 **[Location]**: {event.get('location','无')}\n\n"
            f"📍 **[Weather]**: {event['weather']['description']}\n\n"
            f"📍 **[Event]**: {event.get('event','')}\n\n"
            # f"📍 **[User Intention]**: {event.get('intent','无')}\n\n"
            # f"📍 **[Explicit Intention]**:\n\n"
            # f"- {explicit_intents}\n"
            # f"📍 **[Implicit Intention]**:\n\n"
            # f"- {implicit_intents}\n"
        )
        
        return

    def add_message(self, role, content, emotion, speed=0.01):
        msg_placeholder = st.empty()
        with msg_placeholder.chat_message(role):
            
            if role == "user":
                st.caption(f"_Emotion: {emotion}_ ")
            #     prefix = f"_(Emotion: {emotion})_ "
            # else:
            #     prefix = ""

            placeholder = st.markdown("")
            typed = ""
            for ch in content:
                typed += ch
                placeholder.markdown(typed)
                time.sleep(speed)
        
        return

def render_sidebar_selection(sequence_ids):
    """渲染侧边栏选择区域"""
    sequence_id = st.sidebar.selectbox(
        "请选择要模拟的用户ID: ",
        sequence_ids,
        index=0
    )
    n_events = st.sidebar.selectbox(
        "请选择要模拟的事件数量 (1-6): ",
        list(range(1, 7)),
        index=1
    )
    
    return sequence_id, n_events

def render_sidebar_user_profile(user_profile, cfg):
    """渲染侧边栏用户画像编辑区域"""
    st.sidebar.subheader("👤 用户画像")
    
    with st.sidebar.form(key="user_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.text_input("性别", value=user_profile.get('gender', ''))
            age = st.text_input("年龄", value=str(user_profile.get('age', '')))
            marital = st.text_input("婚姻状况", value=user_profile.get('marital', ''))
            religious = st.text_input("宗教信仰", value=user_profile.get('religious', ''))
        
        with col2:
            area = st.text_input("居住区域", value=user_profile.get('area', ''))
            income = st.text_input("收入水平", value=user_profile.get('income', ''))
            employment = st.text_input("职业状态", value=user_profile.get('employment', ''))
            personality = st.text_input("性格特征（用逗号分隔）", 
                                       value=", ".join(user_profile.get('personality', [])))
        
        preferences = st.text_area("交互偏好（每行一个）", 
                                  value="\n".join(user_profile.get('preferences', [])), 
                                  height=200)
        
        profile_save_button = st.form_submit_button("💾 保存用户画像")
        
        if profile_save_button:
            updated_profile = {
                'user_id': user_profile['user_id'],
                'gender': gender,
                'age': age,
                'marital': marital,
                'religious': religious,
                'area': area,
                'income': income,
                'employment': employment,
                'personality': [p.strip() for p in personality.replace('、', ',').split(',') if p.strip()],
                'preferences': [p.strip() for p in preferences.split('\n') if p.strip()]
            }
            
            try:
                save_user_profile(
                    user_profile['user_id'], 
                    updated_profile, 
                    users_path=cfg["paths"]["users_path"], 
                    saved_path=cfg["paths"]["users_path"]
                )
                st.sidebar.success("✅ 用户画像已保存成功！")
                return updated_profile
            except Exception as e:
                st.sidebar.error(f"❌ 保存失败: {str(e)}")
                return user_profile
        
        return user_profile

def render_sidebar(sequence_ids, cfg, seqid2uid, seqid2eseq, uid2user):
    sequence_id, n_events = render_sidebar_selection(sequence_ids)
    
    current_user_id = seqid2uid[sequence_id]
    user_profile = load_user_by_id(current_user_id, users_path=cfg["paths"]["users_path"])
    event_seq = seqid2eseq[sequence_id].copy()
    
    user_profile = render_sidebar_user_profile(user_profile, cfg)
    uid2user[user_profile['user_id']] = user_profile
    
    return sequence_id, n_events, uid2user, seqid2eseq

def create_layout():
    col_mid, col_right = st.columns([6, 4])
    
    with col_mid:
        next_button = st.button(
            "Next Life Event Generation",
            disabled=st.session_state["simulator_running"]
        )
        main_area = st.container()
        
    with col_right:
        map_area = st.container()
        timeline_area = st.container()
    
    return next_button, main_area, map_area, timeline_area

def render_chat_page():
    # 加载配置
    cfg = load_config('./config.yaml')
    
    # 初始化状态
    init_simulation()
    st.static_path = "./"
    
    # 加载数据
    events = load_jsonl_data(cfg["paths"]["events_path"])
    users = load_jsonl_data(cfg["paths"]["users_path"])
    
    # 构建映射
    sequence_ids = [e['id'] for e in events]
    seqid2eseq = {e['id']: e for e in events}
    seqid2uid = {e['id']: e['user_id'] for e in events}
    uid2user = {u['user_id']: u for u in users}
    
    st.set_page_config(page_title="LifeSim Demo", layout="wide")
    st.title("LifeSim")

    next_button, main_area, map_area, timeline_area = create_layout()
    main_area_render = MainAreaRender(main_area)
    map_render = MapRender(map_area)
    timeline_render = TimelineRender(timeline_area, height=400)
    
    sequence_id, n_events, uid2user, seqid2eseq = render_sidebar(
        sequence_ids=sequence_ids,
        cfg=cfg,
        seqid2uid=seqid2uid,
        seqid2eseq=seqid2eseq,
        uid2user=uid2user
    )
    
    event_engine = OnlineLifeEventEngine(cfg["paths"]["events_path"])
    event_engine.set_event_sequence(sequence_id)

    event_engine.set_event_index(st.session_state.event_counter)
    if len(st.session_state.timeline_events) > 0:
        event = st.session_state.timeline_events[-1]
        map_render.render(lat=event['location_detail']['latitude'], lon=event['location_detail']['longitude'])
        timeline_render.render()
        main_area_render.render({'step': 'init', 'event': event, 'event_index': st.session_state.event_counter})
    
    if len(st.session_state.message_blocks) > 0:
        message_block = st.session_state.message_blocks[-1]
        for m in st.session_state.history_messages:
            with message_block.chat_message(m['role']):
                st.markdown(m['content'])      
            
    user_query = st.chat_input('Tall to our simulated person')
    if user_query:
        st.session_state.history_messages.append({'role': 'user', 'content': user_query})
        message_block = st.session_state.message_blocks[-1]
        with message_block.chat_message("assistant"):
            st.markdown(user_query)
            
    if next_button:
        # clear_state()
        event_engine.set_event_index(st.session_state.event_counter)
        event = event_engine.generate_event()
        map_render.render(lat=event['location_detail']['latitude'], lon=event['location_detail']['longitude'])
        timeline_render.render(event)
        main_area_render.render({'step': 'init', 'event': event, 'event_index': st.session_state.event_counter})
        st.session_state.event_counter += 1
        st.session_state.history_messages = []
        
if __name__ == '__main__':
    render_chat_page()