import streamlit as st
import time
import json
import os
import pandas as pd
import pydeck as pdk
from run_simulation import build_simulator
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_jsonl_data(path):
    data = []
    with open(path) as reader:
        for row in reader:
            data.append(json.loads(row))
    return data

# ----------------------------------------------------
# SessionState 初始化
# ----------------------------------------------------

def init_simulation():
    st.session_state["messages"] = []
    st.session_state["message_placeholders"] = []
    st.session_state["simulator_running"] = False
    st.session_state["last_update"] = 0
    st.session_state["simulator"] = None
    st.session_state["event_blocks"] = []
    st.session_state["map_points"] = []
    st.session_state["current_block_id"] = None

st.static_path = "./"

# ----------------------------------------------------
# 简单聊天渲染器（重写版）
# ----------------------------------------------------
class SimpleChatRenderer:
    def __init__(self, event_box, message_area, analysis_box=None):
        self.event_box = event_box
        self.message_area = message_area
        self.analysis_box = analysis_box
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "message_placeholders" not in st.session_state:
            st.session_state["message_placeholders"] = []

    def clear(self):
        """清空对话，但是不清空事件描述"""
        st.session_state["messages"] = []
        st.session_state["message_placeholders"] = []
        with self.message_area:
            st.markdown("")  # 清空消息区域，不清空事件区域(event_box)

    def add_event(self, event):
        """事件信息始终在顶部显示"""
        explicit_intents = [x['description'] + '\n' for x in event.get('sub_intents','无') if x['type'] == 'explicit']
        explicit_intents = '- '.join(explicit_intents)
        implicit_intents = [x['description'] + '\n' for x in event.get('sub_intents','无') if x['type'] == 'implicit']
        implicit_intents = '- '.join(implicit_intents)
        self.event_box.info(
            f"📍 **[场景]**: {event.get('dialogue_scene','无')}\n\n"
            f"📍 **[当前事件]**: {event.get('event','')}\n\n"
            f"📍 **[用户意图]**: {event.get('intent','无')}\n\n"
            f"📍 **[显性意图]**:\n\n"
            f"- {explicit_intents}\n"
            f"📍 **[隐性意图]**:\n\n"
            f"- {implicit_intents}\n"
        )

    def add_message(self, role, content, emotion, speed=0.01):
        messages = st.session_state["messages"]
        placeholders = st.session_state.get("message_placeholders", [])

        # 如果是修订，把上一条消息从界面和状态中移除
        if 'revise' in role:
            role = role.split('_')[0]
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == role:
                    messages.pop(i)
                    if i < len(placeholders):
                        try:
                            placeholders[i].empty()
                        except Exception:
                            pass
                        placeholders.pop(i)
                    break

        # 追加当前这条消息到状态
        messages.append({
            "role": role,
            "content": content,
            "emotion": emotion,
        })

        # 在消息区域中新建一个占位符，并用它来渲染当前消息
        with self.message_area:
            msg_placeholder = st.empty()
            with msg_placeholder.chat_message(role):
                placeholder = st.markdown("")
                if role == "user":
                    prefix = f"_(情绪: {emotion})_ "
                else:
                    prefix = ""

                typed = prefix
                for ch in content:
                    typed += ch
                    placeholder.markdown(typed)
                    time.sleep(speed)

        # 记录这个占位符，以便后续可以清除
        st.session_state["message_placeholders"].append(msg_placeholder)

    def add_dropout(self, risk, reason):
        with self.message_area:
            st.warning(
                f"🔔 **流失点分析**\n\n"
                f"• **流失风险:** {risk}\n\n"
                f"• **原因:** {reason}"
            )

    def add_analysis(self, round_index, role, advice):
        if self.analysis_box:
            with self.analysis_box:
                with st.expander(f"对于第{round_index}轮{role}话语的建议", expanded=True):
                    st.markdown(
                        f"{advice}"
                    )

# ---------------- 回调函数（核心） ----------------
def create_streamlit_callback(map_placeholder):
    """创建回调，不提前绑定 renderer；由 init 动态创建"""

    def rerender_map():
        with map_placeholder:
            with st.expander("地图", expanded=True):
                pts = st.session_state.get("map_points", [])
                if pts:
                    # inject per‑point icon_data dict
                    for p in pts:
                        p["icon_data"] = {
                            "url": "https://cdn-icons-png.flaticon.com/512/684/684908.png",
                            "width": 128,
                            "height": 128,
                            "anchorY": 128
                        }

                    layer = pdk.Layer(
                        "IconLayer",
                        pts,
                        get_icon="icon_data",
                        get_size=4,
                        size_scale=15,
                        get_position='[lon, lat]',
                        pickable=True,
                    )
                    # add sequence labels with pixel offsets to avoid overlap
                    label_data = []
                    offsets = [
                        [0, -70],
                        [-25, -40],
                        [25, -40],
                        [-35, -10],
                        [35, -10],
                    ]
                    for idx, p in enumerate(pts):
                        label_data.append({
                            "lon": p["lon"],
                            "lat": p["lat"],
                            "label": str(idx + 1),
                            "offset": offsets[idx % len(offsets)],
                        })

                    text_layer = pdk.Layer(
                        "TextLayer",
                        label_data,
                        get_position='[lon, lat]',
                        get_text='label',
                        get_color=[255, 0, 0],
                        get_size=32,
                        get_font_weight=700,
                        size_scale=1,
                        get_alignment_baseline="'bottom'",
                        get_pixel_offset='offset',
                    )
                    path_layer = None
                    # if len(pts) > 1:
                    #     path_layer = pdk.Layer(
                    #         "PathLayer",
                    #         [{"path": [[p["lon"], p["lat"]] for p in pts]}],
                    #         get_path="path",
                    #         get_color=[0, 123, 255],
                    #         width_scale=10,
                    #         width_min_pixels=3,
                    #     )
                    layers = [layer] + ([path_layer] if path_layer else []) + [text_layer]
                    # 取最新点作为地图中心
                    center_lat = pts[-1]["lat"]
                    center_lon = pts[-1]["lon"]

                    view_state = pdk.ViewState(
                        latitude=center_lat,
                        longitude=center_lon,
                        zoom=16
                    )
                    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state))
                else:
                    view_state = pdk.ViewState(latitude=40.7580, longitude=-73.9855, zoom=22)
                    st.pydeck_chart(pdk.Deck(layers=[], initial_view_state=view_state))

    def _callback(data):
        step = data["step"]

        # ======================================================
        # ❶ 收到 init → 新建一个事件块
        # ======================================================
        if step == "init":
            # 创建一个事件 UI block
            block = st.expander(label=f"第{data['event_index']}个交互事件场景", expanded=True)
            with block:
                event_box = st.empty()
                msg_row = st.container()
                with msg_row:
                    left_msg, right_analysis = st.columns([5,3])
                    with left_msg:
                        message_area = st.container()
                    with right_analysis:
                        analysis_box = st.container()
                        # analysis_box = st.expander("Analysis", expanded=False)

            # 记录 block
            st.session_state["current_block_id"] = len(st.session_state["event_blocks"])
            st.session_state["event_blocks"].append({
                "renderer": SimpleChatRenderer(event_box, message_area, analysis_box),
                "data": []   # 如需存储原始内容，可追加到这里
            })

            # 渲染事件
            renderer = st.session_state["event_blocks"][-1]["renderer"]
            renderer.add_event(data["event"])
            loc = data["event"].get("location_detail", {})
            if "latitude" in loc and "longitude" in loc:
                st.session_state["map_points"].append(
                    {"lat": loc["latitude"], "lon": loc["longitude"]}
                )
                rerender_map()
            return

        # ======================================================
        # ❷ turn / done → 写入当前 block
        # ======================================================
        block_id = st.session_state["current_block_id"]
        renderer = st.session_state["event_blocks"][block_id]["renderer"]

        if step == "turn":
            last = data["dialogue"][-1]
            renderer.add_message(last["role"], last["content"], last['emotion'])

        elif step == "analysis":
            last = data["analysis"][-1]
            if last['role'] == 'dropout':
                # 封装到renderer的类方法里
                renderer.add_dropout(last['result']['risk'], last['result']['reason'])
            else:
                if last['result']['flags']:
                    renderer.add_analysis(data['round_index'], last['role'], last['result']['advice'])

        elif step == "done":
            renderer.add_message("system", f"本事件结束（轮次数：{data.get('round_index','-')}）")

    return _callback


def run_simulation(event_box, message_area, map_placeholder, sequence_id, profile, n_events=2, n_rounds=3, n_exp=0):
    st.session_state["simulator_running"] = True

    # 构造渲染器 & 回调
    renderer = SimpleChatRenderer(event_box, message_area, None)
    callback = create_streamlit_callback(map_placeholder)

    exp_name = f"{sequence_id}_{str(n_exp)}"
    # 构造 simulator
    sim = build_simulator(callback, exp_name, config_path='./config.yaml')
    st.session_state["simulator"] = sim

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

    # 初始化环境 & 运行
    sim.init_env_by_custom_profile(profile)
    sim.run_simulation(n_events=n_events, n_rounds=n_rounds, **sim_config)
    sim.save(path=os.path.join('./logs', f'{exp_name}'))

    st.session_state["simulator_running"] = False
    st.session_state["last_update"] = time.time()


def render_custom_page():
    cfg = load_config('./config.yaml')

    # 初始化 Session（不影响画像）
    init_simulation()

    st.set_page_config(page_title="LiveSim Demo (Custom Persona)", layout="wide")
    st.title("自定义画像用户模拟器（Custom Persona）")

    if st.sidebar.button("返回主页"):
        st.session_state["page"] = "home"

    # -----------------------------------------------------------
    # 事件数量 / 对话轮数（保持与你原框架一致）
    # -----------------------------------------------------------
    n_events = st.sidebar.selectbox(
        "请选择要模拟的事件数量 (1-6): ",
        [i for i in range(1, 7)],
        index=1
    )
    n_rounds = st.sidebar.selectbox(
        "请选择每次对话要模拟的最大轮数 (1-10): ",
        [i for i in range(1, 11)],
        index=3
    )
    
    # -----------------------------------------------------------
    # 【侧边栏：画像填写区 + 保存按钮】
    # -----------------------------------------------------------
    st.sidebar.header("画像填写区")

    gender = st.sidebar.selectbox("性别", ["男", "女", "其他"], index=0)
    age = st.sidebar.text_input("年龄（示例：25）")
    marital = st.sidebar.text_input("婚姻状况（示例：单身/已婚）")
    religious = st.sidebar.text_input("宗教信仰")
    employment = st.sidebar.text_input("职业状态（示例：学生/上班族）")
    area = st.sidebar.text_input("居住区域（示例：城区/郊区）")
    income = st.sidebar.text_input("收入水平（示例：中等/偏高）")
    personality = st.sidebar.text_area("性格特征（逗号分隔）", placeholder="外向, 自律, 温和")
    body_state = st.sidebar.text_area("身体状态（文本）")
    sport_food_preferences = st.sidebar.text_area(
        "运动与饮食偏好（多行）"
    )
    preferences = st.sidebar.text_area(
        "偏好描述（多行）",
        placeholder="- 喜欢户外活动\n- 偏好力量训练\n- 不喜欢长跑"
    )

    save_button = st.sidebar.button("保存画像")

    # -----------------------------------------------------------
    # 【默认画像】— 当用户未点击保存按钮时使用
    # -----------------------------------------------------------
    default_profile = {
        "user_id": "",
        "religious": "",
        "employment": "",
        "marital": "",
        "race": "",
        "income": "",
        "area": "",
        "age": "",
        "gender": "",
        "bigfive": {},
        "personality": [],
        "preferences": [],
        "body_state": "",
        "sport_preferences": [],
        "preferences_value": dict,
        "conv_history": []
    }

    # -----------------------------------------------------------
    # 【保存画像逻辑】
    # -----------------------------------------------------------
    if save_button:
        custom_profile = {
            "user_id": "",
            "religious": religious,
            "employment": employment,
            "marital": marital,
            "race": "",
            "income": income,
            "area": area,
            "age": age,
            "gender": gender,
            "bigfive": {},
            "personality": [p.strip() for p in personality.split(",") if p.strip()],
            "preferences": [p.strip("- ").strip() for p in preferences.split("\n") if p.strip()],
            "body_state": body_state,
            "sport_preferences": [p.strip() for p in sport_food_preferences.split("\n") if p.strip()],
            "preferences_value": dict,
            "conv_history": []
        }
        
        st.session_state["custom_profile"] = custom_profile
        st.sidebar.success("画像已保存！")
    else:
        # 用户还没保存 → 用默认画像
        if "custom_profile" not in st.session_state:
            st.session_state["custom_profile"] = default_profile

    # -----------------------------------------------------------
    # 【侧边栏画像展示区】
    # -----------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("当前画像")

    profile = st.session_state["custom_profile"]

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown(f"**性别**：{profile['gender']}")
        st.markdown(f"**年龄**：{profile['age']}")
        st.markdown(f"**婚姻状况**：{profile['marital']}")
        st.markdown(f"**宗教信仰**：{profile['religious']}")

    with col2:
        st.markdown(f"**居住区域**：{profile['area']}")
        st.markdown(f"**收入水平**：{profile['income']}")
        st.markdown(f"**职业状态**：{profile['employment']}")
        st.markdown("**性格特征**：" + "、".join(profile["personality"]))

    st.sidebar.markdown("**身体状态**：")
    st.sidebar.markdown(profile["body_state"])
    
    st.sidebar.markdown("**运动或饮食偏好**：")
    for p in profile["sport_preferences"]:
        st.sidebar.markdown(f"- {p}")

    st.sidebar.markdown("**偏好描述**：")
    for p in profile["preferences"]:
        st.sidebar.markdown(f"- {p}")

    # -----------------------------------------------------------
    # 主界面 UI 块（保持你的框架）
    # -----------------------------------------------------------
    map_placeholder = st.empty()
    with map_placeholder:
        with st.expander("地图", expanded=True):
            view_state = pdk.ViewState(latitude=40.7580, longitude=-73.9855, zoom=11)
            st.pydeck_chart(pdk.Deck(layers=[], initial_view_state=view_state))

    event_box = st.empty()
    message_area = st.container()

    # -----------------------------------------------------------
    # Start Simulation 按钮
    # -----------------------------------------------------------
    start_button = st.button(
        "Start Simulation",
        disabled=st.session_state["simulator_running"]
    )

    if start_button:
        # reset
        for key in list(st.session_state.keys()):
            if key != "custom_profile":   # 保留画像
                del st.session_state[key]
        init_simulation()
        # 再把画像写回 state
        # （避免 init_simulation 覆盖）
        st.session_state["custom_profile"] = profile

        # 默认自定义用户 ID
        sequence_id = "custom-1"

        # TODO：将画像塞入 user_config (你之后定义)
        # sim.run_simulation(...) 内你能在 callback 或 engine 读取 custom_profile
        run_simulation(
            event_box=event_box,
            message_area=message_area,
            map_placeholder=map_placeholder,
            sequence_id=sequence_id,
            profile=profile,
            n_events=n_events,
            n_rounds=n_rounds,
            n_exp=0
        )
