import streamlit as st
import pandas as pd
import pydeck as pdk
import time
import random

# ==========================================
# 1. 模拟数据初始化
# ==========================================
def init_mock_data():
    if "sim_results" not in st.session_state:
        # 预设三个模拟事件
        st.session_state.sim_results = [
            {
                "id": 0,
                "event": "在时代广场丢失钱包",
                "scene": "繁华街区",
                "location": {"lat": 40.7580, "lon": -73.9855},
                "messages": [
                    {"role": "user", "content": "你好，我好像把钱包弄丢了。", "emotion": "焦急"},
                    {"role": "assistant", "content": "别担心，我能帮你。最后一次见到它是在哪？"}
                ]
            },
            {
                "id": 1,
                "event": "前往中央公园警察局报案",
                "scene": "政府机构",
                "location": {"lat": 40.7812, "lon": -73.9665},
                "messages": [
                    {"role": "user", "content": "我想报个案，我钱包在时代广场丢了。", "emotion": "沮丧"},
                    {"role": "assistant", "content": "好的，请提供一下您的身份证件和钱包特征。"}
                ]
            },
            {
                "id": 2,
                "event": "在咖啡馆等待监控结果",
                "scene": "室内休闲",
                "location": {"lat": 40.7614, "lon": -73.9776},
                "messages": [
                    {"role": "user", "content": "这都等了一个小时了，还没消息吗？", "emotion": "愤怒"},
                    {"role": "assistant", "content": "明白您的心情，监控调取需要一些时间，您可以先喝杯咖啡。"}
                ]
            }
        ]
    if "selected_event_idx" not in st.session_state:
        st.session_state.selected_event_idx = 0

# ==========================================
# 2. 组件函数
# ==========================================

def render_map(lat, lon):
    """渲染右侧地图"""
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=45)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": lat, "lon": lon}],
        get_position="[lon, lat]",
        get_color="[200, 30, 0, 160]",
        get_radius=100,
    )
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

def render_chat_column():
    """渲染中间对话栏"""
    idx = st.session_state.selected_event_idx
    event_data = st.session_state.sim_results[idx]
    
    st.subheader(f"💬 对话详情")
    st.info(f"**当前场景**: {event_data['scene']} | **事件**: {event_data['event']}")
    
    with st.container(height=550, border=True):
        for msg in event_data["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "emotion" in msg:
                    st.caption(f"🎭 情绪状态: {msg['emotion']}")

def render_right_column():
    """渲染右侧：地图 + 事件轴"""
    idx = st.session_state.selected_event_idx
    event_data = st.session_state.sim_results[idx]
    
    # 地图部分
    st.subheader("📍 实时位置")
    render_map(event_data["location"]["lat"], event_data["location"]["lon"])
    
    st.divider()
    
    # 时间轴部分
    st.subheader("📑 事件追踪")
    for i, ev in enumerate(st.session_state.sim_results):
        # 用不同颜色区分当前选中的事件
        is_current = (st.session_state.selected_event_idx == i)
        btn_type = "primary" if is_current else "secondary"
        
        if st.button(f"事件 {i+1}: {ev['event'][:12]}...", key=f"nav_{i}", 
                     use_container_width=True, type=btn_type):
            st.session_state.selected_event_idx = i
            st.rerun()

# ==========================================
# 3. 主界面布局
# ==========================================

st.set_page_config(page_title="LifeSim Demo", layout="wide")
init_mock_data()

# --- Sidebar: 配置 ---
with st.sidebar:
    st.title("👤 用户配置")
    st.text_input("用户姓名", value="张三")
    st.slider("仿真强度", 1, 10, 5)
    st.divider()
    if st.button("🚀 开始新仿真", use_container_width=True, type="primary"):
        with st.status("正在生成仿真路径..."):
            time.sleep(1)
            st.write("构建 Agent...")
            time.sleep(1)
        st.success("仿真就绪！")

# --- 主页面三栏分布 ---
# 此时侧边栏已占一部分，主页面分两列，分别对应你的 中间栏(6) 和 右侧栏(4)
col_mid, col_right = st.columns([6, 4])

with col_mid:
    render_chat_column()

with col_right:
    render_right_column()

# 底部页脚
st.caption("LifeSim Concept Demo - 侧边栏配置 | 中间栏对话 | 右侧栏地图与时间轴联动")