import streamlit as st
from components.ui_blocks import render_map, SimpleChatRenderer


def run_simulation_ui(profile, mode, n_events, n_rounds):

    st.markdown("## 地图区域（占位）")
    render_map([])

    st.markdown("## 模拟对话（占位）")

    chat_area = st.container()
    renderer = SimpleChatRenderer(chat_area)

    if st.button("开始模拟", key=f"start_{mode}"):
        renderer.add_message("user", f"开始模拟（模式：{mode}）")
        renderer.add_message("assistant", f"用户画像：{profile}")
        renderer.add_message("assistant", f"事件数量：{n_events}, 轮数：{n_rounds}")
