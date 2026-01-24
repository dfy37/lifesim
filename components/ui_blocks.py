import streamlit as st
import pydeck as pdk
import time


# -----------------------------------------
# 地图渲染
# -----------------------------------------
def render_map(points):
    if not points:
        st.pydeck_chart(
            pdk.Deck(
                layers=[],
                initial_view_state=pdk.ViewState(latitude=40.75, longitude=-73.98, zoom=12)
            )
        )
    else:
        layer = pdk.Layer(
            "ScatterplotLayer",
            points,
            get_position="[lon, lat]",
            get_radius=30,
            get_color=[255, 0, 0],
            pickable=True,
        )
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=points[-1]["lat"], longitude=points[-1]["lon"], zoom=13)
            )
        )


# -----------------------------------------
# 简易聊天渲染器（占位）
# -----------------------------------------
class SimpleChatRenderer:
    def __init__(self, container):
        self.container = container

    def add_message(self, role, content):
        with self.container:
            with st.chat_message(role):
                st.write(content)
