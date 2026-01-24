import streamlit as st
# 子页面导入
from assistant_eval_page import render_assistant_eval_page
from user_life_page import render_user_life_page

st.set_page_config(page_title="全周期用户模拟器", layout="wide")

# 初始化路由状态
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# -------------------------------
# 首页 UI
# -------------------------------
def render_home():
    st.markdown(
        """
        <div style='text-align:center; margin-top:40px;'>
            <h1 style='font-size:42px; font-weight:700;'>全周期用户模拟器</h1>
            <p style='font-size:20px; color:#666;'>选择用户画像模式，进入对应模拟界面</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="
                padding:20px;
                border-radius:12px;
                background:#f7f7f8;
                border:1px solid #e5e5e5;
                height:200px;">
                <h3>助手模拟评估模式</h3>
                <p style='color:#666;'>从系统事件库与画像库选择用户模拟。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("进入助手模拟评估模式", use_container_width=True):
            st.session_state["page"] = "assistant_eval"

    with col2:
        st.markdown(
            """
            <div style="
                padding:20px;
                border-radius:12px;
                background:#f7f7f8;
                border:1px solid #e5e5e5;
                height:200px;">
                <h3>自由交谈模式</h3>
                <p style='color:#666;'>可以实时生成用户经历以及与模拟的用户进行交谈。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("进入自由交谈模式", use_container_width=True):
            st.session_state["page"] = "custom"

# -------------------------------
# 路由
# -------------------------------
if st.session_state["page"] == "home":
    render_home()

elif st.session_state["page"] == "assistant_eval":
    render_assistant_eval_page()

elif st.session_state["page"] == "custom":
    render_user_life_page()