import streamlit as st
import pandas as pd
import time

# Custom CSS for background and text color
st.markdown(
    """
    <style>
        .stApp {
            background-color: #ffe4ec;
            color: #800000;
        }
        h1, h2, h3, h4, h5, h6, p, span, div {
            color: #800000 !important;
        }
        .chat-message {
            font-size: 1.3rem !important;
            margin-bottom: 0.5rem;
            display: flex;
        }
        .bot-message {
            background: #fff0f5;
            color: #800000;
            padding: 0.7rem 1.2rem;
            border-radius: 18px 18px 18px 0px;
            max-width: 70%;
            align-self: flex-start;
            box-shadow: 0 1px 4px #00000010;
        }
        .user-message {
            background: #ffd6e6;
            color: #800000;
            padding: 0.7rem 1.2rem;
            border-radius: 18px 18px 0px 18px;
            max-width: 70%;
            margin-left: auto;
            align-self: flex-end;
            box-shadow: 0 1px 4px #00000010;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Layout: logo on the left, title on the right
col1, col2 = st.columns([1, 8])
with col1:
    st.image("vivi_logo_1.png", width=80)
with col2:
    st.markdown("<h1>ViVi ChatBot</h1>", unsafe_allow_html=True)
    
if "messages" not in st.session_state:
    st.session_state.messages = [
        {'role': 'bot', 'content': 'Hi! I am ViVi! I can assit you in answering queries related to the videos'}
    ]

# display chat history
for msg in st.session_state.messages:
    if msg['role'] == 'bot':
        st.markdown(f"<div class='chat-message'><div class='bot-message'>{msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message'><div class='user-message'>{msg['content']}</div></div>", unsafe_allow_html=True)

# chat input for user
user_input = st.chat_input("  Type your message here...")
if user_input:
    if user_input.strip().lower() == "clear":  #clear the chat history
        st.session_state.messages = [
        {'role': 'bot', 'content': 'Hi! I am ViVi! I can assit you in answering queries related to the videos'}
        ]
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"<div class='chat-message'><div class='user-message'>{user_input}</div></div>", unsafe_allow_html=True)
    # here we need add to chatbots response logic
    bot_reply = "I am currently dumb! Have a nice day!"
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.spinner('ViVi is typing...'):
        time.sleep(1.5)
    st.markdown(f"<div class='chat-message'><div class='bot-message'>{bot_reply}</div></div>", unsafe_allow_html=True)
    
