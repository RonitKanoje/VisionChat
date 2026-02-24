import streamlit as st
from backend.Agentic_AI import chatBot, retrieveThreads
from backend.prompt import prompt1
from langchain_core.messages import HumanMessage,SystemMessage
from PIL import Image
import uuid
import requests
import os

# Streamlit Page Configuration
st.set_page_config(page_title="AI Chatbot ", layout="wide")

# Defining functions
def generateThreadID():
    return str(uuid.uuid4())

def resetChat():
    st.session_state['message_history'] = []
    st.session_state['thread_id'] = generateThreadID()
    addThread(st.session_state['thread_id'])
    
def addThread(thread_id):
    if thread_id not in st.session_state['thread_chats']:
        st.session_state['thread_chats'].append(thread_id)

# def loadConversation(thread_id):
#     # Placeholder for loading conversation logic
#     state = chatBot.get_state(config={'configurable': {'thread_id': thread_id}})
#     return state.values.get('message_history', [])

def loadConversation(thread_id):
    state = chatBot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def showChats(tempMsg, isSubmit):
    # If submit just happened, ignore last user + assistant message
    if isSubmit and len(tempMsg) >= 2:
        tempMsg = tempMsg[:-2]

    for msg in tempMsg:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.text(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.text(msg["content"])

def ai_stream(className):
    for message, metadata in chatBot.stream(
        {"messages": [
            SystemMessage(content=prompt1),
            HumanMessage(content=(user_text + ' ' + className))
        ]},
        config=CONFIG,
        stream_mode="messages"
    ):
        if message.__class__.__name__ == "AIMessageChunk":
            if message.content:
                yield message.content


# Session State Initialization
if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generateThreadID()

if 'thread_chats' not in st.session_state:
    st.session_state['thread_chats'] = retrieveThreads()

# Sidebar - Chat History
with st.sidebar:
    st.title("Chat History")

    if st.button("New Chat"):
        resetChat()
        st.session_state['message_history'] = []

    st.divider()    

    for thread in st.session_state['thread_chats']:
        if st.button(f"Thread {thread}"):
            st.session_state['thread_id'] = thread
            msgs = loadConversation(thread)

            tempMsg = []
            for msg in msgs:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                tempMsg.append({
                    "role": role,
                    "content": msg.content
                })

            st.session_state['message_history'] = tempMsg

# Main UI
st.title("AI Chatbot")

# User Input
user_text = st.text_input(
    "Enter your message",
    placeholder="Ask something about the image or any general question...",
    label_visibility="collapsed"
)

# Image Uploader

API_BASE_URL = os.getenv("API_BASE_URL")
PREDICT_URL = f"{API_BASE_URL}/predict"

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", width=300)
    files = {
    "file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)
    }

    urlRes = requests.post(
        PREDICT_URL,
        files=files
    )
    if urlRes.status_code == 200:
        data = urlRes.json()
        class_names = ", ".join(data['class_name'])
    else:
        st.error(f"API Error :{urlRes.status_code}")

else:
    image = None

# for message in st.session_state['message_history']:
#     with st.chat_message(message['role']):
#         st.text(message['content'])


isSubmit = False
#Submit Button
if st.button("Submit"):
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    if user_text or image:
        # Add user message only when Submit is clicked
        if(image and user_text):
            st.session_state['message_history'].append({"role": "user", "content": user_text + class_names})
            with st.chat_message("user"):
                st.text(user_text)
        elif user_text:
            st.session_state['message_history'].append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.text(user_text)
        elif (image):
            st.session_state['message_history'].append({"role": "user", "content": class_names})
            with st.chat_message("user"):
                st.text(user_text)

        with st.chat_message("assistant"):
            response = st.write_stream(ai_stream(class_names))
            resAppend = response
        st.session_state['message_history'].append({"role": "assistant", "content": resAppend})
        isSubmit = True


    else: 
        st.warning("Please enter text or upload an image.")

showChats(st.session_state['message_history'],isSubmit)