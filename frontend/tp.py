import streamlit as st

upload_img = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

print(upload_img)