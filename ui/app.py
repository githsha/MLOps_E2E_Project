import streamlit as st
import requests

ip = requests.get("https://api.ipify.org").text

API_URL = "http://"+ip+":30080/predict"

st.title("ML Ops feedback analyzer")

text = st.text_area("Enter feedback")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter value")
    else:
        try:
            response = requests.post(API_URL, json={"text": text}, timeout=3)
            response.raise_for_status()
            st.success(response.json())
        except requests.exceptions.ConnectionError:
            st.error("🚨 API is not up. Make the API up.")
        except Exception as e:
            st.error(f"Error: {e}")

