import streamlit as st
from PIL import Image
import requests as rq

class Frontend:
    def __init__(self):
        pass

    def main_menu(self):
        st.title("Ocena zdjecia rtg na podstawie sztucznej inteligencji")
        st.header("Podaj zdjecie rtg a agent je oceni")
        uploaded_file=st.file_uploader("Wgraj zdjecie rtg")
        return uploaded_file



    def send_data(self, uploaded_file):
            image=Image.open(uploaded_file)
            st.image(image, caption='Uploaded image', use_container_width=True)
            button=st.button("Wyslij zdjecie na diagnoze!")
            if button:
                with st.spinner('Trwa analiza zdjęcia...'):
                    try:
                        files = {"uploaded_file": uploaded_file.getvalue()}
                        response = rq.post("http://localhost:5000/make_predict", files=files, timeout=30)
                        response.raise_for_status()
                    except rq.exceptions.RequestException as e:
                        st.error(f"Error! we cannot send an image: {e}")
                        return None
                st.success("Zdjęcie zostało przeanalizowane")
                return response.json()
            return None








