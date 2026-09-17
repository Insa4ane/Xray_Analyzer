import streamlit as st
from PIL import Image
import requests as rq

class Frontend:
    def __init__(self):
        self.url="http://localhost:5000/"

    @staticmethod
    def main_menu():
        st.title("Ocena zdjecia rtg na podstawie sztucznej inteligencji")
        st.header("Podaj zdjecie rtg a agent je oceni")
        uploaded_file=st.file_uploader("Wgraj zdjecie rtg", accept_multiple_files=True)
        return uploaded_file

    def send_data(self, uploaded_files):
            if not uploaded_files:
                st.error("Uploaded files is empty")
                return None
            for uploaded_file in uploaded_files:
                image=Image.open(uploaded_file)
                st.image(image, width='stretch')

            button=st.button("Upload images")
            if button:
                with st.spinner('Trwa analiza na diagnoze...'):
                    try:
                        files_to_send = [
                            ("files", (f.name, f.getvalue(), f.type))
                            for f in uploaded_files
                        ]
                        response = rq.post(self.url+"make_predict", files=files_to_send, timeout=30)
                        response.raise_for_status()
                    except rq.exceptions.RequestException as e:
                        st.error(f"Error! we cannot send an image: {e}")
                        return None
                st.success("Zdjęcia zostały przeanalizowane!")
                return response.json()
            return None
