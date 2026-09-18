import streamlit as st

from frontend import Frontend
from PIL import Image

def main():
    front=Frontend()
    uploaded_files=front.main_menu()
    if uploaded_files:
        result=front.send_data(uploaded_files)
        if result:
            for uploaded_file, result in zip(uploaded_files, result):
                col1, col2 = st.columns(2)
                with col1:
                    image = Image.open(uploaded_file)
                    st.image(image,width='stretch')
                with col2:
                    if result['is_healthy']:
                        st.success("Zdrowe pluca")
                    else:
                        st.error("Chore pluca (pneumonia)")
                    if result['confidence']:
                         st.metric("Pewnosc", f"{result['confidence']*100}%")

if __name__ == "__main__":
    main()


