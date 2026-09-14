from frontend import Frontend
from PIL import Image

def main():
    front=Frontend()
    uploaded_file=front.main_menu()
    if uploaded_file:
        result=front.send_data(uploaded_file)

if __name__ == "__main__":
    main()


