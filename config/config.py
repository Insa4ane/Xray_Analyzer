import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
API_KEY = os.getenv("API_KEY")
PATH="paultimothymooney/chest-xray-pneumonia"
CATEGORIES = {"NORMAL":1, "PNEUMONIA":0}
IMG_SIZE = 224