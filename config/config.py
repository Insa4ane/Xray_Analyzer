import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
API_KEY = os.getenv("API_KEY")
PATH="paultimothymooney/chest-xray-pneumonia"
CATEGORIES = {"NORMAL":1, "PNEUMONIA":0}
IMG_SIZE = 224
INPUT_CHANNELS = 1
CHANNEL_SIZES = [16, 32, 64]
EPOCHS = 150
BATCH_SIZE = 32
PATH_MODEL="Model/xray_cnn_model.pth"
PATH_XGB="Model/xgb_model.joblib"