import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
API_KEY = os.getenv("API_KEY")
META_DIR = os.getenv("META_DIR")
CATEGORIES = {"NORMAL":1, "PNEUMONIA":0}
IMG_SIZE = 224
INPUT_CHANNELS = 1
CHANNEL_SIZES = [16, 32, 64]
EPOCHS = 100
BATCH_SIZE = 32
PATH_MODEL="Model_CNN/xray_cnn_model.pth"
PATH_XGB="Model_XGB/xgb_model.joblib"
PATH_HISTORY="History/"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/")
CHECXPERT_FILE='mimic-cxr-2.0.0-chexpert.csv.gz'
SPLIT_FILE='mimic-cxr-2.0.0-split.csv.gz'
META_FILE='mimic-cxr-2.0.0-metadata.csv.gz'
DROP_LABELS = ["Fracture", "Cardiomegaly", "Enlarged Cardiomediastinum"]

LABELS = [
    "Atelectasis", "Consolidation", "Edema", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

# Kolumny opisowe zdjęcia (nie są etykietami)
ID_COLUMNS = ["path", "dicom_id", "study_id", "subject_id", "split", "ViewPosition"]