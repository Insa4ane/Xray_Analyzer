import os
import numpy as np
import torch as tc
from config.config import META_DIR, CATEGORIES, IMG_SIZE, CHECXPERT_FILE, SPLIT_FILE, META_FILE, DROP_LABELS
import cv2
import random
import logging
from sklearn.model_selection import train_test_split
import pandas as pd
class DataLoader:

    def __init__(self, is_required_to_download:bool = True):
        self.size=IMG_SIZE
        self.meta_dir=META_DIR #directory for metadata

    #process image for pyspark
    def process_image(self, image_path:str, clipLimit=2.0, tileGridSize=(8,8)):
        try:
            img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                return None
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize) #better contrast, if sth is not that way you want, you'll change parameters (e.g (10,10))
            img_array = clahe.apply(img_array) #we apply the contrast
            return cv2.resize(img_array, (self.size, self.size))
        except Exception as e:
            logging.warning(f"Warning! {e}")
            return None

    def build_dataset(self):
        if self.meta_dir is None or not os.path.isdir(self.meta_dir):
            logging.error(f"Error! meta_dir '{self.meta_dir}' does not exist (set META_DIR in .env)")
            raise FileNotFoundError(self.meta_dir)

        chex=pd.read_csv(os.path.join(self.meta_dir, CHECXPERT_FILE))
        split=pd.read_csv(os.path.join(self.meta_dir, SPLIT_FILE))
        meta=pd.read_csv(os.path.join(self.meta_dir, META_FILE), usecols=["dicom_id", "ViewPosition"])
        if chex.empty or split.empty or meta.empty:
            logging.error(f"Error! {chex.shape} or {split.shape} or {meta.shape} are empty")
            raise ValueError(f"{chex.shape} and {split.shape} or {meta.shape} are empty")
        merge=pd.merge(split, meta, on='dicom_id')
        merge=merge[merge["ViewPosition"].isin(['AP', 'PA'])]
        dataset=pd.merge(merge, chex, on=['study_id', 'subject_id']) #merged arrays
        subject = dataset["subject_id"].astype(str)
        dataset['path']= (
                "files/p" + subject.str[:2]
        + "/p" + subject
        + "/s" + dataset["study_id"].astype(str)
        + "/" + dataset["dicom_id"] + ".jpg"
        )
        dataset=dataset.drop(columns=DROP_LABELS)
        return dataset

    def split_data(self): #main function
        dataset = self.build_dataset() #build dataset
        #split for train/test sets
        train=dataset[dataset["split"] == 'train']
        test=dataset[dataset["split"] == 'test']
        val=dataset[dataset["split"] == 'validate']
        return train, val, test


    def prepare_for_training(self, dataset:list[list]):
        X=[]
        y=[]
        try:
            for img_array, label in dataset:
                X.append(img_array)
                y.append(label)
            X = np.array(X).reshape(-1, 1, self.size, self.size)/255.0 #for pytorch (earlier, the 1 was the last (t have tought that I would use a tensorflow))
            y=np.array(y)
            if len(X)==len(y):
                X=tc.tensor(X, dtype=tc.float32)
                y=tc.tensor(y, dtype=tc.float32)
                return X, y
            else:
                logging.error(f"Error! Długość X ({len(X)}) != długość y ({len(y)})")
                return None, None
        except Exception as e:
            logging.error(f"Error! We cannot  {e}")
            return None, None

    def count_training_images_per_class(self) -> dict:
        train_dir = os.path.join(self.path, 'chest_xray', 'train')
        counted_photos = {}
        for category in CATEGORIES:
            folder_path = os.path.join(train_dir, category)
            counted_photos[category] = len(os.listdir(folder_path))
        return counted_photos











