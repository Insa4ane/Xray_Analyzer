import os
import numpy as np
from config.config import PATH, CATEGORIES
import kagglehub
import cv2
import random
import logging

class DataLoader:

    def __init__(self):
        self.path=self.dataset_download()

    def dataset_download(self):
        return kagglehub.dataset_download(PATH)

    def process_image(self, image_path:str, img_size:int=224):
        try:
            img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                return None
            return cv2.resize(img_array, (img_size, img_size))
        except Exception as e:
            logging.warning(f"Warning! {e}")
            return None

    def build_dataset(self, base_directory:str): #function return vector if photo is normal return photo and 1, else photo and 0 ->it is set for AI agent
        dataset=[]
        for category, label in CATEGORIES.items():
            category_path = os.path.join(base_directory, category)
            if os.path.exists(category_path):
                for file_name in os.listdir(category_path):
                    img_path= os.path.join(category_path, file_name)
                    img_array = self.process_image(img_path)
                    if img_array is not None:
                        dataset.append([img_array, label])
            else:
                logging.error(f"Category {category} is not exist")
                raise FileNotFoundError(f"{category_path} is not exist")
        random.shuffle(dataset)
        return dataset

    def split_data(self):
        train_dir = os.path.join(self.path, 'train')
        test_dir = os.path.join(self.path, 'test')
        train = self.build_dataset(train_dir)
        test = self.build_dataset(test_dir)
        if train and test:
            return train, test
        else:
            logging.error(f"Error! {train} or {test} or both are empty")
            raise FileNotFoundError(f"{train} and {test} are empty")







