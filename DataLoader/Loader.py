import os
import numpy as np
import torch as tc
from config.config import PATH, CATEGORIES, IMG_SIZE
import kagglehub
import cv2
import random
import logging

class DataLoader:

    def __init__(self):
        self.path=self.dataset_download()
        self.size=IMG_SIZE

    @staticmethod
    def dataset_download():
        return kagglehub.dataset_download(PATH)

    def process_image(self, image_path:str):
        try:
            img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                return None
            return cv2.resize(img_array, (self.size, self.size))
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

    def split_data(self): #main function
        train_dir = os.path.join(self.path,'chest_xray', 'train')
        test_dir = os.path.join(self.path,'chest_xray', 'test')
        train = self.build_dataset(train_dir)
        test = self.build_dataset(test_dir)

        if not train or not test:
            logging.error(f"Error! {train} or {test} or both are empty")
            raise FileNotFoundError(f"{train} and {test} are empty")

        result_test = self.prepare_for_training(test)
        result_train = self.prepare_for_training(train)

        if result_test is None or result_train is None:
            logging.error("Error! prepare_for_training failed for train and/or test set")
            raise ValueError("prepare_for_training returned None for train and/or test set")

        X_test, y_test = result_test
        X_train, y_train = result_train
        return X_train, X_test, y_train, y_test

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














