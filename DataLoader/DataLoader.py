import numpy as np
from config.config import PATH
import kagglehub
class DataLoader:

    def __init__(self):
        self.path=kagglehub.dataset_download(PATH)

    def get_description(self) -> str:
        pass

    def get_photo(self): #pobieramy dane
        pass

    def split_data(self)->tuple[list, list, list, list]:
        pass

    def prepare_dataset(self)->np.ndarray:
        pass





