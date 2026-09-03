import os

import numpy as np
from config.config import PATH
import kagglehub
class DataLoader:

    def __init__(self):
        self.path=kagglehub.dataset_download(PATH)





