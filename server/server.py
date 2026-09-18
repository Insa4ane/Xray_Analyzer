import logging
import tempfile
import os
from pydantic import BaseModel
import numpy as np
from sympy.stats.rv import probability

from DataLoader.Loader import DataLoader
from agent.XGBoostAgent import XGBoostAgent
from agent.XrayAgent import XrayAgent
import io
from PIL import Image
import torch as tc
class PredictionResponse(BaseModel):
    is_healthy: bool
    confidence: float

class Server:
    def __init__(self):
        self.agent, self.classifier = self._get_agents()

    @staticmethod
    def _get_agents():
        try:
            agent = XrayAgent()
            if not agent.load_model():
                logging.error("Brak zapisanego modelu CNN — uruchom train.py")
                return None, None

            classifier = XGBoostAgent(agent)
            if not classifier.load_model():
                logging.error("Brak zapisanego modelu XGBoost — uruchom train.py")
                return None, None

            return agent, classifier
        except Exception as e:
            logging.error(f"We cannot launch a server: {e}")
            return None, None

    def predict(self, uploaded_files):
        try:
            dataset=[]
            with tempfile.TemporaryDirectory() as tmp_dir:
                directories = self._get_directories(uploaded_files, tmp_dir)
                if directories:
                    loader=DataLoader()
                    for directory in directories:
                        image_tmp=loader.process_image(directory)
                        if image_tmp is not None:
                            dataset.append(image_tmp)
                    X = np.array(dataset).reshape(-1, 1, self.agent.size, self.agent.size)/255.0
                    X=tc.tensor(X, dtype=tc.float32)
                    predictions, proba=self.classifier.predict(X)
                    return [PredictionResponse(is_healthy=bool(pred == 1), confidence=float(conf)) for pred, conf in zip(predictions, proba)]
                else:
                    return []
        except Exception as e:
            logging.error(f"We cannot predict {e}")
            return None

    @staticmethod
    def _get_directories(uploaded_files, tmp_dir):
        try:
            files_dir=[]
            for i, uploaded_file in enumerate(uploaded_files):
                file=io.BytesIO(uploaded_file)
                image_tmp=Image.open(file)
                file_name=f"image_{i}.jpg"
                directory=os.path.join(tmp_dir,file_name)
                image_tmp.save(directory, format="JPEG")
                files_dir.append(directory)
            return files_dir
        except Exception as e:
            logging.error(f"We cannot return directories (ERROR in server): {e}")
            return []

































