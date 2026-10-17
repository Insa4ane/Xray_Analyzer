import logging
import os

from DataLoader.Loader import DataLoader
from agent.XrayAgent import XrayAgent
from agent.XGBoostAgent import XGBoostAgent
from config.config import PATH_HISTORY

os.makedirs(PATH_HISTORY, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PATH_HISTORY, "train.log"), encoding="utf-8"),
    ],
)


def main():
    try:
        loader = DataLoader()
        if not loader.path:
            logging.error("Dataset path is empty")
            return False

        X_train, X_test, y_train, y_test, X_val, y_val = loader.split_data()
        agent = XrayAgent()
        count = loader.count_training_images_per_class()

        history, result_agent = agent.run(X_train, X_val, X_test, y_train, y_val, y_test, count)
        if not (history and result_agent):
            logging.error("Training of XrayAgent failed")
            return False
        logging.info(f"result agent: {result_agent}")

        agent.save_model()
        if not agent.save_history_and_results(history, result_agent):
            logging.warning("History and results of XrayAgent were not saved")

        classifier = XGBoostAgent(agent)
        result_xgb = classifier.run(X_train, X_test, y_train, y_test)
        if not result_xgb:
            logging.error("Training of XGBoostAgent failed")
            return False
        logging.info(f"Results XGB: {result_xgb}")

        classifier.save_model()
        return bool(classifier.save_results(result_xgb))
    except Exception as e:
        logging.error(f"Something's gone wrong with train xgb. {e}")
        return False


if __name__ == "__main__":
    main()