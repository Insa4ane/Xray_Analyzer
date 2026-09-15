from DataLoader.Loader import DataLoader
from agent.XGBoostAgent import XGBoostAgent
from agent.XrayAgent import XrayAgent
import logging

logging.basicConfig( #for console log
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    dataset = DataLoader()
    X_train, X_test, y_train, y_test = dataset.split_data()
    counts = dataset.count_training_images_per_class()  # do zbalansowania sieci

    cnn_agent = XrayAgent()
    cnn_agent.run(X_train, X_test, y_train, y_test, counts)
    cnn_agent.save_model()
    xgb_agent = XGBoostAgent(cnn_agent)
    xgb_agent.run(X_train, X_test, y_train, y_test)
    xgb_agent.save_model()

    logging.info("Training completed")


if __name__ == "__main__":
    main()
