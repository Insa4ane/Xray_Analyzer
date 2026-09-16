import logging
from DataLoader.Loader import DataLoader
from agent.XrayAgent import XrayAgent
from agent.XGBoostAgent import XGBoostAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    try:
        loader=DataLoader()
        if loader.path:
            X_train, X_test, y_train, y_test=loader.split_data()
            agent=XrayAgent()
            count=loader.count_training_images_per_class()
            history, result_agent=agent.run(X_train, X_test, y_train, y_test, count)
            if history and result_agent:
                logging.info(f"result agent: {result_agent}")
                agent.save_model()
                classifier = XGBoostAgent(agent)
                result_xgb=classifier.run(X_train, X_test, y_train, y_test)
                if result_xgb:
                    logging.info(f"Results XGB: {result_xgb}")
                    classifier.save_model()
                    return True
    except Exception as e:
        logging.error(f"Something's gone wrong with train xgb. {e}")
        return False



if __name__=="__main__":
    main()






