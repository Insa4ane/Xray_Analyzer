import logging
from agent.XGBoostAgent import XGBoostAgent
from agent.XrayAgent import XrayAgent
from DataLoader.Loader import DataLoader
class Server:
    def __init__(self):
        self.agent, self.classifier = self._getAgents()

    def _getAgents(self):
        try:
            agent, classifier = self._load_existing_models()
            if agent is not None and classifier is not None:
                return agent, classifier
            return self._train_new_models()
        except Exception as e:
            logging.error(f"We cannot launch a server: {e}")
            return None, None

    def _load_existing_models(self):
        agent = XrayAgent()
        if not agent.load_model():
            return None, None

        classifier = XGBoostAgent(agent)
        if not classifier.load_model():
            return None, None

        return agent, classifier

    def _train_new_models(self):
        agent = XrayAgent()
        dataset = DataLoader()
        X_train, X_test, y_train, y_test = dataset.split_data()
        counts = dataset.count_training_images_per_class()

        agent.run(X_train, X_test, y_train, y_test, counts)
        agent.save_model()

        classifier = XGBoostAgent(agent)
        classifier.run(X_train, X_test, y_train, y_test)
        classifier.save_model()

        return agent, classifier




