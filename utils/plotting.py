import datetime

import matplotlib.pyplot as plt
import os
import logging
from config.config import PATH_HISTORY

class Plotting:
    def __init__(self):
        self.path_history=PATH_HISTORY

    def plot_history(self, history: dict, save: bool = True) -> str | None:
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
            for ax, metric in zip(axes.flat, metrics):
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], marker='o')
                ax.set_title(metric.capitalize())
                ax.set_xlabel("Epochs")
                ax.set_ylabel(metric.capitalize())
                ax.grid(True)
            axes.flat[-1].axis('off')
            fig.tight_layout()
            if save:
                plots_dir = os.path.join(self.path_history, "plots")
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                plot_path = os.path.join(plots_dir, f"training_plot_{timestamp}.png")
                fig.savefig(plot_path)
                logging.info(f"Saved plot to {plot_path}")
                plt.close(fig)
                return plot_path
            plt.show()
            return None

        except Exception as e:
            logging.error(f"We cannot create the plot: {e}")
            return None