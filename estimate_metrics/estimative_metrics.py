"""Analises about the metrics and plots"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, accuracy_score

class MetricsAnaliser:
    """"""
    def __init__(self) -> None:
        pass

    def plot_confusion_matrix(self, y_true: np.array, y_pred: np.array, classes: list, title: str ="Confusion Matrix", cmap: plt = plt.cm.Purples, save_as: str = "fig.png") -> plt.subplot:
        """Plot a consfusion matrix with all classes, considering the y_true and y_pred of the models 

        Args:
            y_true (np.array): True answers about the results
            y_pred (np.array): Prediction of the model
            classes (list): List with all the classes names
            title (str, optional): A text to be the title of plot. Defaults to "Confusion Matrix".
            cmap (plt, optional): Type of colors to fill the table. Defaults to plt.cm.Purples.
            save_as (str, optional): The name so save  the plots. Defaults to "fig.png".

        Returns:
            plt.subplot: Return a a subplot of confusion matrix, 
        """
        self.print_main_metrics(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True',
            xlabel='Predicted')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], '.2f'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(save_as)

        return ax

    def print_main_metrics(self, y_test: str, y_predict: str) -> None:
        """Print principal metrics about the model

        Args:
            y_test (str): True values about the results
            y_predict (str): Value predicted with model
        """
        print('Accuracy:', accuracy_score(y_test, y_predict))
        print('F1-Score:', f1_score(y_test, y_predict, average='macro'))
        print('Precision:', precision_score(y_test, y_predict, average='macro'))
        print('Recall:', recall_score(y_test, y_predict, average='macro'))