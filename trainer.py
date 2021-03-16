"""
    The file has functionality to select the right model for the
    given data set from a host of models and train
"""
import os
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from dataset import Dataset
from config import choose_best_model, train_test_split_ratio
from pipeline import get_estimators, load_pipeline
warnings.filterwarnings("ignore")

__authors__ = "Radhakrishnan Iyer"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Radhakrishnan Iyer"
__email__ = "srivatsan65@gmail.com"
__status__ = "Development"


class Trainer:
    """
        The class has functionality to select the right model for the
        given dataset from a host of models and train
    """

    def __init__(self, data_path, model_name="churn_prediction", estimator="logistic"):
        """
            initialize all the variables
            :param data_path: the path where the dataset is located
            :param train_test_split_ratio: the ratio in which we want to split the training data into
            :param model_name: the name of the model to be used for saving the pickle file
            :param estimator: a default algorithm we want to choose to train the model
        """
        self.data = Dataset(data_path).pre_process_data()
        self.X = self.data.loc[:, self.data.columns != "Churn"]
        self.y = self.data.loc[:, self.data.columns == "Churn"]
        self.random_state = 101
        self.train_test_split_ratio = train_test_split_ratio
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.y,
                                                                                test_size=self.train_test_split_ratio,
                                                                                random_state=self.random_state)
        self.estimator = estimator
        self.model_name = model_name
        self.pipeline = None
        self.classification_report_test, self.classification_report_train, self.confusion_matrix_test, self.confusion_matrix_train, self.precision_score, self.recall_score = None, None, None, None, None, None


    def train(self):
        """
            select the best algorithm from the different ones available and
            train the model
        """
        if choose_best_model is True:
            best_accuracy = 0
            best_estimator = self.estimator
            for estimator in get_estimators():
                self.estimator = estimator
                print("Evaluation for estimator *********** ", estimator)
                self.pipeline = load_pipeline(self.estimator)
                print(self.pipeline)
                self.pipeline.fit(self.train_x, self.train_y)
                accuracy = self.eval("accuracy")
                if self.estimator == "logistic" or self.estimator == "naive_bayes" or self.estimator == "svc":
                    print(self.pipeline.named_steps["clf"].coef_)
                else:
                    print(self.pipeline.named_steps["clf"].feature_importances_)

                print("Accuracy is = ", accuracy)
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    best_estimator = estimator

            print("Chosen best estimator = ", best_estimator, "best accuracy is = ", best_accuracy)
            self.estimator = best_estimator
        self.pipeline = load_pipeline(self.estimator)
        self.pipeline.fit(self.train_x, self.train_y)

    def eval(self, metric=None):
        """
            get all the different evaluation metrics to measure the performance of the model
        """
        predictions_test = self.pipeline.predict(self.test_x)
        predictions_train = self.pipeline.predict(self.train_x)
        self.classification_report_test = classification_report(self.test_y, predictions_test)
        self.classification_report_train = classification_report(self.train_y, predictions_train)
        self.confusion_matrix_test = confusion_matrix(self.test_y, predictions_test)
        self.confusion_matrix_train = confusion_matrix(self.train_y, predictions_train)
        self.precision_score = precision_score(self.test_y, predictions_test, average='micro')
        self.recall_score = recall_score(self.test_y, predictions_test, average='micro')
        if metric == "accuracy":
            return accuracy_score(self.test_y, predictions_test)
        return self.classification_report_test, self.classification_report_train, self.confusion_matrix_test, self.confusion_matrix_train, self.precision_score, self.recall_score

    def save(self, path=os.getcwd()):
        """
            save the model after training
            :param path: the path where the model needs to be saved
        """
        path = os.path.join(path, "models")
        if not os.path.exists(path):
            os.mkdir(path)
        joblib.dump(self.pipeline, os.path.join(path, 'classifier_' + self.model_name + '.pkl'))

if __name__ == "__main__":
    tr = Trainer("data\\dataset.csv")
    tr.train()
