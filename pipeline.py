from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC

__authors__ = "Radhakrishnan Iyer"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Radhakrishnan Iyer"
__email__ = "srivatsan65@gmail.com"
__status__ = "Development"

estimator_list = {
    "logistic": "LogisticRegression",
    "decision_tree": "tree.DecisionTreeClassifier",
    "naive_bayes":"MultinomialNB",
    "random_forest":"RandomForestClassifier",
    "xgboost": "xgb.XGBClassifier",
    "svc": "SVC"
}

def get_estimators():
    return estimator_list.keys()

def load_pipeline(estimator="logistic"):
    if estimator == "logistic":
        pipeline = Pipeline([
            ('clf', LogisticRegression(random_state=19))
        ])
    elif estimator == "decision_tree":
        pipeline = Pipeline([
            ("clf", tree.DecisionTreeClassifier(max_depth=3, random_state=42))
        ])
    elif estimator == "naive_bayes":
        pipeline = Pipeline([
            ('clf', MultinomialNB())
        ])
    elif estimator == "random_forest":
        pipeline = Pipeline([
            ('clf', RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30))
        ])
    elif estimator == "xgboost":
        pipeline = Pipeline([
            ('clf', xgb.XGBClassifier())
        ])
    elif estimator == "svc":
        pipeline = Pipeline([
            ('clf', SVC(kernel='linear') )
        ])
    else:
        print("estimator unavailable in pipeline")
    return pipeline