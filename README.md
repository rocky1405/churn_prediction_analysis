# churn_prediction_analysis

### Usage Guide:

### Installation Instructions

* Python version: Python 3.7 or higher

* Python Package Dependency
```
pip install scikit-learn==0.19.2
pip install xgboost==1.3.3
pip install pandas==0.23.4
pip install numpy==1.15.1
pip install seaborn==0.9.0
pip install scipy==1.1.0
pip install matplotlib==2.2.3
```

### USAGE:

#### For model training:
1. COMMAND: 
   `python train.py <PATH_TO_TRAINING_DIR> <SUBSCRIPT_OF_THE_CLASSIFIER_MODEL_FILE>`

   >  CAUTION: make sure that python command invokes python 3.7 or higher 

2. OUTPUT: 

   ​	- Prints the training and test accuracies on the screen.

   ​	- classifier model as a pickle file is stored in the /models directory
   
### INTERPRETATION:

- Average Precision: Recommended for Multi-class classification. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html  
- Precision Score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
- Recall Score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html 
- Confusion Matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html  (Printed for both the training data and the validation data)
- Classification Report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

### PACKAGE STRUCTURE:

#####  Train.py : Model training script (You start from here)

* dataset.py => package contains routines to create a training data set from a training folder.
* trainer.py => package contains routines for creating a training workflow. The pipeline.py class contains various user defined estimator (ML classifiers) which are imported from scikit learn distribution. User of this classifier product is free to add more classifier by simply importing them from the sklearn package and adding corresponding initializers in the pipeline.py class file.
The trainer package also provides training and validation accuracy metrics and serializes the model into a pickle object. This pickle object is to be imported by the test.py script during unseen/new document classification.
* data_analysis.py => file contains routines to analyse the input data with the help of boxplots, bar charts, correlation charts and sub plots. It helps us understand how the data is interconnected with each other as well as the importance of different variables.

### Train Metrics:

The pre-processed data is passed through multiple algorithms to find the best one for the dataset. Below are the metrics for each algorithm.

1.	Logistic Regression:
```
Accuracy Score: 80.02%
Test Classification Report:
        precision   recall  f1-score  support
0          0.84      0.90     0.87     1052
1          0.63      0.50     0.56     355
avg/total  0.79      0.8      0.79     1407
```

2.	Decision Tree Classifier:
```
Accuracy Score: 79.53%
Test Classification Report:
        precision   recall  f1-score  support
0          0.81      0.94     0.87     1052
1          0.68      0.36     0.47     355
avg/total  0.78      0.8      0.77     1407
```

3.	Multinomial Naïve Bayes
```
Accuracy Score: 70.78%
Test Classification Report:
        precision   recall  f1-score  support
0          0.89      0.69     0.78     1052
1          0.45      0.75     0.56     355
avg/total  0.78      0.71     0.73     1407
```

4.	Random Forest Classifier:
``` 
Accuracy Score: 81.94%
Test Classification Report:
        precision   recall  f1-score  support
0          0.84      0.94     0.89     1052
1          0.72      0.47     0.57     355
avg/total  0.81      0.82     0.81     1407
```

5.	XGBoost:
```
Accuracy Score: 77.96%
Test Classification Report:
        precision   recall  f1-score  support
0          0.83      0.88     0.86     1052
1          0.58      0.48     0.52     355
avg/total  0.77      0.78     0.77     1407
```
6.	SVC:
```
Accuracy Score: 79.67%
Test Classification Report:
        precision   recall  f1-score  support
0          0.85      0.89     0.87     1052
1          0.62      0.52     0.56     355
avg/total  0.79      0.8      0.79     1407
```

Based on the metrics, it is clear that for the current dataset with the pre-processing implemented, Random Forest Classifier algorithm is the best suited algorithm to train a model. Hence, a model was trained and the resulting pickle file was saved to the models folder.
