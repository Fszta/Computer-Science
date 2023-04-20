# Exercise 2

## Instructions
We will use a dataset containing patient information like cholesterol, sugar in blood etc...
The aim is to train a classification model to predict if a patient is exposed to heart attack or not.

You can download the data [here](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/download?datasetVersionNumber=2). 
It will download a zip with two files, we will use `heart.csv` file


Here a table summarizing the dataset:

| Column Name | Description |
| --- | --- |
| Age | Age of the patient |
| Sex | Sex of the patient |
| exang | Exercise induced angina (1 = yes; 0 = no) |
| ca | Number of major vessels (0-3) |
| cp | Chest Pain type: <ul><li>Value 1: typical angina</li><li>Value 2: atypical angina</li><li>Value 3: non-anginal pain</li><li>Value 4: asymptomatic</li></ul> |
| trtbps | Resting blood pressure (in mm Hg) |
| chol | Cholestoral in mg/dl fetched via BMI sensor |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| rest_ecg | Resting electrocardiographic results: <ul><li>Value 0: normal</li><li>Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)</li><li>Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria</li></ul> |
| thalach | Maximum heart rate achieved |
| target | Target variable: 0 = less chance of heart attack, 1 = more chance of heart attack |

### 1 - Create and train an SVM classifier
In this part, simply train an SVM classifier on the heart dataset using its default parameter.
!!! tip

    As for the previous exercise don't forget to:

    * explore the data, provide few visualization, check feature importance
    * clean it if necessary
    * evaluate your model, provide the confusion matrix etc... 

    Remember to check the CSV separator and set the `sep` parameter accordingly if you encounter issues when reading the 
    CSV file using Pandas. For example:

    ````python
    data = pd.read_csv("my_data.csv", sep=";")

    ````

### 2 - Change manually parameters 
Train your model by changing C & kernel. You can have a look to [sklearn documentation]("https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html")
to see what are the possible values.
````python
svm.SVC(kernel='poly', C=10)
````

### 3 - Perform GridSearch to find optimal parameters
For this part, you need to apply grid search on your SVM classifier to find the best parameters
You need to use `GridSearchCV` class to tune the following parameters: `C` & `kernel`.

!!! tip

    If you don't remember how does it work, have a look here : 

    [See an example](../parameter-tuning/){ .md-button .md-button--primary }
    
    You can import `GridSearchCV` as follows : 
    ```python
    from sklearn.model_selection import GridSearchCV
    ```
### 4 - Perform RandomizedSearch to find optimal parameters
For this part, you need to apply random search on your SVM classifier to find the best parameters
You need to use `RandomizedSearchCV` class to tune the following parameters: `C` & `kernel`.

!!! tip
    
    You can import `RandomizedSearchCV` as follows : 
    ```python
    from sklearn.model_selection import RandomizedSearchCV
    ```


### 5 - Automate multi-model training and tuning
The goal of this step is to define some python functions that will automatically train
models and perform parameters tuning. 


!!! tip
    
    One idea is to associate a dictionary to each model, and pass it 
    to a function to run training on all your models, and output the model that perform the best.
    You can access the `best` model from a grid object as follows :
    ```python
    grid = GridSearchCV(model, parameters, cv=5)
    grid.fit(X_train, y_train)
    grid.best_estimator_
    ```
    
```python

models_with_parameters = {
    svm.SVC(): {'C': [0.1, 1, 100, 1000], ['kernel': 'linear', 'rbf', 'poly']},
    KNeighborsClassifier(): {'n_neighbors': [3, 5, 7],'weights': ['uniform', 'distance']}
}

```

### 6 - Send your Github project
