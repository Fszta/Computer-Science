# Exercise

## Human stress classification in and through sleep

!!! note "Goal"

    
    The goal of the exercise is to build a model to classify the stress level of a patient given some physiological data.
    Four levels of stress are defined in the column `stress_level` as follow : 

    * 0 - low/normal
    * 1 – medium low
    * 2 - medium
    * 3 - medium high
    * 4 - high

    **It's a multiclass classification problem** 


### Input data

Here is an example of the data you'll use for the exercise.
> The dataset is composed of 631 samples

| snoring_range | respiration_rate | body_temperature | limb_movement_rate | blood_oxygen_levels | rapid_eye_movement | hour_of_sleep | heart_rate | stress_level |
|---------------|-----------------|------------------|--------------------|---------------------|--------------------|---------------|------------|--------------|
| 93.8          | 25.68           | 91.84            | 16.6               | 89.84               | 99.6               | 1.84          | 74.2       | 3            |
| 91.64         | 25.104          | 91.552           | 15.88              | 89.552              | 98.88              | 1.552         | 72.76      | 3            |
| 60            | 20              | 96               | 10                 | 95                  | 85                 | 7             | 60         | 1            |


#### You can download the dataset here : https://we.tl/t-3hhIBKN0GV

## Instructions

### Tooling

!!! info

    You can use any library, any tool, it's up to you. Here are some ideas :

    * pandas for data manipulation
    * matplotlib / seaborn for data visualization
    * sklearn : contains all the ML models and many functions you'll need
    * Jupyter / GoogleCollab

### 1 - Data exploration

* First, load the heart rate data into a Pandas DataFrame (in Python) on GoogleColab or in a Jupyter notebook.
* Check for any missing or null values in the dataset and handle them appropriately.
* Visualize the distribution of each feature (i.e., snoring_range, respiration_rate, body_temperature, 
limb_movement_rate, blood_oxygen_levels, rapid_eye_movement, hour_of_sleep, heart_rate) 
using histograms or density plots to get a sense of the range and distribution of each variable.
* Examine the correlation between the features and the target variable (i.e., stress_level) to identify any highly correlated features.

!!! tip 

    Feel free to add any visualization that make sense, be creative !



### 2 - Data preprocessing

* Split the data into training and testing datasets (e.g., 80% training and 20% testing).
* Scale the features to a similar range to avoid bias in distance calculations. You can use a method like 
min-max scaling or standard scaling. (You can try without !)


### 3 - KNN model training
!!! tip
    
    Have a look to `sklearn.neighbors.KNeighborsClassifier`...

* Use scikit-learn's KNeighborsClassifier to train a KNN model on the heart rate data.
* Choose an appropriate value of k (i.e., the number of nearest neighbors to consider) by trying different values and 
evaluating the performance of the model using a validation dataset or cross-validation.
* Fit the model on the training dataset.

### 4 - KNN Model evaluation

* Evaluate the performance of the KNN model on the test dataset using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC score.
* Visualize the model's performance using a confusion matrix or ROC curve.
* Analyze the results and identify areas of improvement for the model.

### 5 - Decision tree model training
same as `3 - KNN model training`

!!! tip

    Have a look to `sklearn.tree.DecisionTreeClassifier`...


### 6 - Decision tree model evaluation
same as `4 - KNN Model evaluation`


### 7 - Compare results
!!! tip 
    
    Just choose one of the model, justify your choice ! 

### Bonus - Compare with other Algorithms

### 8 - Send your work
!!! danger 

    Your work (the notebook), must be shared in a github repository, you'll send the link to your repo by email.


#### Thanks for the dataset

```
L. Rachakonda, A. K. Bapatla, S. P. Mohanty, and E. Kougianos, “SaYoPillow: Blockchain-Integrated Privacy-Assured IoMT 
Framework for Stress Management Considering Sleeping Habits”, IEEE Transactions on Consumer Electronics (TCE), Vol. 67,
 No. 1, Feb 2021, pp. 20-29.

L. Rachakonda, S. P. Mohanty, E. Kougianos, K. Karunakaran, and M. Ganapathiraju, “Smart-Pillow: An IoT based Device 
for Stress Detection Considering Sleeping Habits”, in Proceedings of the 4th IEEE International Symposium on Smart 
Electronic Systems (iSES), 2018, pp. 161--166.
```