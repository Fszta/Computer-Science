# Parameter tuning

## What is parameter tuning and why it is important? 

Parameter tuning is the process of selecting the optimal values for the parameters of a machine learning model. 
In most machine learning algorithms, there are various parameters that need to be specified to achieve optimal 
performance on a particular dataset. These parameters can significantly affect the performance of the model and can 
vary depending on the problem at hand.

## How to apply it ? 
!!! info

    There are different ways to perform parameter tuning, you can do it 'manually' by iterating over a list
    of parameters (in such a case, it's a gridsearch)

### GridSearch
The grid search approach would involve creating a list of values for each hyperparameter, and then evaluating the 
performance of the model for all possible combinations of these values. The combination that results in the best 
performance (e.g., highest accuracy, lowest error rate) is then chosen as the optimal set of hyperparameters.

#### Example with KNN
Assuming we want to tune parameters of a KNN classifier. Let's do it with the two following parameters : 

* `n_neighbors` the number of neighbors. 
For this parameter we want to test 3 value : [3, 5, 7]
* `weights` the weight function used in prediction. 
For this parameter we want to test 2 value : ['uniform', 'distance']

##### The stupid way
!!! warning

    Never do that way.

The stupid way would be to manually create some classifier with all the combination...

* `KNeighborsClassifier(n_neighbors=3, weights='uniform')` 
* `KNeighborsClassifier(n_neighbors=3, weights='distance')` 
* `KNeighborsClassifier(n_neighbors=5, weights='uniform')` 
* `KNeighborsClassifier(n_neighbors=5, weights='distance')` 
* `KNeighborsClassifier(n_neighbors=7, weights='uniform')` 
* `KNeighborsClassifier(n_neighbors=7, weights='distance')` 

##### The best way - With GridSearchCV class
A better way would to :

* Define a ``parameter grid`` as follows :
```python
param_grid = {'n_neighbors': [3, 5, 7],
              'weights': ['uniform', 'distance']}
```

* Define a Knn classifier without parameter:
````python
knn_classifier = KNeighborsClassifier()
````

!!! tip

    Sklearn come with a built-in class `GridSearchCV`, design to perform parameter tuning and also
    CrossValidation !
* Pass the knn classifier and the parameter grid to `GridSearchCV()` object :
````python
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
````
* You can now train your model !
````python
grid_search.fit(X_train, y_train)
````
!!! warning
    
    As you can see, with this approach, you will not call the fit method of your knn_classifier object directly
    but on the grid_search object

!!! example "A pseudo example"

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    
    # Create an instance of KNN classifier
    knn_classifier = KNeighborsClassifier()
    
    # Define a grid of hyperparameters to search over
    param_grid = {'n_neighbors': [3, 5, 7],
                  'weights': ['uniform', 'distance']}
    
    # Create a grid search object
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
    
    # Fit the grid search object to the training data
    grid_search.fit(X_train, y_train)
    
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Training Score: ", grid_search.best_score_)
    print("Test Score: ", grid_search.score(X_test, y_test))
    ```


### RandomSearch
Another very similar approach is called `RandomSearch`. The only difference with `GridSearch` is that you don't have to
explicitly pass the values of the parameters, you have to pass the distribution of the parameters, eg : a range for an 
integer

#### Example
Have a look to GridSearch, this is almost the same. The only changes are the input parameters : 
this time you pass a parameter_distribution dictionary, like : 
```python
param_dist = {'n_neighbors': randint(1, 10),
              'weights': ['uniform', 'distance']}
```

And use `RandomizedSearchCv` class instead of GridSearchCv.
````python
random_search = RandomizedSearchCV(knn, param_distributions=param_dist, cv=5, n_iter=20, random_state=42)
````
!!! info

    RandomizedSearchCV takes an additional parameter n_iter which corresponds to the number of iteration to perform.


### Other methods
The two previous methods are very basics, however they can perform well on classic machine-learning problems. 
There many other methods to perform parameter tuning. The choice will highly depends on the complexity of the problem
you are trying to solve. In some cases (when the search space for hyperparameters is large), it's very time-consuming to
evaluate models on each possibility, in those cases, it's better to use other techniques like **Bayesian Optimization**.
