# Code snippets

In this page, you will find some basic code snippets that can help you in your machine-learning journey

| Python Code Snippet | Description |
| --- | --- |
| `import pandas as pd` | Importing the Pandas library as `pd` |
| `import numpy as np` | Importing the NumPy library as `np` |
| `df = pd.read_csv("your_file.csv", sep=";")` | Reading a CSV file and creating a Pandas DataFrame with a specified separator |
| `arr = df.values` | Creating a NumPy array from a Pandas DataFrame |
| `df = pd.DataFrame(arr, columns=['col1', 'col2', 'col3'])` | Creating a Pandas DataFrame from a NumPy array with specified column names |
| `df.head(n)` | Displaying the first `n` rows of a Pandas DataFrame |
| `df.describe()` | Displaying summary statistics of a Pandas DataFrame |
| `df.dropna(axis=0/1)` | Dropping rows or columns with missing values from a Pandas DataFrame |
| `df.fillna(df.mean(), inplace=True)` | Imputing missing values with the mean of the column in a Pandas DataFrame |
| `df = (df - df.mean()) / df.std()` | Standardizing the values in a Pandas DataFrame |
| `df = pd.get_dummies(df, columns=['cat_var'])` | Creating dummy variables for categorical variables in a Pandas DataFrame |
| `from sklearn.model_selection import train_test_split` | Importing the `train_test_split` function from scikit-learn |
| `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)` | Splitting a Pandas DataFrame into training and testing sets |
| `from sklearn.tree import DecisionTreeClassifier` | Importing the `DecisionTreeClassifier` class from scikit-learn |
| `clf = DecisionTreeClassifier()` | Creating a Decision Tree Classifier object |
| `clf.fit(X_train, y_train)` | Training a Decision Tree Classifier on a training set |
| `clf.score(X_test, y_test)` | Evaluating the accuracy of a trained Decision Tree Classifier on a testing set |
| `from sklearn.ensemble import RandomForestClassifier` | Importing the `RandomForestClassifier` class from scikit-learn |
| `from sklearn.model_selection import GridSearchCV` | Importing the `GridSearchCV` class from scikit-learn |
| `rfc = RandomForestClassifier()` | Creating a Random Forest Classifier object |
| `param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}` | Defining a parameter grid for hyperparameter tuning |
| `clf = GridSearchCV(rfc, param_grid, cv=5)` | Creating a Grid Search object |
| `clf.fit(X_train, y_train)` | Fitting the Grid Search object on a training set |
| `clf.best_params_` | Displaying the best hyperparameters found by the Grid Search |
| `confusion_matrix(y_true, y_pred)` | Creating a confusion matrix using scikit-learn |
| `plt.matshow(corr_matrix)` | Creating a correlation matrix plot using Matplotlib |
| `importance = clf.feature_importances_` | Creating a feature importance array from a trained scikit-learn model |
| `indices = np.argsort(importance)[::-1]` | Sorting feature indices in descending order of importance |
| `plt.bar(range(X_train.shape[1]), importance[indices])` | Creating a bar plot of feature importances using Mat
