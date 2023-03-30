## Decision Tree

## What is a decision tree ?
A decision tree is a non-parametric supervised learning algorithm used for `classification` and `regression` tasks.
It is a tree-like structure where each internal node represents a test on an attribute or feature, 
each branch represents the outcome of the test, and each leaf node represents a class label or a numerical value that 
corresponds to a prediction.

!!! example "An abstract example"

    ```mermaid
    
        graph TD
        A{Is feature X >= 5?} -->|yes| B{Is feature Y >= 7?}
        B -->|yes| C[Class 1 - leaf node]
        B -->|no| D[Class 2 - leaf node]
        A -->|no| E{Is feature X >= 3?}
        E -->|yes| F[Class 3 - leaf node]
        E -->|no| D[Class 4 - leaf node]
        
        
    ```


The goal of a decision tree algorithm is to learn a model that can accurately predict the target variable for new, 
unseen data based on a set of training examples. The algorithm works by recursively splitting the data into subsets
based on the values of the attributes or features, until each subset belongs to a single class or has reached a 
stopping criterion.

!!! question "Can i use it for classification and regression?"
There are two main types of decision trees: classification trees and regression trees. Classification trees are used 
for predicting categorical class labels, while regression trees are used for predicting continuous numerical values.


!!! question "How is built a decision tree ?"
To build a decision tree, the algorithm follows a top-down, greedy approach that recursively partitions the data based
on the attribute that maximizes the information gain or minimizes the impurity at each node. 
The information gain measures the reduction in entropy or impurity of the target variable by splitting the data on 
a particular attribute, while the impurity measures the degree of uncertainty or randomness in the target variable.

Once the tree is built, the algorithm can use it to make predictions on new, unseen data by traversing the tree from
the root node down to a leaf node based on the values of the attributes. At each internal node, the algorithm checks 
the value of the corresponding attribute, and follows the appropriate branch based on the outcome of the test. 
At each leaf node, the algorithm outputs the class label or numerical value that corresponds to the prediction.

## Pros & Cons
!!! tip
    
    You can easily visualize a decision tree, have a look to `sklearn.tree.plot_tree`

Decision trees have several advantages, including their interpretability, ease of use, and ability to handle both 
categorical and numerical data. However, they also have some limitations, such as their tendency to overfit the 
training data if not pruned or regularized, their sensitivity to small changes in the data, and their inability to 
capture complex interactions between features.

## Deep dive in the algorithm

!!! question "How the algorithm work for a classification problem ?"
For classification problems, decision trees commonly use the Gini impurity or the information gain (also called entropy)
to measure the quality of a split. The Gini impurity measures the probability of misclassifying a randomly chosen sample
from the dataset, while the information gain measures the reduction in entropy (i.e., uncertainty) after the split. 
The split that minimizes the impurity or maximizes the information gain is chosen.

!!! question "How the algorithm work for a regression problem ?"
For regression problems, decision trees commonly use the mean squared error (MSE) to measure the quality of a split.
The MSE measures the variance of the target variable within each subset of the split. 
The split that minimizes the MSE is chosen.

Once the best split is chosen, the decision tree algorithm partitions the input data into two subsets based on the 
split, and then recursively applies the same process to each subset until a stopping criterion is met.
The stopping criterion could be a maximum depth of the tree, a minimum number of samples in each leaf node,
or other criteria.



!!! example "A less abstract example"
    
    Imagine you are developing an embedded system to monitor heart rate activity, what kind on information can you deduce 
    from the physiological data ?


    ```mermaid
    
        graph TD;
        A((Heart Rate <= 80))
        A --> |Yes| B((Age <= 30))
        A --> |No| C((Age <= 40))
        B --> |Yes| D(At Rest)
        B --> |No| E(Not At Rest)
        C --> |Yes| E
        C --> |No| F(At Rest)

    ```