# Before developing
!!! warning
    
    Before creating a machine learning algorithm, there are several steps that should be taken to ensure that the 
    problem is well-defined, the data is appropriate and that the algorithm can be effectively trained


## 1 - Define the problem
Start by defining the problem you want to solve, the objectives. 

* What is the goal of your algorithm? 
* What kind of data will you be working with?
* What performance are we expecting ?
* Do I have data ? 

## 2 - Collect the data... The right data
Gather the data you will need to train and test your algorithm. This might involve:

* scraping data from the web
* collecting data from sensors
* working with pre-existing data sets

!!! danger

    In real-life application, the data itself is one thing on which you need to spend time :  
    
    * You need to ensure that the data describe your real-life problem 
    * You need to ensure that its quality is good
    * Are you able to produce the data you will use to train your model
    It doesn't make sense to have an algorithm that work well in laboratory condition, if you need to use it in 
    different condition


Let's take an example :
!!! example

    ##Assuming you want to train a classification model to classify eyes diseases, with a camera embedded on a raspberry.
    After some research on google, you'll find some already labeled dataset like this :
    
    ![EyeImage](https://www.kaggleusercontent.com/kf/61507157/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..rX7jZd9z39lSiyXG0mDVBQ.A9jkHT6G4f-_unDjNZPLrTxn4laI373iRkZKCKe3Aja3wZs2qp7lTAPazE3CWwlsXVj4bvfGStcJ_XyKB4-f7dKTxaFKz1Sr1GxXORk-K8zT3xoRZ0huHJFdDRcyU_-XCAfVkAl-rgjzx4_FWIt8X_7UviSzzZwEwx2OtHl11EmKPhSm3BGkgf4-6hIVhXV4JNn1w2JlpAlYlrSxGpqZOL_4MjNsiGDMmuhgHqECpdujCex59Ym_mCB0XpC76ulHZ2rYgeIrsRl-xXiwRUl8mt_Mt2OnwQBZnTg-94HcPZitziZ8uBXhxTDLbaraXQMmh7SB1vJnpjlJUrHT08jgpOCX6t6lscv4XJpQ1TDV0u_f6AHOh2SjkbJeOqI4LOmnN9wwhh9HM9n_W4aCqTZRq5ozBCXmd12VBMDehnlCK186cY1xDgWEJTgdrCB3sksD6Wuwcwgho18g5EFhtxnUDn1n-iyGaKzJcp8BNPWPu6nEv_mr-D0I0krOFm5FoZehFYu6SBKqfiuwseoTcND9CvwLQFrhPKzphjZNyB0N-HtaZytO2ZpX4n-cOFH63K4BZ-HoLX4vevzWwoqkERwGGI363O56VmcGrrXZrk6W4PsBvF5dj5XcY0pO6lqYbYx_igJRxfnPhTuVMO3XRUzDQZZ6fmXC9u8zvErWdwaobcARzxYDXI70SzcNWZZ-0a-dUdWsoSBnpZASqJEiJ1kJpQ.7FkCLL6hOw219IfZiH6FoQ/__results___files/__results___11_0.png)
    
    <br>
    <br>
    
    But with your camera, you're only able to produce image like this : 
    
    ![EyeDisease](https://www.ccteyes.com/wp-content/uploads/2019/12/whatiscataract.jpeg)
    
    It's completly out of scope, it doesn't make sense to train the algorithm on this dataset

    ##What can i do ?
    
    You have mainly two solutions : 

    * Take some pictures of eyes with diseases you want to classify
    * Scrap some data on internet and label it yourself


## 3 - Explore the data
Once you have your data, explore it to get a better understanding of what you're working with. 
This might involve data visualization and basic statistical analysis.

[See an example](../logistic-regression/#the-iris-dataset-example){ .md-button .md-button--primary }


## 4 - Preprocess the data
Preprocessing the data involves transforming and cleaning the data so that it can be used by the machine learning 
algorithm. This might involve feature engineering, data scaling, data normalization, data augmentation and data 
cleaning.

??? question "What is data normalization ?"

    Data normalization is the process of rescaling the values of numeric features in a dataset to a common scale. 
    This is typically done to prevent features with large values from dominating the analysis and to ensure that all 
    features have equal weight. One common technique for normalization is called `min-max scaling` 
    which involves scaling the values of a feature to a range between 0 and 1 by subtracting the minimum value of the 
    feature and dividing by the range of the feature.

??? question "What is data scaling ?"

    Data scaling is a similar technique to normalization that involves transforming the values of numeric features 
    so that they have a mean of 0 and a standard deviation of 1. This is typically done to ensure that all features 
    have the same variance and that the data is centered around 0. One common technique for scaling is called 
    "standardization," which involves subtracting the mean of the feature and dividing by its standard deviation.

??? question "What is data augmentation ?"
    
    Data augmentation is a technique used to increase the size of a training dataset by creating new,
    `synthetic` data points through transformations of existing data. The goal of data augmentation is to improve the 
    performance and generalization of machine learning models by exposing them to a wider range of variations in the data.
    
    Data augmentation can be applied to a wide range of data types, including images, audio, and text. For example, in image
    data augmentation, you can create new images by applying transformations such as rotation, translation, scaling, 
    flipping, and cropping to existing images. In audio data augmentation, you can apply transformations such as changing
    the pitch, tempo, or volume of existing audio recordings. In text data augmentation, you can create new text samples by
    replacing words with synonyms.
    
    Data augmentation is particularly useful in scenarios where the amount of available training data is limited.
    By creating new data points through data augmentation, you can effectively increase the size of the training set,
    which can help to prevent overfitting and improve the generalization of machine learning models.
    
    !!! example
    
        ![KittenAugmentation](https://cdn-images-1.medium.com/max/1000/1*C8hNiOqur4OJyEZmC7OnzQ.png)


??? question "What is data cleaning ?"

 
    Data cleaning is a critical step in the data preprocessing pipeline that involves identifying and correcting errors, 
    inconsistencies, and missing or irrelevant data in a dataset. The goal of data cleaning is to prepare the data for 
    analysis or machine learning by ensuring that it is accurate, complete, and consistent.

    ## Data cleaning typically involves a series of steps, which may include:

    ### Missing data handling
    If there are missing values in the dataset, data cleaning may involve imputing missing values 
    using techniques such as mean imputation, median imputation, or regression imputation.
    
    ### Outlier detection and handling
    Outliers are data points that fall far outside the normal range of values in the 
    dataset. Data cleaning may involve detecting and handling outliers using techniques such as box plots, z-scores, or clustering.
    
    ### Data formatting and type conversion
    Data cleaning may involve converting data types, such as converting categorical
    data to numerical data, or converting date formats to a standardized format.
    
    ### Removing duplicates
    If there are duplicate records in the dataset, data cleaning may involve identifying and removing
    them to avoid biases in the analysis.
    
    ### Data validation and verification
    Data cleaning may involve validating the data to ensure that it is accurate and 
    complete, such as cross-checking data against external sources or conducting manual spot-checks.

