# JackOrRose

This project is a logistic regression model built from scratch using only NumPy and pandas to predict whether a passenger survived the Titanic shipwreck. This project serves as a key learning exercise to build a deeper, more fundamental understanding of machine learning principles.



Why Build from Scratch?

While libraries like scikit-learn make building machine learning models simple, they often hide the core math and logic. By building this model from scratch, I gained a deep understanding of:



Vectorization: The importance of writing vectorized code for efficient computation.



Hypothesis Function: How to translate a mathematical formula into a working function.



Cost Function: How to measure the error in a classification model using cross-entropy loss.



Gradient Descent: The core optimization algorithm used to train the model.



Hyperparameter Tuning: The process of choosing an optimal learning rate and regularization parameter.



Project Workflow

Data Exploration and Cleaning: Performed Exploratory Data Analysis (EDA) on the Titanic dataset to understand the relationships between features and the target variable, Survived.



Feature Engineering: Created new features like isChild and isSF to better capture the non-linear relationships in the data.



Data Preprocessing: Handled missing values, encoded categorical variables, and standardized numerical features to prepare the data for the model.



Model Building: Implemented a regularized logistic regression model from scratch, including the hypothesis function, the cross-entropy cost function, and the gradient descent algorithm.



Model Evaluation: Used the trained model to make predictions on unseen test data, achieving an accuracy of 78.23% as per the Kaggle submission.

