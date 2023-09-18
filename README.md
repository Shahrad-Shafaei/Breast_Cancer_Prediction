# Breast Cancer Diagnostics 
## by (Valentine Ezenwanne)

## Project Objective
- To build a machine learning model that will be used to predict whether the breast cancer cell is benign or malignant.

- The model will take features from a digitized image of a fine aspirate of a breast cancer cell and predict whether the cell is benign or malignant with more 95% accuracy score


## Dataset

The dataset contains Features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 
They describe characteristics of the cell nuclei present in the image.


The dataset can be found in the repository https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data


## Tools used
- Data cleaning: Pandas
- Data Visualization: Matplotlib and seaborn
- Scikit library
- Numpy


## Project Steps

The steps taken for the project include:
1.	Data collection
2.	Data Wrangling
3.	Data Exploration
4.	Data Preprocessing
5.	Model Building
6.	Model Evaluation
7.	Hyperparameter Tuning
8.	Choosing the Best Model



# Summary of findings

- From the baseline model evaluation, there are four models which had accuracy score above 95%, 
1) **Support Vector Machine (98.2%)** 
2) **LogisticRegression (97.4%)** 
3) **Naive Bayes(96.5%)**
4) **RandomForest(95.6%)** 

- From the cross validation, four models did well above 95%
1) **Support Vector Machine (97.58%)** 
2) **LogisticRegression (97.36%)** 
3) **KNN(96.04%)**
4) **RandomForest(95.6%)** 

Hyperparameter Tuning
From performung hyperparameter tuning using GridSearchCV, these three models **Logistic Regression**, **SVM**, **Random Forest Classifier** performs better with more than **97%** best scores

Evaluation with best parameter
The top three model, Support Vector Machine, Random Forest and Logistic Regression were evaluated with there best parameter and **Support Vector Classifier** has the highest accuracy of 98.25% and is the best model to used for the classification 

