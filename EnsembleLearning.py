# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:51:41 2018

@author: Daniel
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold, cross_val_score

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def get_car_data():
    
    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "saftey", "condition"]
    
    # Read the data from the .csv file
    data_df = pd.read_csv("car_data.csv", header=None, names=col_names)

    # Convert categorical entries into numerical values
    cleanup_nums = {"buying": {"vhigh": 3., "high": 2., "med": 1., "low": 0.}, 
                    "maint": {"vhigh": 3., "high": 2., "med": 1., "low":0.}, 
                    "doors": {"2": 2., "3": 3., "4": 4., "5more": 5.}, 
                    "persons": {"2": 2., "4": 4., "more": 5.}, 
                    "lug_boot": {"small": 0., "med": 1., "big": 2.}, 
                    "saftey": {"low": 0., "med": 1., "high": 2.},
                    "condition": {"unacc": 0., "acc": 1., "good": 2., "vgood": 3.}}  
    
    data_df.replace(cleanup_nums, inplace=True)
    data_df.head()

    # Separate the data
    data_cols = col_names[0:6]
    target_col = ["condition"]
    
    # Convert the data into np arrays
    data = np.array(data_df[data_cols])
    targets = np.array(data_df[target_col])
 
    return data, targets

def get_autism_data():
    
    col_names = ["A1_score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", 
                 "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
                 "age", "gender", "ethnicity", "jundice", "autism", 
                 "country_of_res", "used_app_before", "result", "age_desc", 
                 "relation", "Class/ASD"]
    
    # Read the .csv containing the data
    data_df = pd.read_csv("Autism-Adult-Data.csv", header=None, names=col_names, na_values="?")

    # Convert the categorical columns into one-hots
    one_hot_cols = ["ethnicity", "country_of_res", "age_desc", "relation"]
    new_data_df = pd.get_dummies(data_df, columns=one_hot_cols)
    
    # Change the binary labels into 1s or 0s
    cleanup_nums = {"gender": {"f": 0., "m": 1.},
                    "jundice": {"no": 0., "yes": 1.},
                    "autism": {"no": 0., "yes": 1.},
                    "used_app_before": {"no": 0., "yes": 1.},
                    "Class/ASD": {"NO": 0., "YES": 1.}}
    new_data_df.replace(cleanup_nums, inplace=True)
    
    # Fill any missing values with the most frequent value in that column
    fill_NaN = Imputer(missing_values=np.nan, strategy='most_frequent', axis=1)
    imputed_df = pd.DataFrame(fill_NaN.fit_transform(new_data_df))
    imputed_df.columns = new_data_df.columns
    imputed_df.index = new_data_df.index

    # Split the data and the targets and numpy arrays
    data = np.array(imputed_df.loc[:, imputed_df.columns != "autism"])
    targets = np.array(imputed_df["autism"])
    
    return data, targets

def get_cancer_data():
    
    dataset = datasets.load_breast_cancer()

    # Obtain a normalizing scaler for scaling new data if added later
    std_scaler = preprocessing.StandardScaler().fit(dataset.data)
    
    # Normalize the data
    data = preprocessing.scale(dataset.data)
      
    return data, dataset.target

def get_accuracies(data, targets):
    
    print("Obtaining results for kNN...")      
    k_neighbors = 9
    classifier = KNeighborsClassifier(n_neighbors = k_neighbors)
    k_fold = KFold(len(np.ravel(targets)), n_folds=10, shuffle=True, random_state=18)
    accuracy_score = cross_val_score(classifier, data, np.ravel(targets), cv=k_fold, n_jobs=1).mean()
    kNNAcc = accuracy_score * 100.0
    
    print("Obtaining results for Decision Tree...")
    classifier = tree.DecisionTreeClassifier()
    k_fold = KFold(len(np.ravel(targets)), n_folds=10, shuffle=True, random_state=18)
    accuracy_score = cross_val_score(classifier, data, np.ravel(targets), cv=k_fold, n_jobs=1).mean()
    decisionTreeAcc = accuracy_score * 100.0
    
    print("Obtaining results for Neural Network...")
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)
    k_fold = KFold(len(np.ravel(targets)), n_folds=10, shuffle=True, random_state=18)
    accuracy_score = cross_val_score(classifier, data, np.ravel(targets), cv=k_fold, n_jobs=1).mean()
    neuralNetAcc = accuracy_score * 100.0
    
    print("Obtaining results for Bagging...")
    classifier = BaggingClassifier(KNeighborsClassifier(n_neighbors=9), max_samples=0.3, max_features=0.5, n_estimators=100)
    k_fold = KFold(len(np.ravel(targets)), n_folds=10, shuffle=True, random_state=18)
    accuracy_score = cross_val_score(classifier, data, np.ravel(targets), cv=k_fold, n_jobs=1).mean()
    baggingAcc = accuracy_score * 100.0
    
    print("Obtaining results for adaBoostAcc...")
    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
    k_fold = KFold(len(np.ravel(targets)), n_folds=10, shuffle=True, random_state=18)
    accuracy_score = cross_val_score(classifier, data, np.ravel(targets), cv=k_fold, n_jobs=1).mean()
    adaBoostAcc = accuracy_score * 100.0
     
    print("Obtaining results for randForestAcc...")
    classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=1, max_features=3)
    k_fold = KFold(len(np.ravel(targets)), n_folds=10, shuffle=True, random_state=18)
    accuracy_score = cross_val_score(classifier, data, np.ravel(targets), cv=k_fold, n_jobs=1).mean()
    randForestAcc = accuracy_score * 100.0
         
    return kNNAcc, decisionTreeAcc, neuralNetAcc, baggingAcc, adaBoostAcc, randForestAcc

def compare_accuracies(data, targets):
    print("--------------------------------------")
    
    kNNAcc, naiveBayesAcc, decisionTreeAcc, baggingAcc, adaBoostAcc, randForestAcc = get_accuracies(data, targets)
    
    print("kNN Accuracy: {}".format(kNNAcc))
    print("Bagging w/KNN Accuracy: {}".format(baggingAcc))
    print(" ")
    
    print("DecisionTree Accuracy: {}".format(naiveBayesAcc))
    print("AdaBoost w/d-Tree Accuracy: {}".format(adaBoostAcc))
    print(" ")
    
    print("NeuralNet Accuracy: {}".format(decisionTreeAcc))
    print("RandForest Accuracy: {}".format(randForestAcc))
    
    print("--------------------------------------")

def main():
    
    # Autism Data
    print("Autism Data")
    data, targets = get_autism_data()
    compare_accuracies(data, targets)

    # Car Data
    print("Car Data")
    data, targets = get_car_data()    
    compare_accuracies(data, targets)
    
    # Breast Cancer Data
    print("Breast Cancer Data")
    data, targets = get_cancer_data() 
    compare_accuracies(data, targets)
    
main()


