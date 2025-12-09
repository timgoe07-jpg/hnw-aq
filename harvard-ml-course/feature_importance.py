# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from helper2 import plot_permute_importance, plot_feature_importance


# Read the dataset "heart.csv"
df = pd.read_csv("heart.csv")

# Take a quick look at the data 
df.head()

# Assign the predictor and response variables.
# 'AHD' is the response and all the other columns are predictors
X = df.drop("AHD", axis=1)
y = df["AHD"]

# Set the model parameters
random_state = 44
max_depth = 5   # you can choose any depth; 5 is a good default

### edTest(test_decision_tree) ###

# Define a Decision Tree classifier
tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

# Fit the model on the entire data
tree.fit(X, y)

# Permutation importance for the decision tree
tree_result = permutation_importance(
    tree, X, y,
    n_repeats=10,
    random_state=random_state
)

### edTest(test_random_forest) ###

# Define a Random Forest classifier with 10 trees
forest = RandomForestClassifier(
    max_depth=max_depth,
    n_estimators=10,
    random_state=random_state
)

# Fit the model on the entire data
forest.fit(X, y)

# Permutation importance for the random forest
forest_result = permutation_importance(
    forest, X, y,
    n_repeats=10,
    random_state=random_state
)

# Helper code to visualize the MDI (Mean Decrease Impurity)
plot_feature_importance(tree, forest, X, y)

# Helper code to visualize permutation importance
plot_permute_importance(tree_result, forest_result, X, y)
