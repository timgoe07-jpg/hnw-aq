# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
except ImportError:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# Code to read the dataset and take a quick look
df = pd.read_csv("diabetes.csv")
df.head()

# Investigate the response variable for data imbalance
count0, count1 = df['Outcome'].value_counts()
print(f'The percentage of diabetics in the dataset is only {100*count1/(count0+count1):.2f}%')

# Assign the predictor and response variables
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# Fix a random_state
random_state = 22

# Split the data into train and validation sets 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, random_state=random_state
)

# Set tree depth
max_depth = 20

# ------------------------------
# 1️⃣ Vanilla Random Forest
# ------------------------------
random_forest = RandomForestClassifier(
    random_state=random_state,
    max_depth=max_depth,
    n_estimators=10
)

random_forest.fit(X_train, y_train)

### edTest(test_vanilla) ### 
predictions = random_forest.predict(X_val)

# Metrics
f_score = f1_score(y_val, predictions)
score1 = round(f_score, 2)

auc_score = roc_auc_score(y_val, predictions)
auc1 = round(auc_score, 2)

# ------------------------------
# 2️⃣ Balanced Random Forest (class_weight)
# ------------------------------
random_forest = RandomForestClassifier(
    random_state=random_state,
    max_depth=max_depth,
    n_estimators=10,
    class_weight="balanced_subsample"
)

random_forest.fit(X_train, y_train)

### edTest(test_balanced) ###
predictions = random_forest.predict(X_val)

f_score = f1_score(y_val, predictions)
score2 = round(f_score, 2)

auc_score = roc_auc_score(y_val, predictions)
auc2 = round(auc_score, 2)

# ------------------------------
# 3️⃣ Upsampling using SMOTE
# ------------------------------
sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

random_forest = RandomForestClassifier(
    random_state=random_state,
    max_depth=max_depth,
    n_estimators=10,
    class_weight="balanced_subsample"
)

random_forest.fit(X_train_res, y_train_res)

### edTest(test_upsample) ###
predictions = random_forest.predict(X_val)

f_score = f1_score(y_val, predictions)
score3 = round(f_score, 2)

auc_score = roc_auc_score(y_val, predictions)
auc3 = round(auc_score, 2)

# ------------------------------
# 4️⃣ Downsampling using RandomUnderSampler
# ------------------------------
rs = RandomUnderSampler(random_state=2)

X_train_res, y_train_res = rs.fit_resample(X_train, y_train)

random_forest = RandomForestClassifier(
    random_state=random_state,
    max_depth=max_depth,
    n_estimators=10,
    class_weight="balanced_subsample"
)

random_forest.fit(X_train_res, y_train_res)

### edTest(test_downsample) ###
predictions = random_forest.predict(X_val)

f_score = f1_score(y_val, predictions)
score4 = round(f_score, 2)

auc_score = roc_auc_score(y_val, predictions)
auc4 = round(auc_score, 2)

# ------------------------------
# Summary Table
# ------------------------------
pt = PrettyTable()
pt.field_names = ["Strategy","F1 Score","AUC score"]
pt.add_row(["Random Forest - No imbalance correction",score1,auc1])
pt.add_row(["Random Forest - balanced_subsamples",score2,auc2])
pt.add_row(["Random Forest - Upsampling",score3,auc3])
pt.add_row(["Random Forest - Downsampling",score4,auc4])
print(pt)

# ------------------------------
# Final Multiple Choice
# ------------------------------
### edTest(test_chow1) ###
# Which metric is NOT recommended for imbalanced data?
# → Accuracy is misleading when classes are imbalanced
answer1 = 'D'
