# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Read the dataset and take a quick look
df = pd.read_csv("diabetes.csv")
df.head()

# Assign the predictor and response variables.
# Outcome is the response and all the other columns are the predictors
X = df.drop("Outcome", axis=1)
y = df['Outcome']

# Set the seed for reproducibility of results
seed = 0

# Split the data into train and test sets with the mentioned seed
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=seed)

### edTest(test_vanilla) ### 

# Define a Random Forest classifier with randon_state = seed
vanilla_rf = RandomForestClassifier(random_state=seed)

# Fit the model on the train data
vanilla_rf.fit(X_train, y_train);

# Calculate AUC/ROC on the test set
y_proba = vanilla_rf.predict_proba(X_test)[:, 1]
auc = np.round(roc_auc_score(y_test, y_proba),2)
print(f'Plain RF AUC on test set:{auc}')

# Number of samples and features
num_features = X_train.shape[1]
num_samples = X_train.shape[0]
num_samples, num_features

from collections import OrderedDict
clf = RandomForestClassifier(warm_start=True, 
                               oob_score=True,
                               min_samples_leaf=40,
                               max_depth = 10,
                               random_state=seed)

error_rate = {}

# Range of `n_estimators` values to explore.
min_estimators = 150
max_estimators = 500

for i in range(min_estimators, max_estimators + 1):
    clf.set_params(n_estimators=i) 
    clf.fit(X_train.values, y_train.values)

    # Record the OOB error for each `n_estimators=i` setting.
    oob_error = 1 - clf.oob_score_
    error_rate[i] = oob_error
    

# Generate the "OOB error rate" vs. "n_estimators" plot.
# OOB error rate = num_missclassified/total observations (%)\
xs = []
ys = []
for label, clf_err in error_rate.items():
    xs.append(label)
    ys.append(clf_err)   
plt.plot(xs, ys)
plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.show();

from collections import OrderedDict
ensemble_clfs = [
    (1,
        RandomForestClassifier(warm_start=True, 
                               min_samples_leaf=1,
                               oob_score=True,
                               max_depth = 10,
                               random_state=seed)),
    (5,
        RandomForestClassifier(warm_start=True, 
                               min_samples_leaf=5,
                               oob_score=True,
                               max_depth = 10,
                               random_state=seed))
]

# Map a label (the value of `min_samples_leaf`) to a list of (model, oob error) tuples.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 80
max_estimators = 500

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i) 
        clf.fit(X_train.values, y_train.values)

        # Record the OOB error for each model. Error is 1 - oob_score
        # oob_score: score of the training dataset obtained using an 
        # out-of-bag estimate.
        # OOB error rate is % of num_missclassified/total observations
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=f'min_samples_leaf={label}')

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show();

err = 100
best_num_estimators = 0
for label, clf_err in error_rate.items():
    num_estimators, error = min(clf_err, key=lambda n: (n[1], -n[0]))
    if error<err: err=error; best_num_estimators = num_estimators; best_leaf = label

print(f'Optimum num of estimators: {best_num_estimators} \nmin_samples_leaf: {best_leaf}')

### edTest(test_estimators) ### 
estimators_rf = RandomForestClassifier(n_estimators= best_num_estimators,
                                    random_state=seed,
                                    oob_score=True,
                                    min_samples_leaf=best_leaf,
                                    max_features='sqrt') 

# Fit the model on the train data
estimators_rf.fit(X_train, y_train);

# Calculate AUC/ROC on the test set
y_proba = estimators_rf.predict_proba(X_test)[:, 1]
estimators_auc = np.round(roc_auc_score(y_test, y_proba),2)
print(f'Educated RF AUC on test set:{estimators_auc}')

estimators_rf.get_params()

from sklearn.model_selection import GridSearchCV

do_grid_search = True

if do_grid_search:
    rf = RandomForestClassifier(n_jobs=-1,
                               n_estimators= best_num_estimators,
                               oob_score=True,
                               max_features = 'sqrt',
                               min_samples_leaf=best_leaf,
                               random_state=seed)

    param_grid = {
        'min_samples_split': [2,5]}
    
    scoring = {'AUC': 'roc_auc'}
    
    grid_search = GridSearchCV(rf, 
                               param_grid, 
                               scoring=scoring, 
                               refit='AUC', 
                               return_train_score=True, 
                               n_jobs=-1)
    
    results = grid_search.fit(X_train, y_train)
    print(results.best_estimator_.get_params())
    best_rf = results.best_estimator_
    # Calculate AUC/ROC
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    auc = np.round(roc_auc_score(y_test, y_proba),2)
    print(f'GridSearchCV RF AUC on test set:{auc}')
