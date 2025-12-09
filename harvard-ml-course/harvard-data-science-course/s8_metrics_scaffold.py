# Import necessary libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
# Your code here


# Read the datafile "covid_train.csv"
df_train = pd.read_csv("covid_train.csv")

# Take a quick look at the dataframe
df_train.head()


# Read the datafile "covid_test.csv"
df_test = pd.read_csv("covid_test.csv")

# Take a quick look at the dataframe
df_test.head()


# Get the train predictors
X_train = df_train.drop("Urgency", axis=1)

# Get the train response variable
y_train = df_train["Urgency"]


# Get the test predictors
X_test = df_test.drop("Urgency", axis=1)

# Get the test response variable
y_test = df_test["Urgency"]


### edTest(test_model) ###

# Define a kNN classification model with k = 7
knn_model = KNeighborsClassifier(n_neighbors=7)

# Fit the above model on the train data
knn_model.fit(X_train, y_train)

# Define a Logistic Regression model with max_iter as 10000 and C as 0.1 (leave all other parameters at default values)
log_model = LogisticRegression(max_iter=10000, C=0.1)

# Fit the Logistic Regression model on the train data
log_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)
log_pred = log_model.predict(X_test)

knn_cm = confusion_matrix(y_test, knn_pred)
log_cm = confusion_matrix(y_test, log_pred)

knn_TN, knn_FP, knn_FN, knn_TP = knn_cm.ravel()
log_TN, log_FP, log_FN, log_TP = log_cm.ravel()

knn_specificity = knn_TN / (knn_TN + knn_FP)
log_specificity = log_TN / (log_TN + log_FP)

knn_accuracy = accuracy_score(y_test, knn_pred)
log_accuracy = accuracy_score(y_test, log_pred)

knn_recall = recall_score(y_test, knn_pred)
log_recall = recall_score(y_test, log_pred)

knn_precision = precision_score(y_test, knn_pred)
log_precision = precision_score(y_test, log_pred)

knn_f1 = f1_score(y_test, knn_pred)
log_f1 = f1_score(y_test, log_pred)

metric_scores = {
    "Accuracy": [knn_accuracy, log_accuracy],
    "Recall": [knn_recall, log_recall],
    "Specificity": [knn_specificity, log_specificity],
    "Precision": [knn_precision, log_precision],
    "F1-score": [knn_f1, log_f1]
}

# Your code here

### edTest(test_metrics) ###

# Display your results
print(metric_scores)




