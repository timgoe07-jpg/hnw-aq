# Import necessary libraries
import itertools
import numpy as np
import pandas as pd 
from numpy import std
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Read the dataset
df = pd.read_csv("airquality.csv",index_col=0)

# Take a quick look at the data
df.head(10)


# Use the column Ozone to drop any NaNs from the dataframe
df = df[df.Ozone.notna()]


# Assign the values of Ozon column as the predictor variable
x = df[['Ozone']].values

# Use temperature as the response data
y = df['Temp']


# Split the data into train and test sets with train size as 0.8 
# and set random_state as 102
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=102)


# Specify the number of bootstraps as 30
num_bootstraps = 30

# Specify the maximum depth of the decision tree as 3
max_depth = 3

# Define the Bagging Regressor Model
# Use Decision Tree as your base estimator with depth as mentioned in max_depth
# Initialise number of estimators using the num_bootstraps value
model = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=max_depth),
    n_estimators=num_bootstraps,
    random_state=102
)
                        

# Fit the model on the train data
model.fit(x_train, y_train)


# Helper code to plot the predictions of individual estimators 
plt.figure(figsize=(10,8))

xrange = np.linspace(x.min(),x.max(),80).reshape(-1,1)
plt.plot(x_train,y_train,'o',color='#CC1100', markersize=6, label="Train Data")
plt.plot(x_test,y_test,'o',color='#241571', markersize=6, label="Test Data")
plt.xlim()
for i in model.estimators_:
    y_pred1 = i.predict(xrange)
    plt.plot(xrange,y_pred1,alpha=0.5,linewidth=0.5,color = '#ABCCE3')
plt.plot(xrange,y_pred1,alpha=0.6,linewidth=1,color = '#ABCCE3',label="Prediction of Individual Estimators")


y_pred = model.predict(xrange)
plt.plot(xrange,y_pred,alpha=0.7,linewidth=3,color='#50AEA4', label='Model Prediction')
plt.xlabel("Ozone", fontsize=16)
plt.ylabel("Temperature", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='best',fontsize=12)
plt.show();


# Compute the test MSE of the prediction of first estimator
y_pred1 = model.estimators_[0].predict(x_test)

# Print the test MSE 
print("The test MSE of one estimator in the model is", round(mean_squared_error(y_test,y_pred1),2))


### edTest(test_mse) ###
# Compute the test MSE of the model prediction
y_pred = model.predict(x_test)

# Print the test MSE 
print("The test MSE of the model is",round(mean_squared_error(y_test,y_pred),2))


