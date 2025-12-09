# import important libraries
import sys
import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

# Make a dataframe of the file "insurance_claim.csv"

data_filename = 'insurance_claim.csv'
df = pd.read_csv(data_filename)

# Take a quick look of the data, notice that the response variable is binary

df.head()

# Assign age as the predictor variable 
x = df["age"].values

# Assign insuranceclaim as the response variable
y = df["insuranceclaim"].values

# Make a plot of the response (insuranceclaim) vs the predictor (age)
plt.plot(x,y,'o', markersize=7,color="#011DAD",label="Data")

# Add the labels for the axes
plt.xlabel("Age")
plt.ylabel("Insurance claim")

plt.xticks(np.arange(18, 80, 4.0))

# Label the value 1 as 'Yes' & 0 as 'No'
plt.yticks((0,1), labels=('No', 'Yes'))
plt.legend(loc='best')
plt.show()

### edTest(test_beta_guesstimate) ###

beta0 = -20

beta1 = 0.45

### edTest(test_beta_computation) ###
def logistic(x):
    return 1 / (1+np.exp(-(beta0 + beta1 * x)))
    
# P(y=1|x_i) for each x_i in x
probas = logistic(x)

# Get classification predictions

y_pred = (probas >=0.5).astype(int)

### edTest(test_acc) ###
# Use accuracy_score function to find the accuracy 

accuracy = accuracy_score(y, y_pred)

# Print the accuracy
print(accuracy)

# Make a plot similar to the one above along with the fit curve
plt.plot(x, y,'o', markersize=7,color="#011DAD",label="Data")

plt.plot(x,y_pred,linewidth=2,color='black',label="Classifications")
plt.plot(x,probas,linewidth=2,color='red',label="Probabilities")

plt.xticks(np.arange(18, 80, 4.0))
plt.xlabel("Age")
plt.ylabel("Insurance claim")
plt.yticks((0,1), labels=('No', 'Yes'))
plt.legend()
plt.show()

#  Post exercise question:
# In this exercise, you may have had to stumble around to find the right values of
# β0 and β1 to get accurate results.

# Although you may have used visual inspection to find a good fit, in most problems you would need a quantative method to measure the performance of your model. (Loss function)

# Which of the following below are NOT possible ways of quantifying the performance of the model.

# A. Compute the mean squared error loss of the predicted labels.
# B. Evaluate the log-likelihood for this Bernoulli response variable.
# C. Go the the temple of Apollo at Delphi, and ask the high priestess Pythia
# D. Compute the total number of misclassified labels.

### edTest(test_quiz) ###

# Put down your answers in a string format below (using quotes)

# for. eg, if you think the options are 'A' & 'B', input below as "A,B"

answer = 'C'

# A. MSE of the predicted labels:
# Labels are 0/1. If you compare true vs predicted labels and take MSE, you get something proportional to the misclassification rate (0 if correct, 1 if wrong). That is a valid quantitative performance measure (just a clunky way of writing 0–1 loss).

# B. Log-likelihood for a Bernoulli response
# This is literally the standard loss for logistic regression (negative log-likelihood / cross-entropy). Definitely a valid way to quantify performance.

# A. Total number of misclassified labels
# That’s also a valid performance metric: the 0–1 loss (or, divided by N, the error rate).