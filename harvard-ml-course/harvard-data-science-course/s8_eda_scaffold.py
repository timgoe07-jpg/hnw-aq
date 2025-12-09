# Import necessary libraries

# Your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


# Read the datafile "covid.csv"
df = pd.read_csv("covid.csv")

# Take a quick look at the dataframe
df.head()


# Check if there are any missing or Null values
df.isnull().sum()


### edTest(test_na) ###

# Find the number of rows with missing values
num_null = df.isnull().any(axis=1).sum()
print("Number of rows with null values:", num_null)


# kNN impute the missing data
# Use a k value of 5

# Your code here
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(df.drop('Urgency', axis=1))


### edTest(test_impute) ###
# Replace the original dataframe with the imputed data, continue to use df for the dataframe

# Your code here
df[df.columns[:-1]] = X_imputed

# Plot an appropriate graph to answer the following question
# Your code here
plt.figure(figsize=(7,5))

df_age = df.copy()
df_age['age_group'] = pd.cut(
    df_age['age'], 
    bins=[0,20,30,40,50,60,70,100],
    labels=['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+'])

urgent_age = df_age[df_age['Urgency']==1]['age_group'].value_counts().sort_index()
urgent_age.plot(kind='bar', color='orange')
plt.xlabel("Age Group")
plt.ylabel("Count of Urgent Cases")
plt.title("Urgent Admissions by Age Group")
plt.show()

# ⏸ Which age group has the most urgent need for a hospital bed?
# A. 60 - 70
#B. 50 - 60
#C. 20 - 30
#D. 40 - 50

### edTest(test_chow1) ###
# Submit an answer choice as a string below (eg. if you choose option A, put 'A')
answer1 = 'D'

# Plot an appropriate graph to answer the following question    
# Your code here
symptoms = ['cough', 'fever', 'chills', 'sore_throat', 'headache', 'fatigue']
urgent_symptoms = df[df['Urgency']==1][symptoms].sum()

urgent_symptoms.plot(kind='bar', color='green')
plt.xlabel("Symptom")
plt.ylabel("Count")
plt.title("Most Common Symptom in Urgent Cases")
plt.show()



### edTest(test_chow2) ###
# Submit an answer choice as a string below (eg. if you choose option A, put 'A')

# ⏸ Among the following symptoms, which is the most common one for patients with urgent need of hospitalization?
#A. Cough
#B. Fever
#C. Sore Throat
#D. Fatigue

answer2 = 'B'


# Plot an appropriate graph to answer the following question    
# Your code here
plt.figure(figsize=(7,5))
counts = pd.DataFrame({
    'Urgent': df[df['Urgency']==1]['cough'].sum(),
    'Not Urgent': df[df['Urgency']==0]['cough'].sum()
}, index=['Cough'])

counts.T.plot(kind='bar', color=['red', 'blue'])
plt.ylabel("Count")
plt.title("Cough Comparison: Urgent vs Not Urgent")
plt.show()

#⏸ As compared to patients with urgent need of hospitalization, patients with no urgency have cough as a more common symptom?
#A. True
#B. False
#C. It is the same
#D. Cannot say

### edTest(test_chow3) ###
# Submit an answer choice as a string below (eg. if you choose option A, put 'A')
answer3 = 'A'

### edTest(test_split) ###
# Split the data into train and test sets with 70% for training
# Use random state of 60 and set of data as the train split

# Your code here
df_train, df_test = train_test_split(df, test_size=0.3, random_state=60)

# Save the train data into a csv called "covid_train.csv"
# Remember to not include the default indices
df_train.to_csv("covid_train.csv", index=False)

# Save the test data into a csv called "covid_test.csv"
# Remember to not include the default indices
df_train.to_csv("covid_test.csv", index=False)



