import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Advertising.csv')
df.head()

X = df.drop('Sales', axis=1)
y = df.Sales.values

lm = LinearRegression().fit(X,y)

# you can learn more about Python format strings here:
# https://docs.python.org/3/tutorial/inputoutput.html
print(f'{"Model Coefficients":>9}')
for col, coef in zip(X.columns, lm.coef_):
    print(f'{col:>9}: {coef:>6.3f}')
print(f'\nR^2: {lm.score(X,y):.4}')

# From info on this kind of assignment statement see:
# https://python-reference.readthedocs.io/en/latest/docs/operators/multiplication_assignment.html
df *= 1000
df.head()

# refit a new regression model on the scaled data
X = df.drop('Sales', axis=1)
y = df.Sales.values
lm = LinearRegression().fit(X,y)

print(f'{"Model Coefficients":>9}')
for col, coef in zip(X.columns, lm.coef_):
    print(f'{col:>9}: {coef:>6.3f}')
print(f'\nR^2: {lm.score(X,y):.4}')

plt.figure(figsize=(8,3))
# column names to be displayed on the y-axis
cols = X.columns
# coeffient values from our fitted model (the intercept is not included)
coefs = lm.coef_
# create the horizontal barplot
plt.barh(cols, coefs)
# dotted, semi-transparent, black vertical line at zero
plt.axvline(0, c='k', ls='--', alpha=0.5)
# always label your axes
plt.ylabel('Predictor')
plt.xlabel('Coefficient Values')
# and create an informative title
plt.title('Coefficients of Linear Model Predicting Sales\n from Newspaper, '\
            'Radio, and TV Advertising Budgets (in Dollars)');

# Q1: Based on the plot above, advertising in which type of media has the largest effect on Sales?

# A. Newspaper
#B. Radio
#. TV

### edTest(test_Q1) ###
# your answer here
Q1_ANSWER = 'B' 

# Radio model coefficient is the highest
# increasing TV advertising budget leads to highest sales increase

# Q2: If the newspaper advertising budget were higher, what difference might we expect to see in sales?

# A. There would be no change
# B. Sales would increase, but only slightly
# C. Sales would decrease, but only slightly\

### edTest(test_Q2) ###
# your answer here
Q2_ANSWER = 'C'

# newspaper coefficient is very close to Zero 
# increasing newspaper has almost no influence on sales
# sales would decrease (slightly negative coefficient)

# But what happens when our predictors are not all on the same scale?
#To find out, we'll change the units of the 3 budgets by converting them into different currencies. Use the following conversion rates for this exercise:

# # create a new DataFrame to store the converted budgets
X2 = pd.DataFrame()
X2['TV (Rupee)'] = 200 * df['TV'] # convert to Sri Lankan Rupee
X2['Radio (Won)'] = 1175 * df['Radio'] # convert to South Korean Won
X2['Newspaper (Cedi)'] = 6 * df['Newspaper'] # Convert to Ghanaian Cedi

# we can use our original y as we have not converted the units for Sales
lm2 = LinearRegression().fit(X2,y)

print(f'{"Model Coefficients":>16}')
for col, coef in zip(X2.columns, lm2.coef_):
    print(f'{col:>16}: {coef:>8.5f}')
print(f'\nR^2: {lm2.score(X2,y):.4}')

# This time, scaling our predictors but not the response clearly caused a change in our coefficients. Thinking about this question may help you us appreciate why.

# Q3: Assume that a $1 increase in the `Radio` budget is associated with an increase in `Sales` of $25. Then a 1 Won increase in the Radio budget would see Sales increase by ___ dollars.

# Hint: 1,175 Won = $1

### edTest(test_Q3) ###
# in the oroginal (properly scaled) model:
# $1 increase in radio budget -> $25 increase in sales
# 1,175 Korean Won = $1
# 1 Won = $1 / 1175
# effect of 1 won = 25/1175 = 0.0212766
Q3_ANSWER = 25/1175

# Q4: How did your answer in Q3 compare to the original hypothetical increase of $25?

# A. It was higher
# B. It was lower
# C. No change (scale invariant)

### edTest(test_Q4) ###

### edTest(test_Q4) ###
# your answer here
Q4_ANSWER = 'B'
# won is smaller unit of currency than a dollar
# increasing budget by 1 won is a tiny increase
# effect on sales of won is also tiny
# compared to $25:
# 0.0213 dollars < 25 dollars
# lower effect

plt.figure(figsize=(8,3))
plt.barh(X2.columns, lm2.coef_)
plt.axvline(0, c='k', ls='--', alpha=0.5)
plt.ylabel('Predictor')
plt.xlabel('Coefficient Values')
plt.title('Coefficients of Linear Model Predicting Sales\n from Newspaper, '\
            'Radio, and TV Advertising Budgets (Different Currencies)');

# Q5: Based on the plot above, which advertising in which type of media has the least effect on sales?

# A. Newspaper
# B. Radio
# C. TV

### edTest(test_Q5) ###
# magnitude of coefficient for radio is the lowest
Q5_ANSWER = 'B'

# Q6: True or False: This is the same interpretation we had in our original model where all budgets were in dollars.

### edTest(test_Q6) ###
# Use the boolean values True or False
# features were converted to different currencies
Q6_ANSWER = False

# Q7: Imagine we have a 3rd regression models whose budgets have again been converted to 3 different currencies.
# True or False: we can compare the 2nd and 3rd models' MSE losses to determine which model's coefficients provide a more accurate interpretation of what type of media advertising has the largest effect on Sales.

### edTest(test_Q7) ###
# Use the boolean values True or False
# MSE only measures prediction accuracy
# it does not tell us:
# whether coefficients are interpretable
# whether coefficients reflect true feature importance
# whether predictors are on a comparable scale
# scaling features changes cofficients but not the underlying relationship
# coefficiient interpretation requires comparable scales, not low MSE
Q7_ANSWER = False

# Finally, it's important to recognize the limits of the x-axis differ between the two bar plots we've seen so far. We can better appreciate this difference by ploting both with a shared x-axis.


fig, axes = plt.subplots(2,1, figsize=(8,6), sharex=True)

axes[0].barh(X.columns, lm.coef_)
axes[0].set_title('Dollars');
axes[1].barh(X2.columns, lm2.coef_)
axes[1].set_title('Different Currencies')
for ax in axes:
    ax.axvline(0, c='k', ls='--', alpha=0.5)
axes[0].set_ylabel('Predictor')
axes[1].set_xlabel('Coefficient Values');

#We've seen that having our predictors on different scales can bias our interpretation of the coefficients. In a future notebook we'll look at one way of insuring that all our predictors are on the same scale.