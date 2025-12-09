# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# "Advertising.csv" containts the data set used in this exercise
data_filename = 'Advertising.csv'

# Read the file "Advertising.csv" file using the pandas library
df = pd.read_csv(data_filename)

# Get a quick look of the data
df.head()

### edTest(test_pandas) ###

# Be aware that edTest comments at the top of some cells (Example: ### edTest(test_msg) ### ) 
# are important as they determine when the autograding tests are performed. 
# Altering these comments will result in the autograding tests not working properly.
# Do not move, edit, or remove these edTest comments. 


# Create a new dataframe by selecting the first 7 rows of
# the current dataframe
df_new = df.iloc[:7]

# Print your new dataframe to see if you have selected 7 rows correctly
print(df_new)

# Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(df_new['TV'], df_new['Sales'])

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel('TV Budget')
plt.ylabel('Sales')

# Add plot title 
plt.title('TV Budget vs Sales (First 7 Observations)')
plt.show()

# Your code here
df_new = df
print(df_new)
# Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(df_new['TV'], df_new['Sales'])

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel('TV Budget')
plt.ylabel('Sales')

# Add plot title 
plt.title('TV Budget vs Sales (All Observations)')
plt.show()
