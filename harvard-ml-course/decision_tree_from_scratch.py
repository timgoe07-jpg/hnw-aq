# Import necessary libraries
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Read the dataframe in
tree_df = pd.read_csv('two_classes.csv')
# Inspect the top 5 rows
tree_df.head()

# Plot the data to visualize class patterns

# Create figure of specific size and the axes objects
fig, ax = plt.subplots(figsize=(6,6))

# Scatter the data points
# Other colormaps could be found here:
# https://stackoverflow.com/questions/34314356/how-to-view-all-colormaps-available-in-matplotlib
scatter = ax.scatter(tree_df['x1'], tree_df['x2'], c=tree_df['y'], cmap='rainbow')

# Create a legend object with title "Classes" and specified location on the plot
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.add_artist(legend)

# Add labeling to the plot
ax.set_xlabel('x1', fontsize='14')
ax.set_ylabel('x2', fontsize='14')
ax.set_title('Synthetic data with two classes', fontsize='15')

# Display the plot
plt.show()

# Function that calculates the total Gini impurity index for each value provided
def get_total_gini(predictor_values, pred_name, df):
    '''
    Parameters: an array of _unique_ predictor values,
                a name of the predictor (String),
                a corresponding dataframe object.
    Returns: an array of total Gini index for each provided predictor value.
    '''
    total_gini = []
    
    # try each value as a potential split location
    for val in predictor_values:
    
        # Left Leaf
        # counts of each class (1 or 0) to the left of candidate split
        left_1 = df[(df[pred_name] <= val) & (df['y']==1)].shape[0]
        left_0 = df[(df[pred_name] <= val) & (df['y']==0)].shape[0]
        
        # total number of points on the left
        # (max with small number to avoid devision by zero later)
        N_left_raw = left_1 + left_0
        N_left = max(1e-5, N_left_raw)

        # Gini impurity for the left leaf
        p_left_1 = left_1 / N_left
        p_left_0 = left_0 / N_left
        gini_left = 1 - (p_left_1**2 + p_left_0**2)
    
        # Right Leaf
        # counts of each class (1 or 0) to the right of candidate split
        right_1 = df[(df[pred_name] > val) & (df['y'] == 1)].shape[0]
        right_0 = df[(df[pred_name] > val) & (df['y'] == 0)].shape[0]

        # total number of points on the right
        # (max with small number to avoid division by zero later)
        N_right_raw = right_1 + right_0
        N_right = max(1e-5, N_right_raw)
        # Gini impurity for the right leaf
        p_right_1 = right_1 / N_right
        p_right_0 = right_0 / N_right
        gini_right = 1 - (p_right_1**2 + p_right_0**2)

        # total number of points in both leaves
        N_total = N_left_raw + N_right_raw
        # total weighted gini impurity index
        total_gini.append((N_left_raw / N_total)* gini_left + (N_right_raw / N_total)* gini_right)
    
    return total_gini

### edTest(test_x1_unique_values) ###

# Get unique values of x1
x1_unique = np.unique(tree_df['x1'].values)

### edTest(test_x2_unique_values) ###

# Get unique values of x2
x2_unique = np.unique(tree_df['x2'].values)

### edTest(test_x1_total_gini) ###

# get total Gini index for x1
x1_total_gini = get_total_gini(x1_unique, 'x1', tree_df)

### edTest(test_x2_total_gini) ###

# get total Gini index for x2
x2_total_gini = get_total_gini(x2_unique, 'x2', tree_df)


# plot the resulting total Gini indexes for each predictor
plt.plot(x1_unique, x1_total_gini, 'r+', label="$x_1$")
plt.plot(x2_unique, x2_total_gini, 'b+', label="$x_2$")
plt.xlabel("Predictor values", fontsize="13")
plt.ylabel("Total Gini index", fontsize="13")
plt.title("Total gini indexes for x1 and x2 predictors - first split", fontsize="14")
plt.legend()
plt.show()

def report_splits(x1_unique, x2_unique, x1_total_gini, x2_total_gini):
     # predictor providing the split with the lowest gini (x1=0, x2=1)
     best_pred = np.argmin([min(x1_total_gini), min(x2_total_gini)])
     # index of lowest gini score for best predictor 
     best_gini_idx = np.argmin([x1_total_gini, x2_total_gini][best_pred])
     # best predictor value corresponding to that lowest gini score
     best_pred_value = [x1_unique, x2_unique][best_pred][best_gini_idx]
     print("The lowest total gini score is achieved when splitting on "+\
          f"{['x1','x2'][best_pred]} at the value {best_pred_value}.")
          
report_splits(x1_unique, x2_unique, x1_total_gini, x2_total_gini)

### edTest(test_first_threshold) ###

# Find the threshold to split on
def get_threshold(unique_values, gini_scores):
    # index of lowest gini score
    idx = np.argmin(gini_scores)
    # threshold is the midpoint between
    # the predictor value resulting in lowest gini split
    # and the next highest unique predictor value
    # (you can assume values are sorted now and during gini calculation)
    threshold = (unique_values[idx] + unique_values[idx + 1]) / 2
    return threshold

x2_threshold = get_threshold(x2_unique, x2_total_gini)
print(f"Our threshold will be {x2_threshold}")

def get_split_labels(splits):
    '''
    Parameters:
        splits: List of ('predictor name', threshold) tuples
                Ex: [('x1', 42), ('x2', 109)]
    Returns: List of dictionaries, one for each split. 
             Dictionaries contain class labels for each side of the split
    '''
    split_labels = []
    region = tree_df
    for pred, thresh in splits:
        region_labels = {
            'left': region.loc[region[pred] <= thresh, 'y'].mode().values[0],
            'right': region.loc[region[pred] > thresh, 'y'].mode().values[0]
        }
        split_labels.append(region_labels)
        region = region[region[pred] <= thresh]
    return split_labels

# Example of how get_split_labels works
splits = [('x2', x2_threshold)]
split_labels = get_split_labels(splits)
print('class labels for the children of the root node:', split_labels)

def predict_class(x1, x2, splits):
    # get split labels to use when predicting
    split_labels = get_split_labels(splits)
    y_hats = []
    # iterate over each data point
    for x1_i, x2_i in zip(x1.ravel(), x2.ravel()):
        # dict lets us specify a predictor based on split rule
        obs = {'x1': x1_i, 'x2': x2_i}
        # apply each split rule
        for n_split, (pred, thresh) in enumerate(splits):
            # left
            if obs[pred] <= thresh:
                if n_split == len(splits)-1:
                    y_hats.append(split_labels[n_split]['left'])
            # right
            else:
                y_hats.append(split_labels[n_split]['right'])
                break
    return np.array(y_hats)

# Plot the split:
# Create figure of specific size and the axes objects
fig, ax = plt.subplots(figsize=(6, 6))

# Scatter the data points
# Other colormaps could be found here:
# https://stackoverflow.com/questions/34314356/how-to-view-all-colormaps-available-in-matplotlib
scatter = ax.scatter(tree_df['x1'], tree_df['x2'], c=tree_df['y'], cmap='rainbow')

# Create a legend object with title "Classes" and specified location on the plot
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.add_artist(legend)

# Split line
ax.hlines(x2_threshold, xmin=0, xmax=500,
          color ='black', lw = 2, ls=':', label='new split')
ax.legend()

# Add labeling to the plot
ax.set_xlabel('x1', fontsize='14')
ax.set_ylabel('x2', fontsize='14')
ax.set_title('The first split of the data', fontsize='16')

# Plot decision regions
# dummy x1 & x2 grid values to predict on
eps = 5 # padding for the grid
xx1, xx2 = np.meshgrid(np.arange(tree_df['x1'].min()-eps, tree_df['x1'].max()+eps, 1),
                       np.arange(tree_df['x2'].min()-eps, tree_df['x2'].max()+eps, 1))
# predict class labels on grid points
class_pred = predict_class(xx1, xx2, splits)
# contour plot for decision regions
plt.contourf(xx1, xx2, class_pred.reshape(xx1.shape), alpha=0.2, zorder=-1, cmap=plt.cm.coolwarm)

# Display the plot
plt.show()

first_split_df  = tree_df[tree_df['x2'] <= x2_threshold]
# Peak at the result
first_split_df.head()

# provide unique values from our new dataframe
x1_unique_2split = np.unique(first_split_df['x1'].values)
x2_unique_2split = np.unique(first_split_df['x2'].values)

tot_gini_x1_2split = get_total_gini(x1_unique_2split, 'x1', first_split_df)
tot_gini_x2_2split = get_total_gini(x2_unique_2split, 'x2', first_split_df)

plt.plot(x1_unique_2split, tot_gini_x1_2split, 'r+', label='$x_1$')
plt.plot(x2_unique_2split, tot_gini_x2_2split, 'b+', label='$x_2$')
plt.xlabel("Predictor values", fontsize="13")
plt.ylabel("Total Gini index", fontsize="13")
plt.title("Total Gini indexes for x1 and x2 predictors - second split", fontsize="14")
plt.legend()
plt.show()

report_splits(x1_unique_2split, x2_unique_2split,
              tot_gini_x1_2split, tot_gini_x2_2split)

### edTest(test_second_threshold) ###

# The value to split on
# Hint: be sure to use the variables defined 
# in the 2 cells above!
x1_threshold_2split = get_threshold(x1_unique_2split,tot_gini_x1_2split)
x1_threshold_2split

# Plot the split:
# Create figure of specific size and the axes objects
fig, ax = plt.subplots(figsize=(6, 6))

# Scatter the data points
scatter = ax.scatter(tree_df['x1'], tree_df['x2'], c=tree_df['y'], cmap='rainbow')

# Create a legend object with title "Classes" and specified location on the plot
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.add_artist(legend)

# Split lines
ax.hlines(x2_threshold, xmin=0, xmax=500,
          color ='black', lw = 0.5, ls='--')
ax.vlines(x1_threshold_2split, ymin=0, ymax=x2_threshold,
          color ='black', lw = 2, ls=':', label='new split')
ax.legend()

# Plot decision regions
# predict class labels on grid points
class_pred = predict_class(xx1, xx2, [('x2', x2_threshold), ('x1', x1_threshold_2split)])
# contour plot for decision regions
plt.contourf(xx1, xx2, class_pred.reshape(xx1.shape), alpha=0.2, zorder=-1, cmap=plt.cm.coolwarm)

# Add labeling to the plot
ax.set_xlabel('x1', fontsize='13')
ax.set_ylabel('x2', fontsize='13')
ax.set_title('First and second splits on the data', fontsize='14')

# Display the plot
plt.show()

first_right_intern_node_df  = first_split_df[first_split_df['x1'] > x1_threshold_2split]
first_right_intern_node_df.head()

first_right_intern_node_df['y'].value_counts()

second_split_left_df  = first_split_df[first_split_df['x1'] <= x1_threshold_2split]

second_split_left_df['y'].value_counts()

# Find the predictor and threshold for the split

# get unique values of x1 and x2 in this region
x1_unique_3split = np.unique(second_split_left_df['x1'].values)
x2_unique_3split = np.unique(second_split_left_df['x2'].values)

# Get total gini index for each predictor
tot_gini_x1_3split = get_total_gini(x1_unique_3split, 'x1', second_split_left_df)
tot_gini_x2_3split = get_total_gini(x2_unique_3split, 'x2', second_split_left_df)

plt.plot(x1_unique_3split, tot_gini_x1_3split, 'r+', label='$x_1$')
plt.plot(x2_unique_3split, tot_gini_x2_3split, 'b+', label='$x_2$')
plt.xlabel("Predictor values", fontsize="13")
plt.ylabel("Total Gini index", fontsize="13")
plt.title("Total Gini indexes for x1 and x2 predictors - third split", fontsize="14")
plt.legend()
plt.show()

report_splits(x1_unique_3split, x2_unique_3split,
              tot_gini_x1_3split, tot_gini_x2_3split)

### edTest(test_third_threshold) ###

# The value to split on
x2_threshold_3split = get_threshold(x2_unique_3split, tot_gini_x2_3split)
x2_threshold_3split

# Plot the split:
# Create figure of specific size and the axes objects
fig, ax = plt.subplots(figsize=(6, 6))

# Scatter the data points
scatter = ax.scatter(tree_df['x1'], tree_df['x2'], c=tree_df['y'], cmap='rainbow')

# Create a legend object with title "Classes" and specified location on the plot
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.add_artist(legend)

# Split lines
ax.hlines(x2_threshold, xmin=0, xmax=500,
          color ='black', lw = 0.5, ls='--')
ax.vlines(x1_threshold_2split, ymin=0, ymax=x2_threshold,
          color ='black', lw = 0.5, ls='--')
ax.hlines(x2_threshold_3split, xmin=0, xmax=x1_threshold_2split,
          color ='black', lw = 2, ls=':', label='new split')
ax.legend()

# Plot decision regions
# predict class labels on grid points
class_pred = predict_class(xx1, xx2, [('x2', x2_threshold),
                                      ('x1', x1_threshold_2split),
                                      ('x2', x2_threshold_3split)])
# contour plot for decision regions
plt.contourf(xx1, xx2, class_pred.reshape(xx1.shape), alpha=0.2, zorder=-1, cmap=plt.cm.coolwarm)

# Add labeling to the plot
ax.set_xlabel('x1', fontsize='13')
ax.set_ylabel('x2', fontsize='13')
ax.set_title('First, second, and third splits on the data', fontsize='14')

# Display the plot
plt.show()

# Perform the split
# Get the left leaf node
left_final_leaf_df  = second_split_left_df[second_split_left_df['x2'] <= x2_threshold_3split]
# Confirm the purity of the final leaf
left_final_leaf_df['y'].value_counts()

# Get the right leaf node
right_final_leaf_df = second_split_left_df[second_split_left_df['x2'] > x2_threshold_3split]
# Confirm the putity of the final leaf
right_final_leaf_df['y'].value_counts()

# Split data into predictor matrix and target series
X = tree_df[['x1','x2']]
y = tree_df['y']

# Fit the tree to the data
sklearn_tree = DecisionTreeClassifier(max_depth=3)
sklearn_tree.fit(X, y)

# Visualize the tree
tree.plot_tree(sklearn_tree, fontsize=6)
plt.show()
