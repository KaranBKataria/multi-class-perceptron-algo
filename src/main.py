import utils
import grid_search
import algorithms

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt  
import random
import seaborn as sns  
import plotly.express as px

#%% Define all global and constant variables
accents = ["ES", "FR", "GE", "IT", "UK", "US"]

#%% Importing in the CSV file containing the data 

dataset = pd.read_csv("accent-mfcc-data-1.data", header=None)

dataset.columns = ["Accent"] + [f"X{i}" for i in range(1, 13)]

# Checking the dimensions of the dataframe:
dataset.shape

#%% Create a training dataset containing the predictors and apply PCA

X_train = dataset.drop('Accent', axis=1)
pca_matrix = utils.PCA(X_train)

# Rename the columns of the PCA matrix
pca_matrix.columns = [f"PC{i}" for i in range(1, 13)]

#%% Appending the labels column back onto the PCA dataset

pca_matrix_accent = pca_matrix.assign(Accent=dataset.Accent)

#%% Plotting the first two (and most important) Principal Components, PC1 and PC2,
# to visually determine whether the data is linearly separable in the first two PCs

# PC1-against-PC2.png
plt.figure(figsize=(15, 10))
sns.scatterplot(x=pca_matrix_accent["PC1"], y=pca_matrix_accent["PC2"], hue=pca_matrix_accent.Accent, s=150)
plt.legend(loc="center right", fontsize=15)
plt.xlabel("PC1", fontsize=25)
plt.ylabel("PC2", fontsize=25)
plt.title("Feature space defined by the top 2 PCs (PC1 against PC2)", fontsize=25)
plt.show()


#%%  Extend visual inspection to three dimensions by plotting PC1 vs PC2 vs PC3

# PC1-vs-PC2-vs-PC3-(1).png
# PC1-vs-PC2-vs-PC3-(2).png
three_d_fig = px.scatter_3d(
    ca_matrix_accent, x="PC1", y="PC2", z="PC3", color="Accent"
)

three_d_fig.show()

#%% First encode the accents into a numerical value

def encoding(x: str) -> int:
    if x == "ES":
        return 0 
    elif x == "FR":
        return 1
    elif x == "GE":
        return 2 
    elif x == "IT":
        return 3 
    elif x == "UK":
        return 4 
    elif x == "US":
        return 5

labels = pca_matrix_accent.Accent.apply(lambda x: encoding(x))


#%% Compute the hyperparameter set for gamma, containing the minimum and
# maximum value

grid_search.gamma_ranges(X_train)

# Then we produce the set of gamma values we wish to perform hyperparameter
# tuning on

grid_search.gamma_increments(X_train)

#%% Add bias to the PCA and original dataset and rearrange the columns

pca_matrix_bias = pca_matrix.insert(
    loc=0,
    column="bias",
    value=1
)

dataset_with_bias = dataset.insert(
    loc=0,
    column="bias",
    value=1
)

# Rearrange the columns
pca_matrix_bias = pca_matrix_bias[["bias"] + [f"PC{i}" for i in range(1, 13)]]

dataset_with_bias = dataset_with_bias[["bias"] + [f"X{i}" for i in range(1, 13)]]


#%% Implementing the multi-class perceptron algorithm

# Test the linear seperability in the ORIGINAL 12-D feature space
W_combined1 = utils.combined_weights(np.array(dataset_with_bias), labels)

# Determining the refined weight matrix from the multi-class perceptron after 10 iterations 
multi_class_12d = algorithms.multi_class_perceptron(dataset_with_bias, labels, W_combined1)

print(f"For the original 12D feature space, the {utils.sanity_check(dataset_with_bias, multi_class_12d)['r10']}")


# Test the linear seperability in the feature space defined by linear PCA
pca_matrix_bias = np.array(pca_matrix_bias)

# iterate through the index of the columns in the array pca_matrix_bias 
for i in list(np.arange(3, 14, 1)):

  W_combined = utils.combined_weights(pca_matrix_bias[:, 0:i], labels)
  W_combined_optimal = algorithms.multi_class_perceptron(pca_matrix_bias[:, 0:i], labels, W_combined)

  print(f"For PC1 to {str(i-1)} the {utils.sanity_check(pca_matrix_bias[:,0:i], W_combined_optimal)['r10']}")



# Test the linear seperability in the feature space defined by kernel PCA


# We produce a grid-search matrix for the entire hyperparameter space
# The hyperparameter space includes 316 PCs (13 PCs to 329 PCs) and 8 gamma values
final_grid_matrix = grid_search.grid_matrix2(X_train, labels, 329, 13)


# We do not plot the entire grid-search matrix to produce a heatmap with the
# gradient color indicating the total number of errors made for a given weight 
# matrix refined by the multi-class perceptron. However, due to the large size
# of the hyperparameter space, the plot below is simply used to showcase a successful
# plot of the grid-search matrix 

# Grid-Search-1.png
plt.figure(figsize=(15, 20))
ax = sns.heatmap(final_grid_matrix, linewidths=0.30, cbar_kws={'label': 'No. of errors'}) 
ax.invert_yaxis()
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.set_xticklabels(gamma_increments(X_train), rotation=30)
plt.xlabel("Gamma", fontsize=15)
plt.ylabel("No. of top PCs", fontsize=15)
plt.title("Matrix plot showcasing the grid search results", fontsize=15)
plt.legend("No. of errors", fontsize=15)
plt.show()



# To dervive an intuitive understanding of whether linear separability is reached
# and for which pair of hyperparameter values, we plot a grid-search matrix for
# a subset of top PCs (225 to 329 PCs)

subset_grid_matrix = grid_search.grid_matrix2(X_train, labels, 330, 225)

# Grid-Search-2.png
plt.figure(figsize=(15, 30))
ax = sns.heatmap(subset_grid_matrix, annot=True, linewidths=0.30, cbar_kws={'label': 'No. of errors'})
ax.invert_yaxis()
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.set_xticklabels(gamma_increments(X_train), rotation=30)
ax.set_yticks(list(range(0,len(range(225, 330)))))
ax.set_yticklabels(list(range(225, 330)))
plt.xlabel("Gamma", fontsize=15)
plt.ylabel("No. of top PCs", fontsize=15)
plt.title("Matrix plot showcasing the grid search results", fontsize=15)
plt.legend("No. of errors", fontsize=15)
plt.show()