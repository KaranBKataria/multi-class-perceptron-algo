import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt  
import random
import seaborn as sns  
import plotly.express as px  

#%% Importing in the CSV file containing the data 

dataset = pd.read_csv("accent-mfcc-data-1.data", header=None)

dataset.columns = ["Accent"] + [f"X{i}" for i in range(1, 13)]

# Checking the dimensions of the dataframe:
dataset.shape

#%% Create a training dataset containing the predictors and apply PCA

X_train = dataset.drop('Accent', axis=1)
pca_matrix = PCA(X_train)

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


#%% We can extend visual inspection to three dimensions by plotting PC1 vs PC2 vs PC3

three_d_fig = px.scatter_3d(
    ca_matrix_accent, x="PC1", y="PC2", z="PC3", color="Accent"
)

three_d_fig.show()