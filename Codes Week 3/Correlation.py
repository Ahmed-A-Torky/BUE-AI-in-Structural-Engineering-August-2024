# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:48:06 2020
https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
@author: Ahmed_A_Torky
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Read dataset to pandas dataframe
dataset = pd.read_csv('quakes_values.csv')

plt.scatter(dataset['Magnitude(M)'],dataset['NumberOfSites'], s=10, marker='*')
plt.title("Correlation between variables")
plt.xlabel('Magnitude(M)')
plt.ylabel('NumberOfSites')
plt.show()

plt.scatter(dataset['Magnitude(M)'],dataset['Intensity'], s=10, marker='*')
plt.title("Correlation between variables")
plt.xlabel('Magnitude(M)')
plt.ylabel('Intensity')
plt.show()

# Find the pearson correlations matrix
corr = dataset.corr(method = 'pearson')
 
# correaltions between age and sex columns
c = np.corrcoef(dataset['Magnitude(M)'],dataset['NumberOfSites'])
print('Correlations between M and Sites\n',c)

# Plot the correlation heatmap 
plt.figure(figsize=(10,8), dpi =500)
plt.title("Pearson Correlation")
sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.5)
plt.show()

# Find the pearson correlations matrix
corr = dataset.corr(method = 'spearman')

# Plot the correlation heatmap 
plt.figure(figsize=(10,8), dpi =500)
plt.title("Spearman Correlation")
sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.5)
plt.show()