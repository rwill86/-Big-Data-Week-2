#!/usr/bin/python

# Try to open imports
try:
    import sys
    import random
    import math
   import os
    import time
    import numpy as np
	import pandas as pd
    from matplotlib import pyplot as plt
	from sklearn.decompostion import PCA as sklearnPCA

# Error when importing
except ImportError:
    print('### ', ImportError, ' ###')
    # Exit program
    exit()


# Read input
def read():
    # Examples Panda
    ser1 = pd.Series([1, 2, 3 , 4], index = ['USA', 'Germany', 'USSR'. 'Japan'])
	dates = pd.date_range('20130101', peroids = 6)
	df = pd.DataFrame(np.random.randn(5, 4), index = 'A B C D E'.split(), columns = 'W X Y Z'.split())
	df['W']
	type(df['W'])
	df[['W', 'Z']]
	df[df['W'] > 0]
	
	# Drop
	df = pd.DataFrame({'A': [1, 2, np.nan], 'B':[5, np.nan, np.nan], 'C':[1, 2, 3]})
	df.dropna()
	df.dropn(axis = 1)
	df.dropna(thresh = 2)
	
	# Fill
	df.fillna(value = "FILL VALUE")
	df['A'].fillna(value=df['A'].mean())
	
	# Find
	df.head()
	df.isnull()
	
	# Groupby
	df.groupby('Company').mean()
	
	# Merging
	left = pd.DataFrame({'A': [1, 2, 4], 'B':[5, 8, 2], index = ['K0', 'K1', 'K2'])
	right = pd.DataFrame({'C': [1, 2, 9], 'D':[4, 7, 1], index = ['K0', 'K2', 'K2'])
	pd.merge(left, right, how = 'inner', on = 'key')
	
	#Join
	left.join(right)
	left join(right, how = 'outer')
	
	# Sort
    df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list('ABCD'))
    df.sort_values(by = 'B')
	
	df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list('ABCD'))
	df.head()
	df.tail(3)
	df.index
	df.columns
	df.describe()
	df.T
	df.sort_index(axis = 1, ascending = False)
	df.sort_values(by = 'B')
	df.to_numpy()
	
	# Week 2
	# Basic Functions
	# df.sum()
	# df.count()
	# median()
	# min()
	# max()
	# var()
	# std()
	pd.read_csv('PastHires.csv')  
	df.sort_values(by = 'Years Experience')
    df.value_counts(by = 'Level of Education')
	df.plot.hist()
	
	#PCA
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(url, header = None)
    y = data[4]
    X = data.iloc[:, 0:4]

    plt.scatter(X[y=='Iris-setosa'].iloc[:,0], X[y=='Iris-setosa'].iloc[:,1],label='Iris-setosa', c = 'red')
    plt.scatter(X[y=='Iris-versicolor'].iloc[:,0], X[y=='Iris-versicolor'].iloc[:,1],label='Iris-versicolor', c = 'blue')
    plt.scatter(X[y=='Iris-virginica'].iloc[:,0], X[y=='Iris-virginica'].iloc[:,1],label='Iris-virginica', c = 'lightgreen')

    plt.legend()
    plt.ylabel('sepal_length')
    plt.ylabel('sepal_width')
    plt.show()

    X_norm = (X - X.min()) / (X.max() - X.min())
    plt.scatter(X_norm[y=='Iris-setosa'].iloc[:,0], X_norm[y=='Iris-setosa'].iloc[:,1],label='Iris-setosa', c = 'red')
    plt.scatter(X_norm[y=='Iris-versicolor'].iloc[:,0], X_norm[y=='Iris-versicolor'].iloc[:,1],label='Iris-versicolor', c = 'blue')
    plt.scatter(X_norm[y=='Iris-virginica'].iloc[:,0], X_norm[y=='Iris-virginica'].iloc[:,1],label='Iris-virginica', c = 'lightgreen')


    plt.legend()
    plt.ylabel('Feature A')
    plt.ylabel('Feature B')
    plt.show()
	
    pca = skleanrPCA(n_componets = 2)
    transformed = pd.DataFrame(pca.fit_transform(x_norm))
	
    data = pd.read_csv('wine.data')
	wine = []
    standardScaler = preprocessing.StandardScaler()
    standardScaler.fit(wine)
    X_scaled_array = standardScaler.transform(wine)
    normalizedData = pd.DataFrame(X_scaled_array, columns = wine.columns)
	
	kMeansClustering = KMeans(n_clusters - 3, random_state = seed)
    res = kMeansClustering.fit_predict(normalizedData)
	
	normalizedDate["cluster"] = label_pred_KM.astype('float64')
	sns_plot = sns.pairplot(normalizedData, hue = 'cluser', diag_kind = 'hist')
	
    adjected_rand_score(label, label_pred_KM_PCA)
# Main
def main():
    # Read Input
    read()
    # Close Program
    exit()


# init
if __name__ == '__main__':
    # Begin
    main()


