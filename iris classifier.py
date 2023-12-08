# %%
# If you'd like to install packages that aren't installed by default, list them here.
# This will ensure your notebook has all the dependencies and works everywhere

import sys


# %%
# Import libraries

import pandas as pd
import numpy as np
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# %%
# Small datasets can be added to the project directly and imported by referring to the file name
data = pd.read_csv("iris.csv")

# Preview of Data
# There are 150 observations with 4 features each (sepal length, sepal width, petal length, petal width).
# There are no null values, so we don't have to worry about that.
# There are 50 observations of each species (setosa, versicolor, virginica).

# To import large datasets (over 5MBs), you can host them externally and import by directly referring to the URL.
# For example
# data = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv')

# %%

from pandas.plotting import andrews_curves

andrews_curves(data, 'Name')

# %%
data.head()

# %%
data.info()

# %%
data.describe()

# %%
data['Name'].value_counts()

# %%
data

# %%
# data = data.drop('Id', axis=1)
g = sns.pairplot(data, hue='Name', markers='+')
plt.show()

# %%
# Data Visualization Observation
# After graphing the features in a pair plot, it is clear that the relationship between pairs of features of a iris-setosa (in pink) is distinctly different from those of the other two species.
# There is some overlap in the pairwise relationships of the other two species, iris-versicolor (brown) and iris-virginica (green).

# %%
g = sns.violinplot(y='Name', x='SepalLength', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Name', x='SepalWidth', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Name', x='PetalLength', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Name', x='PetalWidth', data=data, inner='quartile')
plt.show()

# %%
# Modeling with scikit-learn

# %%
# X = data.drop(['Id','Name'],axis=1)
X = data.drop(['Name'], axis=1)
y = data['Name']
print(X.head() , '\n')
print(X.shape  , '\n')
print(y.head() , '\n')
print(y.shape  , '\n')


# %%
# Train and test on the same dataset
# This method is not suggested since the end goal is to predict iris species using a dataset the model has not seen before.
# There is also a risk of overfitting the training data.

# %%
# experimenting with different n values
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y,y_pred))
    #if k == 3:
    #    print(knn , '\n')
    #    print(knn.fit(X,y), '\n')
    #    print(y_pred, '\n')
    #    print(scores, '\n')
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

# %%
logreg = LogisticRegression()
print(logreg, '\n')
logreg.fit(X,y)
print(logreg.fit(X,y), '\n')
y_pred = logreg.predict(X)
print(metrics.accuracy_score(y,y_pred))

# %%
# Split the dataset into a training set and a testing set
# Advantages
# - By splitting the dataset pseudo-randomly into a two separate sets, we can train using one set and test using another.
# - This ensures that we won't use the same observations in both sets.
# - More flexible and faster than creating a model using all of the dataset for training.
# Disadvantages
# - The accuracy scores for the testing set can vary depending on what observations are in the set.
# - This disadvantage can be countered using k-fold cross-validation.
# Notes
# - The accuracy score of the models depends on the observations in the testing set, which is determined by the seed of the pseudo-random number generator (random_state parameter).
# - As a model's complexity increases, the training accuracy (accuracy you get when you train and test the model on the same data) increases.
# - If a model is too complex or not complex enough, the testing accuracy is lower.
# - For KNN models, the value of k determines the level of complexity. A lower value of k means that the model is more complex.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %%
# experimenting with different n values
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    #if k == 3:
    #    print(knn , '\n')
    #    print(knn.fit(X_train,y_train), '\n')
    #    print(y_pred, '\n')
    #    print(scores, '\n')
        
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

# %%
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# %%
# Choosing KNN to Model Iris Species Prediction with k = 12
# After seeing that a value of k = 12 is a pretty good number of neighbors for this model, I used it to fit the model for the entire dataset instead of just the training set.

# %%
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)

# %%
# make a prediction for an example of an out-of-sample observation
knn.predict([[6, 3, 4, 2]]) 


