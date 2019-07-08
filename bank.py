import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datasets/bank-additional-full.csv',sep = ';')

X = dataset.iloc[:,0:20].values
y = dataset.iloc[:,-1].values

dataset.isnull().sum()


from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()

X[:,1] = lab.fit_transform(X[:,1])
X[:,2] = lab.fit_transform(X[:,2])
X[:,3] = lab.fit_transform(X[:,3])
X[:,4] = lab.fit_transform(X[:,4])
X[:,5] = lab.fit_transform(X[:,5])
X[:,6] = lab.fit_transform(X[:,6])
X[:,7] = lab.fit_transform(X[:,7])
X[:,8] = lab.fit_transform(X[:,8])
X[:,9] = lab.fit_transform(X[:,9])
X[:,14] = lab.fit_transform(X[:,14])

y = lab.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,2,3,4,5,6,7,8,9,14])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 3)
dt.fit(X_train,y_train)

dt.score(X_train,y_train)
y_pred = dt.predict(X_test)

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
cm = confusion_matrix(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)