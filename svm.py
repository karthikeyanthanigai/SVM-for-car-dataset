#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the Dataset
Dataset = pd.read_csv("cars_dataset.csv")
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:, -1].values


#list all the string value and convert them in to int values!!
#Note we are doing only labelencoder 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 4] = labelencoder.fit_transform(X[:, 4])
X[:, 5] = labelencoder.fit_transform(X[:, 5])


#Feature scaling the independent variable
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#SVM Classification model
from sklearn.svm import SVC
classifier = SVC(C = 10.0, kernel = "linear",decision_function_shape='ovr')
classifier.fit(X_train, y_train)

#Predicting the output for our SGD Linear Model with the test set
y_pred = classifier.predict(X_test)

#evaluating our model.
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
acc_train = accuracy_score(y_train, classifier.predict(X_train))
f1_train = f1_score(y_train, classifier.predict(X_train), average= 'weighted')

print("Traing set results")
print("ACCURACY for train set",acc_train)
print("F1 SCORE for train set",f1_train)

#evaluate our test set
acc_test = accuracy_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred, average= 'weighted')

print("Test set results")
print("ACCURACY for test set",acc_test)
print("F1 SCORE for test set",f1_test)

#Confusion Matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)