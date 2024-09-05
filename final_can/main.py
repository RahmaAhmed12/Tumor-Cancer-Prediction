import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 31].values

from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=24)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
# X = scaler.fit_transform(X)


# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier = KNeighborsClassifier(n_neighbors=100, weights='distance')
KNeighborsClassifier.fit(X_train, Y_train)
y_pred5 = KNeighborsClassifier.predict(X_test)

cm5 = confusion_matrix(Y_test, y_pred5)
cr5 = classification_report(Y_test, y_pred5)

# Fitting svm to the training set
from sklearn.svm import SVC

SVC = SVC(kernel='linear', random_state=24)
SVC.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = SVC.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm4 = confusion_matrix(Y_test, y_pred)
cr = classification_report(Y_test, y_pred)

# logistic regression
from sklearn.linear_model import LogisticRegression

LogisticRegression = LogisticRegression(solver='liblinear', C=15, max_iter=10)
LogisticRegression.fit(X_train, Y_train)
# predicting the test set result
y_pred2 = LogisticRegression.predict(X_test)
# making confusion matrix
from sklearn.metrics import confusion_matrix

cm2 = confusion_matrix(Y_test, y_pred2)
cr2 = classification_report(Y_test, y_pred2)

# Decision Tree Classification
# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier

DecisionTreeClassifier = DecisionTreeClassifier(criterion='entropy', random_state=5)
DecisionTreeClassifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred3 = DecisionTreeClassifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm3 = confusion_matrix(Y_test, y_pred3)
cr3 = classification_report(Y_test, y_pred3)

# Random forest classification
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier = RandomForestClassifier(n_estimators=2, criterion='entropy', random_state=24)
RandomForestClassifier.fit(X_train, Y_train)
# predicting the test set result
y_pred4 = RandomForestClassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm4 = confusion_matrix(Y_test, y_pred4)
cr4 = classification_report(Y_test, y_pred4)



def predictLogisticRegression():
    D = pd.read_csv('DD.csv')
    D = D.dropna()
    D.drop('Index',axis='columns',inplace=True)
    D.drop_duplicates()
    v = D.iloc[:, :-1].values

    logisticRegressionPredict = LogisticRegression.predict(v)

    return logisticRegressionPredict


def predictSVM():
    D = pd.read_csv('DD.csv')
    D = D.dropna()
    D.drop('Index', axis='columns', inplace=True)
    D.drop_duplicates()
    v = D.iloc[:, :-1].values

    SVMPredict = SVC.predict(v)

    return SVMPredict


def predictKN():
    D = pd.read_csv('DD.csv')
    D = D.dropna()
    D.drop('Index', axis='columns', inplace=True)
    D.drop_duplicates()
    v = D.iloc[:, :-1].values
    KNPredict = KNeighborsClassifier.predict(v)

    return KNPredict


def predictDecisionTree():
    D = pd.read_csv('DD.csv')
    D = D.dropna()
    D.drop('Index', axis='columns', inplace=True)
    D.drop_duplicates()
    v = D.iloc[:, :-1].values
    DecisionTreePredict = DecisionTreeClassifier.predict(v)

    return DecisionTreePredict


def predictRandomforest():
    D = pd.read_csv('DD.csv')
    D = D.dropna()
    D.drop('Index', axis='columns', inplace=True)
    D.drop_duplicates()
    v = D.iloc[:, :-1].values
    RandomforestPredict = RandomForestClassifier.predict(v)

    return RandomforestPredict


p1= predictLogisticRegression()
p2 = predictSVM()
p3 = predictDecisionTree()
p4 = (predictRandomforest())
p5 = (predictKN())

A = p1 + p2 + p3 + p4 + p5

if A >= 3:
    print('M')
else:
    print('B')