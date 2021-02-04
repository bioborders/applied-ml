# here we use the UCI repository: (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)
# => Wisconsin Breast Cancer Database (1991) from Dr William H. Wolberg

import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd

accuracies = []
runs = 99

for i in range(runs):
    df = pd.read_csv('kNN-breast-cancer-classification/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True) #replace missing data attributes
    df.drop('id', 1, inplace=True) #drop id column

    # split into training and testing sets
    X = np.array(df.drop(['class'],1)) #features = all except class; 1 = column number
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
    clf = svm.SVC(kernel = "linear") # kernel trick solves non-linearity problem
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    #print(accuracy)

    #example_measures = np.array([[6, 8, 1, 3, 2, 1, 2, 3, 1]])
    #example_measures = example_measures.reshape(len(example_measures), -1)

    #prediction = clf.predict(example_measures)
    # print(prediction)
    #if prediction == 2:
    # print("Benign")
    #else:
    # print("Malignant")

    accuracies.append(accuracy)

total_accuracy = ((sum(accuracies)) / (runs))
print(total_accuracy)
