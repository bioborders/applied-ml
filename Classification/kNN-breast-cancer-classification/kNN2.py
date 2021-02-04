
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def kNN(data, predict, k=5):
    if len(data) >= k:
        warnings.warn("K is set to value less than total voting groups...")

    distances = []
    for cluster in data:
        for features in data[cluster]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict)) #dynamic array dimensions & faster
            distances.append([euclidean_distance, cluster])

    kNN_votes = [i[1] for i in sorted(distances)[:k]]
    kNN_result = Counter(kNN_votes).most_common(1)[0][0]
    confidence_kNN_result = (Counter(kNN_votes).most_common(1)[0][1] / k)
    return kNN_result
    return confidence_kNN_result

accuracies = []

for i in range(99): # 100 runs

    # load data
    df = pd.read_csv('kNN-breast-cancer-classification/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True) #replace missing data attributes
    df.drop('id', 1, inplace=True) #drop id column
    #convert all values to floats to avoid attribute being treated as a string
    full_data = df.astype(float).values.tolist()

    # shuffle data
    random.shuffle(full_data)

    # slice data
    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))] # first 80% = training data
    test_data = full_data[-int(test_size*len(full_data)):]  # last 20% = testing data

    # populate train_set and test_set dicts
    for i in train_data:
        train_set[i[-1]].append(i[:-1]) #last column = class (2 = benign; 4 = malignant)

    for i in test_data:
        test_set[i[-1]].append(i[:-1]) #last column = class (2 = benign; 4 = malignant)

    correct = 0
    total = 0

    for cluster in test_set:
        for data in test_set[cluster]: #iterate through features
            vote = kNN(train_set, data, k=5) #sklearn's k=5 (we are comparing our kNN to this)
            if cluster == vote:
                correct += 1
            total += 1

    accuracy = correct/total
    #print(Accuracy)
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies)) # mean accuracy for 100 runs
