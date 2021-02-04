# building a kNN classifier (instead of using sklearn's kNN)
# euclidean distance =
#                       LaTeX -> \sum_{i=1}^{n}{(q_i - p_i)}

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('ggplot')


# plot1 = [1,3]
# plot2 = [2,5]
# euclidean_distance = sqrt(((plot1[0] - plot2[0])**2) + ((plot1[1] - plot2[1])**2))
# print(euclidean_distance)
# = 2.23606797749979


dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]


def kNN(data, predict, k=5):
    if len(data) >= k:
        warnings.warn("K is set to value < total voting groups...")

    distances = []
    for cluster in data:
        for features in data[cluster]:
            #2D euclidean_distance = sqrt(((plot1[0] - plot2[0])**2) + ((plot1[1] - plot2[1])**2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict)) #dynamic array dimensions & faster
            distances.append([euclidean_distance, cluster])

    kNN_votes = [i[1] for i in sorted(distances)[:k]]
    kNN_result = Counter(kNN_votes).most_common(1)[0][0]
    return kNN_result

result = kNN(dataset, new_features, k=3)
print(result)

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s = 50, color = i)
        plt.scatter(new_features[0], new_features[1], s = 200, color=result)
plt.show()
