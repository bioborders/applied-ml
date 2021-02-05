import matplotlib as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# we want the svm to act like an object, so we can train it an save it later

class Support_Vector_Machine:
    def __init__(self, visualisation=True):
        self.visualisation = visualisation
        self.colours  = {1:'r', -1:'b'}
        if self.visualisation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data):
        pass
        # self.w
        # self.b

    def predict(self, features):
        # sign(x_i.w+b)
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        return classification





data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
              1:np.array([[5,1],
                          [6,-1],
                          [7,3],])}
