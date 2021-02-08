import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# we want the svm to act like an object, so we can train it an save it later

class Support_Vector_Machine: #intitialisation
    def __init__(self, visualisation=True):
        self.visualisation = visualisation
        self.colours  = {1:'r', -1:'b'}
        if self.visualisation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data): #training / optimisation
        self.data = data
        opt_dict = {}
        # { ||W||: [w,b]}
        transforms = [[1,1],
                     [-1,1],
                     [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,]

        b_range_multiple = 5
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimised = False #convex problem
            while not optimised:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value & b_range_multiple,
                                   step * b_multiple):

                   for transformation in transforms:
                       w_t = w * transformation
                       found_option = True
                       # calculation must be run on ALL data!
                       # yi (xi . w_b) >= 1
                       # if a single value is false - break.
                       for i in self.data:
                           for xi in self.data[i]:
                               yi = i
                               if not yi * (np.dot(w_t,xi)+b) >=1:
                                   found_option = False

                       if found_option:
                           opt_dict[np.linalg.norm(w_t)] = [w_t, b] #magnitude

            if w[0] < 0:
               optimised = True
               print('Optimised a Step')
            else:
               w = w - step

           # sorted list of all magnitudes
           # optimal = 0th element of opt_dict (smallest norm)
           # dict = ||w|| [w,b]
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step * 2


        # self.w
        # self.b

    def predict(self, features): #prediction
        # sign(x_i.w+b)
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        if classification != 0 and self.visualisation:
            self.ax.scatter(features[0], features[1], s = 100, marker='*', c=self.colors[classification])
        return classification

        def visualise(self): # has no bearing on the SVM
            #self.data = data_dict
            [[self.ax.scatter(x[0],x[1], size=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

            # v = x.w+#
            # +ve = 1; -ve = -1
            # decision boundary f(x)
            def hyperplane(x, w, b, v):
                return (-w[0]*x-b+v) / w[1]

            datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
            hyp_x_min = datarange[0]
            hyp_x_max = datarange[1]

            # positive SV hyperplane
            pos1 = hyperplane(hyp_x_min, self.w, self.b, 1)
            pos2 = hyperplane(hyp_x_max, self.w, self.b, 1)
            self.ax.plot([hyp_x_min, hyp_x_max], [pos1, pos2])

            # negative SV hyperplane
            neg1 = hyperplane(hyp_x_min, self.w, self.b, -1)
            neg2 = hyperplane(hyp_x_max, self.w, self.b, -1)
            self.ax.plot([hyp_x_min, hyp_x_max], [neg1, neg2])

            # decision boundary
            dec1 = hyperplane(hyp_x_min, self.w, self.b, 0)
            dec2 = hyperplane(hyp_x_max, self.w, self.b, 0)
            self.ax.plot([hyp_x_min, hyp_x_max], [dec1, dec2])

            plt.show()

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
              1:np.array([[5,1],
                          [6,-1],
                          [7,3],])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)
svm.visualise()
