
# see definitions of gradient and y-intercept below in LaTeX
# $y = mx + b$   → m is gradient; b is y-intercept
# $m = \frac{\bar{x}.\bar{y} - \overline{xy}} {(\bar{x})^2 - \overline{x^2}}$
# $b = \bar{y} - m\bar{x}$

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('seaborn-dark-palette')
np.set_printoptions(precision = 2)


from statistics import mean
import numpy as np
import  matplotlib.pyplot as plt

#x = np.array([4, 6, 17, 22, 25, 26], dtype=np.float64)
#y = np.array([3, 6, 4, 8, 9, 10], dtype=np.float64)

def create_dataseet(number_data_points, variance, step=2, correlation=False):
    val = 1
    y = []
    for i in range(number_data_points):
        y_val = val + random.randrange(-variance, variance)
        y.append(y_val)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
        x = [i for i in range(len(y))]

    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def line_of_best_fit(x,y):
    numerator = (np.mean(x) * np.mean(y)) - np.mean((x*y))
    denominator = np.mean(x)**2 - np.mean(x**2)
    m = numerator/denominator
    return m

def y_intercept(x,y):
    b = ((np.mean(y)) - (m*np.mean(x)))
    return b

def square_error(y_points, y_line):
    return sum((y_line - y_points)**2)

    #r**2 = 1 - ((SE*np.mean(y - regression line)) /
    #            (SE*np.mean(y - mean line))

def coefficient_of_determination(y_points, y_line):
    y_mean_line = [mean(y_points) for y in y_points]
    square_error_regression_line = square_error(y_points, y_line)
    square_error_y_mean = square_error(y_points, y_mean_line)
    return 1 - (square_error_regression_line / square_error_y_mean)


x, y = create_dataseet(50, 40, 2, correlation='pos')
m = line_of_best_fit(x,y)
b = y_intercept(x,y)

line = (m*x)+ b
predict_x = 60
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(y, line)

print(r_squared)
print(m,b)


plt.title("y = mx + b")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x,y)
plt.scatter(predict_x, predict_y, s=100, color='r')
plt.plot(x, line)
plt.show()
