
# see definitions of gradient and y-intercept below in LaTeX
# $y = mx + b$   → m is gradient; b is y-intercept
# $m = \frac{\bar{x}.\bar{y} - \overline{xy}} {(\bar{x})^2 - \overline{x^2}}$
# $b = \bar{y} - m\bar{x}$

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('dark_background')
np.set_printoptions(precision = 2)


from statistics import mean
import numpy as np
import  matplotlib.pyplot as plt

x = np.array([4, 6, 17, 22, 25, 26], dtype=np.float64)
y = np.array([3, 6, 4, 8, 9, 10], dtype=np.float64)


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


m = line_of_best_fit(x,y)
b = y_intercept(x,y)

line = (m*x)+ b

predict_x = 50
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(y, line)
print(r_squared)
# 0.6737903126614351

print(m,b)
# 0.2402031930333817 2.663280116110305


plt.title("y = mx + b")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x,y)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(x, line)
plt.show()
