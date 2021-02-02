
# see definitions of gradient and y-intercept below in LaTeX
# $y = mx + b$   → m is gradient; b is y-intercept
# $m = \frac{\bar{x}.\bar{y} - \overline{xy}} {(\bar{x})^2 - \overline{x^2}}$
# $b = \bar{y} - m\bar{x}$

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


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

m = line_of_best_fit(x,y)
b = y_intercept(x,y)

line = m*x + b


print(m)
print(b)

plt.scatter(x,y)
plt.plot(x, line)
plt.show()
