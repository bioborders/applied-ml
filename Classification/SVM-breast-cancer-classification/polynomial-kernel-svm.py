# SVM very cumbersome to train due to quadratic programming and convex optimisation problem
# kernels are a similarity f(x)

# kernels make working with non-linear data easier by transforming the feature set
# into a different dimension and creating a linearly-seperable context without the processing cost

# inner product and dot product effectively the same (np.dot versus np.inner)

# recall that y = +/-((w-> . x->) + b)
# transforming a given dimension x, to any other dimension will not change the classifier

# y_i(x_i*w + b) - 1 >= 0
# W = sum(a_i * y_i * x_i)
# L = sum(a_i - 1/2 sum(a_i * a_j*y_i*y_j).(x_i . x_j))

# POLYNOMIAL KERNAL [see: https://en.wikipedia.org/wiki/Polynomial_kernel]
