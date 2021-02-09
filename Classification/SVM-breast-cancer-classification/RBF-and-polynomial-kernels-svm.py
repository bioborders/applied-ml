# SVM very cumbersome to train due to quadratic programming and convex optimisation problem
# kernels are a similarity f(x). Of the kernelised ML algorithms, SVM is most well known.

# kernels make working with non-linear data easier by transforming the feature set
# into a different dimension and creating a linearly-seperable context without the processing cost

# Kernel methods owe their name to the use of kernel functions, which enable them to operate in a
# high-dimensional, implicit feature space without ever computing the coordinates of the data in
# that space, but rather by simply computing the inner products between the images of all pairs of
# data in the feature space. This operation is often computationally cheaper than the explicit
# computation of the coordinates. This approach is called the "kernel trick". Kernel functions have
# been introduced for sequence data, graphs, text, images, as well as vectors.

# inner product and dot product effectively the same (np.dot versus np.inner)

# Note:  Any linear model can be turned into a non-linear model by applying the kernel trick to the model:
# replacing its features (predictors) by a kernel function

# recall that y = +/-((w-> . x->) + b)
# transforming a given dimension x, to any other dimension will not change the classifier

# y_i(x_i*w + b) - 1 >= 0
# W = sum(a_i * y_i * x_i)
# L = sum(a_i - 1/2 sum(a_i * a_j*y_i*y_j).(x_i . x_j))

# POLYNOMIAL KERNAL [see: https://en.wikipedia.org/wiki/Polynomial_kernel]

# note: phi denotes the kernel, here I refer to it as 'K'
# e.g. y = WKx+b

# X = [x1,x2] -> 2nd order polynomial -> Z = [, x1, x2, x1^2, x2^2, x1x2]
