import numpy as np
from scipy.stats import multivariate_normal


def gaussian_data(mu, sigma, x_size, y_size):

    x, y = np.mgrid[-1.0:1.0:1j*x_size, -1.0:1.0:1j*y_size]
    xy = np.column_stack([x.flat, y.flat])
    covariance = np.diag(sigma**2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    # # Reshape back to a grid.
    
    return z.reshape(x.shape)

mu = np.array([5.0, 5.0])
sigma = np.array([5.0, 5.0])
x_size = 1024
y_size = 1024

data = gaussian_data(mu, sigma, x_size, y_size)
print(data)