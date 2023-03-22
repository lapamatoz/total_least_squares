import numpy as np
import json # to format output
import matplotlib.pyplot as plt # to plot example results

def total_least_squares(x):
    # gives total least squares regression line
    # for any dimensional data
    
    # evaluate covariance matrix
    # variance most is eliminated with an eigenvector,
    # that corresponds to the largest eigenvalue
    
    w, v = np.linalg.eig(np.cov(x))
    max_ind = np.argmax(w)
    slope_vector = v[:,max_ind]
    
    # the regression line intersects the mean vector point
    mean = np.mean(x, axis=1)
    
    # different ways to output results
    result = json.loads('{}')
    
    # for multivariate regression line
    result['slope vector'] = slope_vector
    result['mean vector'] = mean
    
    # for 2D regression line
    result['slope'] = slope[1]/slope[0]
    result['intercept'] = -mean[0] * result['slope'] + mean[1]
    
    return result


# EXAMPLE

# define noisy data
x = np.random.normal(np.linspace(-1,1,20),0.3)
y = np.random.normal(np.linspace(-1,1,20),0.3)

# data is given this way
data = np.array([x,y])
regression_result = total_least_squares(data)

# define the regression line
x_range = np.array([np.min(x), np.max(x)])
regression_values = regression_result['slope'] * x_range + regression_result['intercept']

# plot data and the regression line
plt.scatter(x,y);
plt.plot(x_range, regression_values);
