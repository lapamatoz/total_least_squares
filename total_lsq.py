import numpy as np
import json # to format output
import matplotlib.pyplot as plt # to plot example results

def tls(x,y):
    # gives total least squares regression plane
    # for any dimensional data
    
    data = np.concatenate(([y], x), axis=0)
    
    # Evaluate covariance matrix and evaluate its
    # eigendecomposition.
    w, v = np.linalg.eig(np.cov(data))
    
    # There is least variance towards the eigenvector
    # that corresponds to the smallest eigenvalue.
    # That eigenvector is the normal of the regression plane
    min_ind = np.argmin(w)
    normal_vector = v[:,min_ind]
    
    # the regression plane intersects the mean vector point
    mean = np.mean(data, axis=1)
    
    # output results in json
    result = json.loads('{}')
    
    # some vector arithmetic to find the coefficients and intercept
    result['coefficients'] = -normal_vector[1:] / normal_vector[0]
    result['intercept'] = np.dot(mean,normal_vector) / normal_vector[0]
    
    return result

##############
# 2D EXAMPLE #
##############

# define noisy data
x = np.random.normal(np.linspace(-1,1,20),0.3)

# Define y with model: y = 2.41*x + 7.42 + noise
y = np.random.normal(2.41*x + 7.42,0.3)

# 1D x-data vector needs to be wrapped in one more brackets
x = np.array([x])

# Run tls
regression_result = tls(x,y)
print(regression_result)

# Define the regression line
x_range = np.array([np.min(x), np.max(x)])
regression_values = regression_result['coefficients'] * x_range + regression_result['intercept']

# Plot data and the regression line
plt.scatter(x,y);
plt.plot(x_range, regression_values);

##############
# 3D EXAMPLE #
##############

# define noisy data
x0 = np.random.normal(np.linspace(-1,1,20),0.3)
x1 = np.random.normal(np.linspace(-1,1,20),0.3)

# Define y as: y = -3*x0 + 0.3*x1 + 5 + noise
y = np.random.normal(-3*x0 + 0.3*x1 + 5,0.3)

# Combine the x data
X = np.array([x0,x1])

# Run tls
regression_result = tls(X,y)
print(regression_result)
