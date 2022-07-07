Importing initial libraries

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn.linear_model import LinearRegression #fitting the linear regression model to the dataset
from sklearn.metrics import r2_score

# Generating randome data

n = 100 # number of inputs
x = random.uniform(0.0, 1.0, n) # x_i || x = random.randint(100,size=(n)) #=> x_i
epsilon = random.rand(n) # Return a sample (or samples) from the “standard normal” distribution. && we have {epsilon=sigma * np.random.randn(n) + mu} for creating other distributions

# Plotting what we generated

plt.hist(x, 10) # draw a histogram of x_i with 10 bars
plt.show()

plt.hist(epsilon, 10) # draw a histogram of epsilon_i with 10 bars
plt.show()

# Function Definition

f = lambda t : t**3-2*t+1
y = f(x)+epsilon # adding the noise of system

# Alternatively we can use "noise=slope * x + intercept" and put "y=y+noise"

plt.scatter(x, f(x))
plt.show()

plt.scatter(x, y)
plt.show()

# Linear Regression

slope, intercept, r, p, std_err = stats.linregress(x, y)
print(slope, intercept, r, p, std_err )

#This relationship - the coefficient of correlation - is called r.
#The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related.

#Create a function that uses the slope and intercept values to return a new value.
#slope <=> sigma && mu <=> intercept

def m_func(x):
  return slope * x + intercept

model0 = list(map(m_func, x))

plt.scatter(x, y)
plt.plot(x, model0)
plt.show()

# Polynomial Regression

model1 = np.poly1d(np.polyfit(x,y,3)) # polynomial deg: 3
print(model1.coefficients)
line1 = np.linspace(0, 1, 100)

plt.scatter(x, y)
plt.plot(line1, model1(line1))
plt.show()

# Fitting the data

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

plt.scatter(test_x, test_x,color='red')
plt.scatter(train_x, train_y)
plt.show()

model2 = np.poly1d(np.polyfit(train_x, train_y, 3)) # deg : 3
line2 = np.linspace(0, 1, 100)

plt.scatter(train_x, train_y)
plt.plot(line2, model2(line2),color='red')
plt.show()

# R2
# It measures the relationship between the x axis and the y axis, and the value ranges from 0 to 1, where 0 means no relationship, and 1 means totally related.

r2 = r2_score(train_y, model2(train_x))
print(r2)

import numpy as np
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse import csr_matrix

arr = np.array([
  [0, 1, 0, 1,0],
  [1, 1, 1, 1,0],
  [2, 1, 1, 0,0],
  [1, 1, 1, 1,1],
  [0, 1, 0, 1,0]
])

newarr = csr_matrix(arr)

print(depth_first_order(newarr, 6))
