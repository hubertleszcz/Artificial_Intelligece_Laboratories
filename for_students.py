import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data
import RegressionTools as tools

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy().reshape(-1, 1)


y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy().reshape(-1, 1)

# TODO: calculate closed-form solution

extendedTrainData = tools.extendDataSet(x_train)

theta_best = tools.closedFormSolution(extendedTrainData, y_train)
print(theta_best)


extendedTestData = tools.extendDataSet(x_test)

predictedData = np.dot(extendedTestData, theta_best)
# TODO: calculate error
MSE = tools.calculateMeanSquareError(y_test, predictedData)
print(MSE)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization

standarizedXData = tools.standarize(x_train, x_train)
#print(len(standarizedXData))

standarizedYData = tools.standarize(y_train, y_train)
#print(standarizedYData)

# TODO: calculate theta using Batch Gradient Descent
thetaForGradient = np.random.rand(1, 2)
print(thetaForGradient)
learningRate = 0.1
thetaGradient = tools.getGradientDescent(thetaForGradient, learningRate, standarizedXData, standarizedYData)
thetaGradient = thetaGradient.T


# TODO: calculate error
standarizedXTest = tools.standarize(x_test, x_train)
standarizedYTest = tools.standarize(y_test, y_train)
predictedGradientData = tools.extendDataSet(standarizedXTest).dot(thetaGradient)
MSE = tools.calculateMeanSquareError(standarizedYTest, predictedGradientData)
print(MSE)

# plot the regression line
x = np.linspace(min(standarizedXTest), max(standarizedXTest), 100)
y = float(thetaGradient[0]) + float(thetaGradient[1]) * x
plt.plot(x, y)
plt.scatter(standarizedXTest, standarizedYTest)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
