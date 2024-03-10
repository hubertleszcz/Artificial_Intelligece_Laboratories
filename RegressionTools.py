import numpy as np


def extendDataSet(data):
    ones_column = np.ones((len(data), 1))
    extendedData = np.concatenate((ones_column, data), axis=1)
    return extendedData


def closedFormSolution(extendedData, y_vector):
    result = extendedData
    result = np.transpose(result)
    result = np.dot(result, extendedData)
    result = np.linalg.inv(result)
    result = np.dot(result, np.transpose(extendedData))
    result = np.dot(result, y_vector)
    return result


def calculateMeanSquareError(actualData, predictedData):
    result = 0
    iterator = len(actualData)
    for i in range(iterator):
        result += (predictedData[i]-actualData[i])**2
    result /= iterator
    return result


def standarize(data, population):
    return (data - np.mean(population))/np.std(population)


def reverseStandarizization(data, population):
    return data * np.std(population) + np.mean(population)


def gradientMSE(X, theta, yVector):
    m = len(yVector)
    yVector = yVector.reshape(-1, 1)
    return (2/m) * np.dot(np.transpose(X), (np.dot(X, theta) - yVector))


def getGradientDescent(theta, learningRate, dataMatrix, yVector):
    loopIterator = 100
    currentMSE = 100
    previousMSE = 100
    for _ in range(loopIterator):
        theta = theta - learningRate * gradientMSE(dataMatrix, theta, yVector)
        previousMSE = currentMSE
        currentMSE = calculateMeanSquareError(yVector, np.dot(dataMatrix, theta))
        #if abs((currentMSE - previousMSE)).any() <= 1e-8:
         #   break

    return theta
