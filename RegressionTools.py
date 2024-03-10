import numpy as np


def extendDataSet(data):
    extendedData = np.concatenate((data, np.ones((len(data), 1))), axis=1)
    return extendedData


def closedFormSolution(extendedData, y_vector):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(extendedData), extendedData)),
                     np.transpose(extendedData)), y_vector)


def calculateMeanSquareError(actualData, predictedData):
    result = 0
    iterator = len(actualData)
    for i in range(iterator):
        result += (predictedData[i]-actualData[i])**2
    result /= iterator
    return result


