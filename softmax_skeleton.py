import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta, tempParameter):
    #YOUR CODE HERE
    pass

def computeCostFunction(X, Y, theta, lambdaFactor, tempParameter):
    #YOUR CODE HERE
    pass

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter):
    #YOUR CODE HERE
    pass

def updateY(trainY, testY):
    #YOUR CODE HERE
    pass

def computeTestErrorMod3(X, Y, theta, tempParameter):
    #YOUR CODE HERE
    pass

def softmaxRegression(X, Y, tempParameter, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor, tempParameter))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter)
    return theta, costFunctionProgression
    
def getClassification(X, theta, tempParameter):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta, tempParameter)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def computeTestError(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    return 1 - np.mean(assignedLabels == Y)
