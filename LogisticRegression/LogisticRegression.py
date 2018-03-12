import numpy as np
import sys
import math

class ModelLogisticRegression:
    def __init__(self,learningrate,tol=0.005,max_iter=1000):
        self.learningrate = learningrate
        self.threshold = tol
        self.iterations = max_iter

    def fit(self,X,y):

        i = 1
        self.w = np.array([0] * len(X[0]))
        diff = self.threshold + 1
        allLosses = []
        initial_loss = self.calculatelogisticLoss(X, y)
        allLosses.append(initial_loss)
        while diff > self.threshold and i < self.iterations:
            scores = np.array([np.dot(self.w.T,x) for x in X])
            predictions = self.sigmoid(scores)
            error = predictions - y
            gradient = np.dot(X.T,error)
            self.w = self.w - (self.learningrate * gradient)
            updated_loss = self.calculatelogisticLoss(X,y)
            allLosses.append(updated_loss)
            diff = abs(initial_loss - updated_loss)
            initial_loss = updated_loss
            i += 1
        return range(i),allLosses

    @staticmethod
    def sigmoid(scores):
        return np.array([1 / (1 + np.exp(-score)) for score in scores])

    #model prediction for the given feature data X
    def predict(self, X):
        final_scores = np.array([np.dot(self.w.T,x) for x in X])
        preds = np.round(self.sigmoid(final_scores))
        return preds

    def calculatelogisticLoss(self,X,y):

        scores = np.array([np.dot(self.w.T,x) for x in X])
        predictions = self.sigmoid(scores)
        return -1 * (np.sum((y * np.log(predictions)) + ((1 - y) * np.log(1 - predictions))))
