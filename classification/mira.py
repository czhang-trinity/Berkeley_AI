# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        weights = {}
        def train_c(C):
            for iteration in range(self.max_iterations):
                print("start iteration" + str(iteration))
                self.weights = dict((label, util.Counter()) for label in self.legalLabels)
                for i in range(len(trainingData)):
                    best_score = -1
                    best_label = None
                    data = trainingData[i]
                    for label in self.legalLabels:
                        result = self.weights[label]* data
                        if result > best_score:
                            best_label = label
                            best_score = result
                    data_label = trainingLabels[i]
                    factors = ((self.weights[best_label]-self.weights[data_label])*data+1.0) / (2.0 * (data * data))
                    factors = min(factors,C)
                    copy_data = data.copy()
                    for key, value in copy_data.items():
                        value *= factors
                    if data_label != best_label:
                        self.weights[data_label] += copy_data 
                        self.weights[best_label] -= copy_data
            weights[C] = self.weights
            return sum(int(y == y_p) for y,y_p in zip(validationLabels,self.classify(validationData)))

        c_scores = [train_c(c) for c in Cgrid]
        max_c = max(c_scores)
        for c,c_score in zip(Cgrid,c_scores):
            if c_score == max_c and c < max_c:
                max_c = c
        self.C = max_c
        self.weights = weights[max_c]
        return max_c

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


