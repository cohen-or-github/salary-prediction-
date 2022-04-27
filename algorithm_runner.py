from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_validate
import numpy as np


class AlgorithmRunner:

    def __init__(self, classifier, is_knn, print_all):
        """
        :param classifier: the wanted classifier
        :param is_knn: is the classifier is KNN
        """
        self.classifier = classifier
        self.is_knn = is_knn
        self.print_all = print_all

    def run(self, data, folds=5):
        """
        Run the classifier and print the accuracy, precision, recall
        :param data: the data
        :param folds: KFolds object
        :return: the accuracy
        """
        cv = data.split_to_k_folds(folds)
        # cv = cv.split(data.data)
        x = data.data.drop(columns=['salary'])
        score = cross_validate(self.classifier, x, data.data['salary'], cv=cv, scoring=['precision', 'recall',
                                                                                        'accuracy'])
        accuracy = np.mean(score['test_accuracy'])

        if self.is_knn:
            name = "KNN"
        else:
            name = "Rocchio"

        if self.print_all:
            recall = np.mean(score['test_recall'])
            precision = np.mean(score['test_precision'])
            print(str(name) + " classifier: " + str(round(precision, 3)) + ", "+ str(round(recall, 3)) + ", "
                  + str(round(accuracy, 3)))

        else:
            print(str(name) + " classifier: " + str(round(accuracy, 3)))

        return accuracy


