import sys
from data import Data
from algorithm_runner import AlgorithmRunner
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


def main(argv):
    
    # Question 1:
    dict = Data(argv[1], MinMaxScaler())
    a = AlgorithmRunner(KNeighborsClassifier(n_neighbors=15), True, True)
    best_knn = a.run(dict)
    b = AlgorithmRunner(NearestCentroid(), False, True)
    best_rocchio = b.run(dict)

    # Question 2:

    print("")
    print("Question 2:")

    # Check which scaler gives the best results:
    # scalers_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]
    # best_k = 1
    # best_scaler = MinMaxScaler
    # for scaler in scalers_list:
        # new_data = Data(argv[1], scaler)

        # new_rocchio = AlgorithmRunner(NearestCentroid(), False, False)
        # new_result_rocchio = new_rocchio.run(new_data)

        # for k in range(1,31):
            # new_knn = AlgorithmRunner(KNeighborsClassifier(n_neighbors=k), True, False)
            # new_result_knn = new_knn.run(new_data)
            # if new_result_knn > best_knn and new_result_rocchio >= best_rocchio:
                # best_scaler = scaler
                # best_k = k
                # best_knn = new_result_knn
                # best_rocchio = new_result_rocchio

    # print("best scaler is: " + str(best_scaler) + " with " + str(best_k) + " neighbours")

    # according to the calculation, the best scaler is Standard scaler with 25 neighbours.

    new_data = Data(argv[1], StandardScaler())
    new_a = AlgorithmRunner(KNeighborsClassifier(n_neighbors=25), True, False)
    best_knn = new_a.run(new_data)
    new_b = AlgorithmRunner(NearestCentroid(), False, False)
    best_rocchio = new_b.run(new_data)


if __name__ == '__main__':
    main(sys.argv)
