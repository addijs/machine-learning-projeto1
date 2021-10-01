from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def knn(metric, k, X_train, y_train, X_test, y_test):

    # Treinamento do KNN
    model = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm='brute')
    model = model.fit(X_train, y_train)

    result = model.predict(X_test)

    acc = metrics.accuracy_score(result, y_test)

    show = round(acc * 100)

    print("\nKNN {}: K = {} -> {}%".format(metric, k, show))
    # print(list(result))
    # print("\n")
    # print(list(y_test))