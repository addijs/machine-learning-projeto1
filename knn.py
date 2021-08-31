from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def knn(k, X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', algorithm='brute')
    model = model.fit(X_train, y_train)

    result = model.predict(X_test)

    acc = metrics.accuracy_score(result, y_test)

    show = round(acc * 100)

    print("\nK = {} -> {}%".format(k, show))
    print(list(result))
    print("\n")
    print(list(y_test))