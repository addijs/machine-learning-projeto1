from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from numpy import array

def mlp(layers, X_train, y_train, X_test, y_test):

    if len(layers) == 3:
        tuple = (layers[0], layers[1], layers[2])
    else:
        tuple = (layers[0], layers[1])

    model = MLPClassifier(hidden_layer_sizes=tuple, activation='tanh',max_iter=2000)
    model = model.fit(X_train, y_train)

    result = model.predict(X_test)

    acc = metrics.accuracy_score(result, y_test)

    show = round(acc * 100)

    print("\nMLP {} -> {}%".format(tuple, show))

    # print(list(result))
    # print("\n")
    # print(list(y_test))