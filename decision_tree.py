from sklearn import tree
from sklearn import metrics
def decision_tree(X_train, y_train, X_test, y_test, criterion="gini"):
    # Treinamendo da Árvore de Decisão
    model = tree.DecisionTreeClassifier(criterion=criterion)
    model = model.fit(X_train, y_train)

    result = model.predict(X_test)
    acc = metrics.accuracy_score(result, y_test)
    show = round(acc * 100)
    print("\n{} {}%".format(criterion, show))

    print(list(result))
    print("\n")
    print(list(y_test))