from sklearn import metrics
from collections import Counter
from sklearn.cluster import KMeans

def kmeans(X_train, y_train, X_test, y_test):

    myset = set(y_train) # Cria um conjunto. Em conjuntos, dados não se repetem. Assim, esse conjunto conterá apenas um valor de cada
    clusters = len(myset) # Quantos clusters teremos no KMeans

    model = KMeans(n_clusters = clusters)
    model = model.fit(X_train)

    # == Mapeamento das classes == #
    
    # Pegar os labels dos padrões de Treinamento
    labels = model.labels_
    
    map_labels = []

    for i in range(clusters):
        map_labels.append([])

    y_train_list = y_train.to_list()

    for i in range(len(y_train)):
        for c in range(clusters):
            if labels[i] == c:
                map_labels[c].append(y_train_list[i])
    
    # print(map_labels)

    mapping = {}

    for i in range(clusters):
        final = Counter(map_labels[i])
        value = final.most_common(1)[0][0]
        mapping[i] = value
    
    # print(mapping)

    # ============================ #

    result = model.predict(X_test)
    result = [mapping[i] for i in result]

    acc = metrics.accuracy_score(result, y_test)
    show = round(acc * 100)
    print("\nKMeans -> {}%".format(show))
    