from sklearn.model_selection import train_test_split
# from sklearn import metrics
import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier

from knn import knn

url = './data/balance-scale.data'

# Carregar base de dados
dataset = pd.read_csv(url, header=None)

columns = len(dataset.columns)

y = dataset[0] # extrai a primeira coluna, que é o label
X = dataset.loc[:,1:columns-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) 
# 80% treino e 20% teste

n1 = 5  # K = 5
n2 = 10 # K = 10
n3 = 20 # K = 20

knn(n1, X_train, y_train, X_test, y_test)
knn(n2, X_train, y_train, X_test, y_test)
knn(n3, X_train, y_train, X_test, y_test)