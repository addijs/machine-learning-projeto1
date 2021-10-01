from kmeans import kmeans
from mlp import mlp
from decision_tree import decision_tree
from sklearn.model_selection import train_test_split
import pandas as pd

from knn import knn

url = './data/transfusion.data'

# Carregar base de dados
dataset = pd.read_csv(url, header=None)

columns = len(dataset.columns)

y = dataset.loc[:, columns-1] # extrai a última coluna, que é o label
X = dataset.loc[:, 0:columns-2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) 
# 80% treino e 20% teste

n1 = 5  # K = 5
n2 = 10 # K = 10

mlp1_hidden_layer_sizes = [3,2]
mlp2_hidden_layer_sizes = [4,4,4]

decision_tree(X_train, y_train, X_test, y_test, criterion="entropy")
knn('euclidean', n1, X_train, y_train, X_test, y_test)
knn('euclidean', n2, X_train, y_train, X_test, y_test)
mlp(mlp1_hidden_layer_sizes, X_train, y_train, X_test, y_test)
mlp(mlp2_hidden_layer_sizes, X_train, y_train, X_test, y_test)
kmeans(X_train, y_train, X_test, y_test)