import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

file_path = 'Lanches.xlsx'
df = pd.read_excel(file_path)

df_train = df[df['Tipo de proteína'] != '?']
df_test = df[df['Tipo de proteína'] == '?']

X_train = df_train.drop(columns=['Tipo de proteína'])
y_train = df_train['Tipo de proteína']

encoder = OneHotEncoder(sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train)

X_test = df_test.drop(columns=['Tipo de proteína'])
X_test_encoded = encoder.transform(X_test)

def knn_prediction(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_encoded, y_train)
    y_pred = knn.predict(X_test_encoded)
    return y_pred[0]

protein_k1 = knn_prediction(1)
print(f"Proteína sugerida com K=1: {protein_k1}")

protein_k3 = knn_prediction(3)
print(f"Proteína sugerida com K=3: {protein_k3}")

protein_k5 = knn_prediction(5)
print(f"Proteína sugerida com K=5: {protein_k5}")
