import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_csv('breast-cancer-wisconsin.data')

# Buat kolom nya
df.columns = ['id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion',
             'single_epith_cell_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses', 'class']

# Cek value kosong
# print(df.isnull().values.any())

# Agar terbaca sebagai Outliner
df.replace('?', -99999, inplace=True)

# Hapus kolom ID
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# Bagi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pilih klasifikasinya
clf = neighbors.KNeighborsClassifier()

print(X_train.shape[1])

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

# Testing
data_test = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])

data_test = data_test.reshape(len(data_test), -1)

prediksi = clf.predict(data_test)

print(prediksi)