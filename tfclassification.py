import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv('breast-cancer-wisconsin.data')

# Buat kolom nya
df.columns = ['id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion',
             'single_epith_cell_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses', 'class']

# Cek value kosong
print(df.isnull().values.any())

# print(df.head())
# Untuk Outliner
df.replace('?', -99999, inplace=True)

# Hapus kolom ID
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
#
X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')


# Bagi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build Multi Layer Perceptron (MLP) Model

# Normalisasi
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

onehot_y_train = pd.get_dummies(y_train).values
onehot_y_test = pd.get_dummies(y_test).values


# Initialize Parameter
input_dim = scaled_x_train.shape[1] # jumlah atribut
output_dim = 2 # menghasilkan output binary
lr = 0.0001
optimizer = Adam(learning_rate=lr)
batch_size = 512
epochs = 350


# Build Model
model = Sequential([
          # Dense layer with 100 neuron, input shape and relu activation function
          Dense(300, input_shape=(input_dim,), activation='relu'),
          # Dropout layer with 0.5 probability
          Dropout(0.5),
          # Dense layer with 200 neuron and relu activation function
          Dense(200, activation='relu'),
          # Dropout layer with 0.5 probability
          Dropout(0.5),
          # Dense layer with 100 neuron and relu activation function
          Dense(100, activation='relu'),
          # Dropout layer with 0.5 probability
          Dropout(0.5),
          # Dense layer with 100 neuron and sigmoid activation function
          Dense(output_dim, activation='sigmoid')
], name="cardionet")

# model.summary()
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Train Model
history = model.fit(scaled_x_train, onehot_y_train,
                    validation_data=(scaled_x_test, onehot_y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2)

#Visualize Training Result
fig, ax = plt.subplots(1,2,figsize=(18,3))
ax[0].plot(history.history["accuracy"])
ax[0].plot(history.history["val_accuracy"])
ax[0].set_title("Train and Validation Accuracy")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].legend(['Train', 'Validation'], loc='upper right')

ax[1].plot(history.history["loss"])
ax[1].plot(history.history["val_loss"])
ax[1].set_title("Train and Validation Loss")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend(["Train", "Validation"], loc="upper right")
plt.show()

#Calculate Accuracy
results = model.evaluate(scaled_x_test, onehot_y_test)
print(results)

#Predict the data test
# output = model.predict(scaled_x_test)
# print(output)

data_test = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1],[8,10,10,8,7,10,9,7,1]])

data_test = data_test.reshape(len(data_test), -1)

data_test = scaler.transform(data_test)

prediksi = model.predict(data_test)

print(prediksi)