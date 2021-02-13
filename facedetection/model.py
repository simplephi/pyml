import tensorflow as tf
# print(tf.__version__)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from imutils import paths
import os

# Membuat folder untuk training dan testing dataset
try:
    os.mkdir('./data/training')
    os.mkdir('./data/testing')
    os.mkdir('./data/training/with_mask')
    os.mkdir('./data/training/without_mask')
    os.mkdir('./data/testing/with_mask')
    os.mkdir('./data/testing/without_mask')

except OSError:
    pass

# Memisahkan dan membagi data training dan data testing pada tiap-tiap label, serta membagi nya menjadi 80% data training dan 20% data testing
import random
import shutil
from shutil import copyfile

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " Jika kosong, Abaikan.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

MASK_SOURCE_DIR = "./data/with_mask/"
TRAINING_MASK_DIR = "./data/training/with_mask/"
TESTING_MASK_DIR = "./data/testing/with_mask/"

WITHOUT_SOURCE_DIR = "./data/without_mask/"
TRAINING_WITHOUT_DIR = "./data/training/without_mask/"
TESTING_WITHOUT_DIR = "./data/testing/without_mask/"

split_size = .8
split_data(MASK_SOURCE_DIR, TRAINING_MASK_DIR, TESTING_MASK_DIR, split_size)
split_data(WITHOUT_SOURCE_DIR, TRAINING_WITHOUT_DIR, TESTING_WITHOUT_DIR, split_size)

# Memeriksa hasil split(pembagian)
print(len(os.listdir('./data/training/with_mask/')))
print(len(os.listdir('./data/training/without_mask/')))
print(len(os.listdir('./data/testing/with_mask/')))
print(len(os.listdir('./data/testing/without_mask/')))


# Inisiasi datagen untuk training dan testing
train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        validation_split=0.2,
    )

# except for rescaling, no augmentations are needed for validation and testing generators
validation_gen = ImageDataGenerator(
        rescale=1.0 / 255
    )


# Menyesuaikan data image generator
TRAINING_DIR = './data/training'
train_generator = train_gen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    target_size=(224, 224))

VALIDATION_DIR = './data/testing'
validation_generator = validation_gen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=32,
                                                              class_mode='categorical',
                                                              target_size=(224, 224))


# Load the MobileNetV2 network, ensuring the head FC layer sets are left off

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model that will be placed on top of the base model

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# Place the head FC model on top of the base model (this will become the actual model we will train)
from tensorflow.keras.models import Model
model = Model(inputs=baseModel.input, outputs=headModel)


# Loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False


# Initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

# Menggunakan fungsi callback
class panggilCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.99):
            print("\n Proses Training dibatalkan jika sudah mencapai akurasi 99%")
            self.model.stop_training = True

callbacks = panggilCallback()


# Compile the model
from tensorflow.keras.optimizers import Adam

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# Training Model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // BATCH_SIZE,
                    verbose=2,
                    callbacks=[callbacks])


# Melihat hasil akurasi apakah overfitting atau underfitting
import matplotlib.pyplot as plt

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss, label='Training set')
plt.plot(val_loss, label='Test set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(accuracy, label='Training set')
plt.plot(val_accuracy, label='Test set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)


plt.show()


# Save the model

MODEL_PATH = "../models/face_mask_detector.h5"
model.save(MODEL_PATH, save_format="h5")

# Export into TensorFlow.js

import tensorflowjs as tfjs

TFJS_MODEL_DIR = "../models/tfjs"
tfjs.converters.save_keras_model(model, TFJS_MODEL_DIR)
