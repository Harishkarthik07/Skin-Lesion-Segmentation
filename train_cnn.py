import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore


images = np.load("images.npy")


labels = np.random.randint(0, 2, size=(images.shape[0],))


train_images, test_images = images[:int(len(images)*0.8)], images[int(len(images)*0.8):]
train_labels, test_labels = labels[:int(len(labels)*0.8)], labels[int(len(labels)*0.8):]


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

model.save("cnn_model.h5")
