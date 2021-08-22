import tensorflow as tf
import numpy as np
import data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


LABELS = {0: "WITH MASK", 1: "WITHOUT MASK"}
model = tf.keras.models.load_model("model.h5")


def new_model(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(200, (3, 3), activation="relu",
                                     input_shape=(data.IMG_SIZE, data.IMG_SIZE, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(100, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=5)
    model.save("model.h5")
    return model


if __name__ == "__main__":
    x, y = data.data_preprocessing()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    model = new_model(x_train, y_train, x_test, y_test)
    pred = model.predict(x_test)
    for i in range(10):
        plt.figure(i)
        plt.imshow(x_test[i])
        plt.title(f"Predict: {np.argmax(pred[i])}")
        plt.xlabel(f"Actual: {y_test[i]}")

    plt.show()
