import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class CNN:
    def __init__(
        self,
        epochs: int,
        learning_rate: tf.float32,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        x_test: tf.Tensor,
        y_test: tf.Tensor,
        file_name: str = None,
    ) -> None:
        self.__epochs: int = epochs
        self.__learning_rate: tf.float32 = learning_rate
        self.__x_train: tf.Tensor = x_train
        self.__y_train: tf.Tensor = y_train
        self.__x_test: tf.Tensor = x_test
        self.__y_test: tf.Tensor = y_test
        self.__file_name: str = file_name

    def __create_model(self) -> None:
        self.__model = tf.keras.models.Sequential()

        self.__model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', dtype=tf.float32, input_shape=(28,28,1)))
        self.__model.add(tf.keras.layers.Dropout(0.2))
        self.__model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), dtype=tf.float32))
        self.__model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), activation='relu', dtype=tf.float32))
        self.__model.add(tf.keras.layers.Dropout(0.1))
        self.__model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), dtype=tf.float32))
        self.__model.add(tf.keras.layers.Flatten())
        self.__model.add(tf.keras.layers.Dense(units=64, activation='relu', dtype=tf.float32))
        self.__model.add(tf.keras.layers.Dropout(0.2))
        self.__model.add(tf.keras.layers.Dense(units=32, activation='relu', dtype=tf.float32))
        self.__model.add(tf.keras.layers.Dropout(0.1))
        self.__model.add(tf.keras.layers.Dense(units=10, activation='softmax', dtype=tf.float32))

        self.__model.compile(self.__optimizer, self.__loss, self.__metric)

    def __set_params(self) -> None:
        self.__optimizer = tf.optimizers.Adam(learning_rate=self.__learning_rate)
        self.__loss = tf.keras.losses.CategoricalCrossentropy()
        self.__metric = tf.keras.metrics.CategoricalAccuracy()

    def __train(self) -> None:
        self.__model.fit(self.__x_train, self.__y_train, batch_size=2048, epochs=self.__epochs, validation_split=0.2)

    def __test(self) -> None:
        print(self.__model.evaluate(self.__x_test, self.__y_test, batch_size=256))

    def run(self) -> None:
        self.__set_params()
        self.__create_model()
        self.__train()
        self.__test()
        if self.__file_name != None:
            self.__save_model()

    def __save_model(self) -> None:
        self.__model.save(self.__file_name)


def main():
    train_data = pd.read_csv('train_mnist.csv')
    test_data = pd.read_csv('test_mnist.csv')
    y_train = tf.keras.utils.to_categorical(tf.convert_to_tensor(train_data.pop('target').values, dtype=tf.uint8), 10)
    y_test = tf.keras.utils.to_categorical(tf.convert_to_tensor(test_data.pop('target').values, dtype=tf.uint8), 10)
    x_train = tf.reshape(tf.convert_to_tensor(train_data.values, dtype=tf.uint8), shape=(-1,28,28))
    x_test = tf.reshape(tf.convert_to_tensor(test_data.values, dtype=tf.uint8), shape=(-1,28,28))
    cnn = CNN(100, 0.001, x_train, y_train, x_test, y_test, file_name='mnist_28_classification')
    cnn.run()

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    with tf.device("/device:GPU:0"):
        main()
