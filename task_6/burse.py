import tensorflow as tf
import pandas as pd
import numpy as np

class BurseAgent:
    def __init__(
            self,
            epochs: int,
            learning_rate: tf.float32,
            x_train: tf.Tensor,
            y_train: tf.Tensor,
            x_test: tf.Tensor,
            y_test: tf.Tensor,
            ) -> None:
        self.__epochs: int = epochs
        self.__learning_rate: tf.float32 = learning_rate
        self.__x_train: tf.Tensor = x_train
        self.__y_train: tf.Tensor = y_train
        self.__x_test: tf.Tensor = x_test
        self.__y_test: tf.Tensor = y_test


    def __set_paramas(self) -> None:
        self.__optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)
        self.__metric = tf.keras.metrics.MeanAbsoluteError()
        self.__loss = tf.keras.losses.MeanSquaredError()


    def __create_model(self) -> None:
        self.__model: tf.keras.models.Sequential = tf.keras.models.Sequential()

        self.__model.add(tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu' ,input_shape=(12, 4)))
        self.__model.add(tf.keras.layers.MaxPool1D(2))
        self.__model.add(tf.keras.layers.LSTM(128, return_sequences=True))
        self.__model.add(tf.keras.layers.Dropout(0.25))
        self.__model.add(tf.keras.layers.LSTM(64, return_sequences=False))
        self.__model.add(tf.keras.layers.Dropout(0.2))
        self.__model.add(tf.keras.layers.Dense(units=1, activation='linear'))

        self.__model.compile(self.__optimizer, self.__loss, self.__metric)


    def __train_model(self) -> None:
        self.__model.fit(self.__x_train, self.__y_train, batch_size=1024, epochs=self.__epochs)


    def __test_model(self) -> None:
        print(self.__model.evaluate(self.__x_test, self.__y_test, batch_size=64))


    def __save_model(self) -> None:
        tf.keras.models.save_model(self.__model, 'model_2')


    def run(self) -> None:
        self.__set_paramas()
        self.__create_model()
        self.__train_model()
        self.__test_model()
        self.__save_model()


def main() -> None:
    x_train: tf.Tensor = tf.convert_to_tensor(np.load('x_train.npy'))
    y_train: tf.Tensor = tf.convert_to_tensor(np.load('y_train.npy'))
    x_test: tf.Tensor= tf.convert_to_tensor(np.load('x_test.npy'))
    y_test: tf.Tensor= tf.convert_to_tensor(np.load('y_test.npy'))
    agent = BurseAgent(1000, 0.01, x_train, y_train, x_test, y_test)
    agent.run()


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    with tf.device('/device:GPU:0'):
        main()