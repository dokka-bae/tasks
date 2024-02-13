import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import BatchNormalization, Add, Conv1D, MaxPool1D, LSTM, Dense, LeakyReLU, AveragePooling1D, Attention
from keras.regularizers import L1


class BurseAgent:
    def __init__(
            self,
            epochs: int,
            learning_rate: tf.float32,
            x_train: tf.Tensor,
            y_train: tf.Tensor,
            x_val: tf.Tensor,
            y_val: tf.Tensor,
            x_test: tf.Tensor,
            y_test: tf.Tensor,
            ) -> None:
        self.__epochs: int = epochs
        self.__learning_rate: tf.float32 = learning_rate
        self.__x_train: tf.Tensor = x_train
        self.__y_train: tf.Tensor = y_train
        self.__x_val: tf.Tensor = x_val
        self.__y_val: tf.Tensor = y_val
        self.__x_test: tf.Tensor = x_test
        self.__y_test: tf.Tensor = y_test


    def __set_paramas(self) -> None:
        self.__optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.__learning_rate,
            amsgrad=True,
            use_ema=True
            )
        self.__metric = tf.keras.metrics.MeanAbsoluteError()
        self.__loss = tf.keras.losses.MeanSquaredError()


    def __create_model(self) -> None:
        self.__input_layer = tf.keras.layers.InputLayer((12,1))

        x = Conv1D(filters=256, kernel_size=2, padding='same', activation=LeakyReLU())(self.__input_layer.output)
        x = BatchNormalization()(x)
        x = Conv1D(filters=128, kernel_size=2, padding='same', activation=LeakyReLU())(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=64, kernel_size=2, padding='same', activation=LeakyReLU())(x)
        x = BatchNormalization()(x)
        

        y = Conv1D(filters=256, kernel_size=2, padding='same', activation=LeakyReLU())(self.__input_layer.output)
        y = BatchNormalization()(y)
        y = Conv1D(filters=128, kernel_size=2, padding='same', activation=LeakyReLU())(y)
        y = BatchNormalization()(y)
        y = Conv1D(filters=64, kernel_size=2, padding='same', activation=LeakyReLU())(y)
        y = BatchNormalization()(x)


        z = Add()([x,y])


        block = Conv1D(filters=32, kernel_size=6, padding='same', activation=LeakyReLU())(z)
        block = AveragePooling1D(pool_size=2, padding='same')(block)
        block = LSTM(units=512, return_sequences=True)(block)
        block = BatchNormalization()(block)
        block = LSTM(units=256, return_sequences=True)(block)
        block = BatchNormalization()(block)
        block = LSTM(units=128, return_sequences=False)(block)
        block = BatchNormalization()(block)
        block = Dense(units=64, activation=LeakyReLU())(block)
        block = BatchNormalization()(block)
        block = Dense(units=8, activation=LeakyReLU())(block)
        block = BatchNormalization()(block)
        output = Dense(units=1, activation=LeakyReLU())(block)


        self.__model = Model(inputs = self.__input_layer.input, outputs = output)

        
        self.__model.compile(self.__optimizer, self.__loss, self.__metric)

        # self.__model.add(tf.keras.layers.Conv1D(128, kernel_size=2, padding='same', activation='LeakyReLU' ,input_shape=(12, 2)))
        # self.__model.add(tf.keras.layers.BatchNormalization())
        # self.__model.add(tf.keras.layers.Conv1D(32, kernel_size=4, activation='LeakyReLU'))
        # self.__model.add(tf.keras.layers.MaxPool1D(2))
        # self.__model.add(tf.keras.layers.LSTM(128, return_sequences=True))
        # self.__model.add(tf.keras.layers.BatchNormalization())
        # self.__model.add(tf.keras.layers.LSTM(64, return_sequences=False))
        # self.__model.add(tf.keras.layers.BatchNormalization())
        # self.__model.add(tf.keras.layers.Dense(units=1, activation='LeakyReLU'))

        # self.__model: tf.keras.models.Sequential = tf.keras.models.Sequential()


    def __train_model(self) -> None:
        self.__check_point = tf.keras.callbacks.ModelCheckpoint('model_15m_cp', monitor='val_loss', save_best_only=True, mode='min', initial_value_threshold=0.00005)
        self.__model.fit(self.__x_train, self.__y_train, batch_size=2048, epochs=self.__epochs, callbacks=[self.__check_point], validation_data=(self.__x_val, self.__y_val))


    def __test_model(self) -> None:
        print(self.__model.evaluate(self.__x_test, self.__y_test, batch_size=64))


    def __save_model(self) -> None:
        tf.keras.models.save_model(self.__model, 'model_5_15m')


    def run(self) -> None:
        self.__set_paramas()
        self.__create_model()
        self.__train_model()
        self.__test_model()
        self.__save_model()


def main() -> None:
    x_train: tf.Tensor = tf.convert_to_tensor(np.load('x_train.npy'))
    y_train: tf.Tensor = tf.convert_to_tensor(np.load('y_train.npy'))
    x_val: tf.Tensor = tf.convert_to_tensor(np.load('x_val.npy'))
    y_val: tf.Tensor = tf.convert_to_tensor(np.load('y_val.npy'))
    x_test: tf.Tensor= tf.convert_to_tensor(np.load('x_test.npy'))
    y_test: tf.Tensor= tf.convert_to_tensor(np.load('y_test.npy'))
    agent = BurseAgent(5000, 0.001, x_train, y_train, x_val, y_val, x_test, y_test)
    agent.run()


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    with tf.device('/device:GPU:0'):
        main()