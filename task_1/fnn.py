import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

class FNN:
    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> None:
        self.__epochs: int = epochs
        self.__learning_rate: tf.Variable = tf.Variable(learning_rate,dtype=tf.float32)
        self.__x_train: tf.Tensor = tf.convert_to_tensor(x_train.values)
        self.__y_train: tf.Tensor = tf.one_hot(y_train, depth=len(y_train.unique()), dtype=tf.int64)
        self.__x_test: tf.Tensor = tf.convert_to_tensor(x_test.values)
        self.__y_test: tf.Tensor = tf.one_hot(y_test, depth=len(y_test.unique()), dtype=tf.int64)
        self.__model: tf.keras.models.Sequential
        self.__run()


    def __run(self) -> None:
        self.__set_model_params()
        self.__create_model()
        self.__train()
        self.__test()
        self.__model.save("fnn_model")


    def __set_model_params(self) -> None:
        '''
        меток класса 0 в разы больше, чем класса 1(файлик classes_test.py)
        тут создается словарь с весами для классов, ибо без этого
        модель будет относить практически любой набор данных к классу 0 т.к. его в разы больше (~5.5раз)\n
        оптимизатор Adam(считается эффективным)\n
        функция стоимости - бинарная кроссэнтропия\n
        метрика - AUC\n
        '''
        self.__class_labels = np.unique(np.argmax(self.__y_train, axis=1))
        self.__class_weights = compute_class_weight('balanced', classes=self.__class_labels, y=np.argmax(self.__y_train, axis=1))
        self.__class_weight_dict = {label: weight for label, weight in zip(self.__class_labels, self.__class_weights)} 

        self.__optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)
        self.__loss = tf.keras.losses.BinaryCrossentropy()
        self.__metric = tf.keras.metrics.AUC()


    def __create_model(self) -> None:
        
        self.__model = tf.keras.models.Sequential()

        self.__model.add(tf.keras.layers.Dense(64,activation='relu',dtype=tf.float32, input_shape = (self.__x_test.shape[1],)))
        self.__model.add(tf.keras.layers.Dropout(0.5))
        self.__model.add(tf.keras.layers.Dense(32,activation='relu',dtype=tf.float32))
        self.__model.add(tf.keras.layers.Dropout(0.4))
        self.__model.add(tf.keras.layers.Dense(16,activation='relu',dtype=tf.float32))
        self.__model.add(tf.keras.layers.Dropout(0.3))
        self.__model.add(tf.keras.layers.Dense(2,activation='sigmoid',dtype=tf.float32))

        self.__model.compile(optimizer=self.__optimizer, loss=self.__loss, metrics=[self.__metric])


    def __train(self) -> None:
        self.__model.fit(x = self.__x_train, y = self.__y_train, epochs = self.__epochs, validation_split=0.2, batch_size=32768, class_weight = self.__class_weight_dict)


    def __test(self) -> None:
        print(self.__model.evaluate(self.__x_test, self.__y_test, batch_size=512))


def main():
    train_data = pd.read_csv('train_data.csv', sep=';')
    test_data = pd.read_csv('test_data.csv', sep=';')
    y_train = train_data.pop('open_account_flg')
    y_test = test_data.pop('open_account_flg')
    fnn = FNN(2000,0.001,train_data,y_train,test_data,y_test)


if __name__ == "__main__":
    '''
    заставляем tf юзать видеокарту и докручиваем ей памяти, ибо видит он лишь 4гб
    '''
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    with tf.device('/device:GPU:0'):
        main()
