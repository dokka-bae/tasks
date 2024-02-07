import tensorflow as tf
import pandas as pd


def main():
    '''
    Тест модели
    '''
    model: tf.keras.models.Sequential = tf.keras.models.load_model('fnn_model')
    test_data_frame = pd.read_csv("test_data.csv", encoding='utf-8', sep=';')
    y_true = test_data_frame.pop("open_account_flg")
    y_true = tf.Tensor = tf.one_hot(y_true, depth=len(y_true.unique()), dtype=tf.int64)
    test_data: tf.Tensor = tf.convert_to_tensor(test_data_frame.values)
    print(model.evaluate(test_data, y_true))


if __name__ == '__main__':
    main()
