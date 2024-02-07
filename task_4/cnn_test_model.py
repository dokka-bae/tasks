import tensorflow as tf
import pandas as pd
import numpy as np


def main() -> None:
    test_mnist = pd.read_csv('test_mnist.csv')
    y = test_mnist.pop('target')
    x_test = tf.reshape(tf.convert_to_tensor(test_mnist, dtype=tf.float32), shape=(-1,28,28))
    mnist_classifier: tf.keras.models.Sequential = tf.keras.models.load_model('mnist_28_classification')
    y_pred = []
    y_pred.append(mnist_classifier.predict(x_test))
    y_pred = np.array(y_pred)
    print(y_pred.shape)
    for i in range(40):
        print(np.argmax(y_pred[0][i]), y[i])


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    with tf.device('/device:GPU:0'):
        main()