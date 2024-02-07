import pandas as pd
import matplotlib.pyplot as plt

def main():
    '''
    График с распределением классов
    '''
    train_data = pd.read_csv('train_data.csv',encoding='utf-8', sep=';')
    test_data = pd.read_csv('test_data.csv', encoding='utf-8',sep=';')

    y_train = train_data['open_account_flg']
    y_test = test_data['open_account_flg']

    train_classes_distribution = y_train.value_counts()
    test_classes_distribution = y_test.value_counts()

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    train_classes_distribution.plot(kind='bar')
    plt.title('метки класса train_data')
    plt.xlabel('метка')
    plt.ylabel('число примеров')

    plt.subplot(1, 2, 2)
    test_classes_distribution.plot(kind='bar')
    plt.title('метки класса test_data')
    plt.xlabel('метка')
    plt.ylabel('число примеров')

    plt.show()


if __name__ =='__main__':
    main()