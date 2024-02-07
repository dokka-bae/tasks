from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from scipy.stats import normaltest
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def graphs(data: pd.DataFrame) -> None:
    '''shows graphs. other features are categorical or just insipid '''
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    sns.boxplot(ax=axes[0], x=data['credit_sum'])
    axes[0].set_title('credit_sum')

    sns.boxplot(ax=axes[1], x=data['score_shk'])
    axes[1].set_title('score_shk')

    sns.boxplot(ax=axes[2], x=data['monthly_income'])
    axes[2].set_title('monthly_income')
    
    plt.suptitle('outliers')
    plt.show()

    # sns.boxplot(x=data['credit_sum'])
    # plt.title('credit_sum')
    # plt.show()

    # sns.boxplot(x=data['score_shk'])
    # plt.title('score_shk')
    # plt.show()

    # sns.boxplot(x=data['monthly_income'])
    # plt.title('monthly_income')
    # plt.show()


def scaling(data: pd.DataFrame) -> List[pd.DataFrame]:
    '''
    Outliers peeked from boxplots -> (credit_sum, score_shk, montly_income).\n
    scaler = QuantileTransformer yielded the best results\n
    returns tuple of train&test datasets.
    '''
    np.random.seed(23)
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=23)
    target_columns = ['credit_sum', 'score_shk', 'monthly_income']
    scalers = {column: QuantileTransformer() for column in target_columns}
    for column, scaler in scalers.items():
        train_data[column] = scaler.fit_transform(train_data[[column]])
        test_data[column] = scaler.transform(test_data[[column]])

    return [train_data,test_data]


def correlation(data: pd.DataFrame) -> None:
    '''pearson's correlation'''
    correlation_matrix = data.corr(method='pearson')
    print(correlation_matrix.iloc[:, :len(data.columns)//2])
    print(correlation_matrix.iloc[:, len(data.columns)//2:])


def dtype_conversion_unit8(data: pd.DataFrame):
    '''
    converting both int32&int64 types is into uint8\n
    for the purpose of memory savings
    '''
    for column in data.columns:
        if data[column].dtype == 'int32' or data[column].dtype == 'int64':
            data[column] = data[column].astype(np.uint8)
        if column in ["age", "credit_count", "overdue_credit_count", "open_account_flg"]: # потому что они почему-то заданы как float, хотя на деле все являются целочисленными
            data[column] = data[column].astype(np.uint8)
    print('completed uint8 conversion')


def encode_objects(data: pd.DataFrame) -> None:
    '''
    label encoding for data with dtype object
    '''
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])
    print('completed encoding objects')


def to_float(data: pd.DataFrame) -> None:
    '''
    conversion numbers - 0,1234 into a standard - 0.1234
    '''
    data['credit_sum'] = data['credit_sum'].apply(
        lambda x: np.float64(x.replace(',', '.')))
    data['score_shk'] = data['score_shk'].apply(
        lambda x: np.float64(x.replace(',', '.')))
    print('completed float conversion')


def encode_regions(data: pd.DataFrame) -> None:
    '''
    encoding regions by data regions.csv file\n
    if pattern has less than a 75% match with an object, the row is removed 
    '''
    regions = pd.read_csv('regions.csv', encoding='utf-8',
                          sep=',', index_col='id')
    unique_values = data['living_region'].unique()
    for pattern in unique_values:
        match = process.extractOne(
            query=pattern, choices=regions['region'], scorer=fuzz.WRatio)
        if match[1] < 75:
            data.drop(data[data['living_region'] ==
                      pattern].index, inplace=True)
            continue
        data['living_region'].replace(
            pattern, np.uint8(regions.loc[regions['region'] == match[0]].index[0]), inplace=True)
    print('completed region encoding')


def main() -> None:
    data = pd.read_csv('credit_train.csv', encoding='cp1251', sep=';')
    data.drop('client_id', inplace=True, axis=1)
    data.reset_index(drop=True, inplace=True)  # хочу свои индексы
    data.info()  # типы данных в колонках

    '''
    строки с пустыми ячейками выбрасываем. не вижу смысла сначала делить выборку,\n
    а потом обрабатывать пропуски по той причине, что я никак не изменяю данные.
    '''
    data.dropna(inplace=True, axis=0)
    to_float(data)  # коммент в функции
    graphs(data)
    encode_regions(data)  # аналогично
    encode_objects(data)  # аналогично
    dtype_conversion_unit8(data)  # аналогично

    print(data.describe())  # статистика
    print(data.head())

    correlation(data)
    print(normaltest(data)) # на выходе значения stats и pvalue
    train_data, test_data = scaling(data)


    train_data.to_csv('train_data.csv', index=False, encoding='utf-8', sep=';')
    test_data.to_csv('test_data.csv', index=False, encoding='utf-8', sep=';')


if __name__ == '__main__':
    main()
