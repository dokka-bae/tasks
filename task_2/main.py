import spacy
import numpy as np

from typing import List
from string import punctuation

from spacy.lang.en import STOP_WORDS

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


def preprocessing(file_name: str, NLP: spacy.Language) -> List:
    '''
        Очищаем текст от стоп слов, пунктуации и приводи все к базе(лемматизация)
    '''
    with open(file_name, encoding="utf-8") as f:
        text_file = f.read()

    text_file = NLP(text_file)

    text_lemmatized = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in text_file]

    text_cleared = [word for word in text_lemmatized if word not in punctuation and word not in STOP_WORDS]

    return text_cleared


def vectorizing(text: List[str], NLP: spacy.Language) -> np.ndarray:
    '''Берем вектора'''
    array = []
    for word in text:
        array.append(NLP(word).vector)
    return np.array(array)


def graph_words(pca_matrix: List[List[float]], text: List[str], variance_ratio: np.ndarray) -> None:
    '''Визуализация слов в 2D'''
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_matrix[:, 0], pca_matrix[:, 1])
    for i, word in enumerate(text):
        plt.annotate(word, (pca_matrix[i, 0], pca_matrix[i, 1]))
    plt.show()


def graph_variance(matrix: np.ndarray, n_components: int) -> None:
    '''График падения объема информации'''
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    explained_variance_ratio = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    plt.bar(range(1, n_components + 1), explained_variance_ratio)
    plt.xlabel('Кол-во компонент')
    plt.ylabel('Процент дисперсии')
    plt.show()


def main() -> None:
    NLP: spacy.Language = spacy.load('en_core_web_lg')
    text: List[str] = preprocessing("text.txt", NLP)
    matrix: np.ndarray = vectorizing(text, NLP)
    print(matrix.shape)

    pca = PCA(n_components=2, random_state=23)
    pca_matrix: np.ndarray = pca.fit_transform(matrix)
    print("Матрица PCA:\n", pca_matrix)

    tsne = TSNE(n_components=2, random_state=23)
    tsne_matrix: np.ndarray = tsne.fit_transform(matrix)
    print("Матрица TSNE:\n", tsne_matrix)

    graph_words(pca_matrix, text, pca.explained_variance_ratio_)
    graph_variance(matrix, n_components=10)


if __name__ == '__main__':
    main()
