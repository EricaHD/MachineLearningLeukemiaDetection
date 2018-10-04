import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv():
    """
    :return: a dataframe containing our X and Y values
    """
    aml_data = pd.read_csv('data.csv', index_col=0)
    del aml_data['DrawID']
    aml_data['caseflag'].replace({'Yes': 1, 'No': -1}, inplace=True)
    return aml_data


def fill_missing_values(aml_data):
    """
    :param aml_data: a dataframe containing our X and Y values
    :return: an updated version of aml_data with no missing values (replaced by the mean of the column)
    """
    for column in aml_data.columns:
        aml_data[column].fillna(aml_data[column].mean(), inplace=True)


def add_total_genes(aml_data):
    """
    :param aml_data: a dataframe containing our X and Y values
    :return: an updated version of aml_data containing a new column with the sum of all allele frequencies
    """
    aml_data['Total.Genes'] = 0
    for column in aml_data.columns:
        if 'Gene.' in column:
            aml_data['Total.Genes'] += aml_data[column]


def add_extra_features(aml_data):
    """
    :param aml_data: a dataframe containing our X and Y values
    :return: an updated version of aml_data containing new interaction terms
    """
    columns = ['Total.Genes', 'HEMATOCR', 'PLATELET', 'WBC', 'HEMOGLBN', 'Age']
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            aml_data[columns[i] + columns[j]] = aml_data[columns[i]] * aml_data[columns[j]]


def preprocessing(add_features=False):
    """
    :param add_features: if set to true, we add some "interaction features"
    :return: our featurized x and y values
    """
    aml_data = load_csv()
    fill_missing_values(aml_data)
    add_total_genes(aml_data)

    if add_features:
        add_extra_features(aml_data)

    y_values = aml_data['caseflag'].values
    del aml_data['caseflag']
    return aml_data.values, y_values


def get_train_and_test(x_values, y_values):
    """
    :param x_values: the featurized values of x
    :param y_values: the featurized values of y
    :return: a tuple containing x_train, x_test, y_train, and y_test
    """
    test_set_size = int(len(y_values) * 0.35)
    return train_test_split(x_values, y_values, test_size=test_set_size, random_state=8)


def standardize_features(x_train, x_test):
    """
    :param x_train: 2D numpy array of size (num_instances, num_features)
    :param x_test: 2D numpy array of size (num_instances, num_features)
    :return: a tuple containing the newly standardized values for train and test
    """
    i = 0
    while i < x_train.shape[1]:
        std = np.std(x_train[:, i])
        mean = np.mean(x_train[:, i])
        if not std:
            x_train = np.delete(x_train, i, 1)
            x_test = np.delete(x_test, i, 1)
        else:
            x_train[:, i] = (x_train[:, i] - mean)/std
            x_test[:, i] = (x_test[:, i] - mean)/std
            i += 1
    num_features = x_train.shape[1]
    return np.insert(x_train, num_features, 1, axis=1), np.insert(x_test, num_features, 1, axis=1)
