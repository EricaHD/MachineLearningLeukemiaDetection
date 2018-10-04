from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from util import roc_results, results


def rf_training(x_train, y_train):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param y_train: the y-values that correspond to x_train (1D numpy array)
    :return: sklearn random forest classifier object that can now be used for predictions
    """
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4, max_features='log2',
                                 min_samples_split=25, min_samples_leaf=7, bootstrap=True, random_state=0)
    clf.fit(x_train, y_train)
    return clf


def rf_classification(clf, x_test):
    """
    :param clf: trained sklearn random forest classifier object
    :param x_test: the x-values we want to get predictions on (2D numpy array)
    :return: a 1D numpy array containing the predictions
    """
    return clf.predict(x_test)


def rf_pipeline(x_train, y_train, x_test, y_test):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param x_test: the y-values that correspond to x_train (1D numpy array)
    :param y_train: the x-values we want to test on (2D numpy array)
    :param y_test: the y-values that correspond to x_test (1D numpy array)
    :return: the roc auc score
    """
    clf = rf_training(x_train, y_train)
    y_pred = rf_classification(clf, x_test)
    roc_results(y_pred, y_test, 'Random Forest')
    return roc_auc_score(y_test, y_pred), results(y_pred, y_test)
