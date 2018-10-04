import plot
from binary_classification.svm import svm_pipeline
from binary_classification.cart import cart_pipeline
from binary_classification.boosted_tree import xgb_pipeline
from binary_classification.random_forests import rf_pipeline
from conditional_probability.logistic import lr_pipeline
from conditional_probability.naive_bayes import nb_pipeline
from preprocessing import preprocessing, get_train_and_test, standardize_features

import warnings
warnings.filterwarnings("ignore")


def pipeline():
    """
    This function acts as a pipeline and calls the needed functions before
    any actual machine learning occurs.
    """
    x_values, y_values = preprocessing()
    x_train, x_test, y_train, y_test = get_train_and_test(x_values, y_values)
    x_train, x_test = standardize_features(x_train, x_test)

    print("                AUC              Accuracy")
    print("SVM:  ", svm_pipeline(x_train, y_train, x_test, y_test))
    print("CART: ", cart_pipeline(x_train, y_train, x_test, y_test))
    print("XGB:  ", xgb_pipeline(x_train, y_train, x_test, y_test))
    print("RF:   ", rf_pipeline(x_train, y_train, x_test, y_test))
    print("LOG:  ", lr_pipeline(x_train, y_train, x_test, y_test))
    print("NB:   ", nb_pipeline(x_train, y_train, x_test, y_test))
    plot.show_data()


pipeline()
