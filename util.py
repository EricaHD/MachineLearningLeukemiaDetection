import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def results(y_pred, y_test):
    """
    :param y_pred: the predicted y-values
    :param y_test: the actual y-values
    :return: the number of correct predictions, incorrect predictions, and the percent correct
    """
    num_right = 0
    num_wrong = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            num_right += 1
        else:
            num_wrong += 1
    return num_right/(num_right + num_wrong)


def roc_results(y_pred, y_test, model_type):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr,
             lw=lw, label=f'{model_type} (AUC = {round(auc(fpr, tpr), 3)})')
    plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
