import torch
import numpy as np
from tqdm import tqdm


def evaluate_model(model, loader, mgpus=True):

    if mgpus:
        confusion_matrix = np.zeros((model.module._num_classes, model.module._num_classes), dtype=int)
    else:
        confusion_matrix = np.zeros((model._num_classes, model._num_classes), dtype=int)

    scalar_dict = {}

    model.eval()
    data_iter = iter(loader)

    for (xs, ys) in tqdm(data_iter):

        with torch.no_grad():
            ys = ys.cuda()
            xs = xs.cuda()
            ys_pred, _ = model.forward(xs)

        ys_pred = torch.argmax(ys_pred, dim=1)

        for y_pred, y_true in zip(ys_pred, ys):
            confusion_matrix[y_true][y_pred] += 1

    scalar_dict["accuracy"] = acc_from_cm(confusion_matrix)

    return scalar_dict


def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total
