import typing as t
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess(df: pd.DataFrame):
    """
    (TODO): Implement your preprocessing function.
    """
    class_le = LabelEncoder()
    df['person_gender'] = class_le.fit_transform(df['person_gender'])
    df['person_education'] = class_le.fit_transform(df['person_education'])
    df['person_home_ownership'] = class_le.fit_transform(df['person_home_ownership'])
    df['loan_intent'] = class_le.fit_transform(df['loan_intent'])
    df['previous_loan_defaults_on_file'] = class_le.fit_transform(df['previous_loan_defaults_on_file'])
    df = df.to_numpy()
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layers model.
    Here, for example:
        - Linear(input_dim, 1) is a single-layer model.
        - Linear(input_dim, k) -> Linear(k, 1) is a two-layer model.

    No non-linear activation allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.layer(x)
        return nn.functional.sigmoid(x)


def accuracy_score(y_trues, y_preds) -> float:
    if isinstance(y_preds, np.ndarray):
        return (y_trues == y_preds).mean()
    y_preds = y_preds.detach().numpy()
    return (y_trues == y_preds).mean()


def entropy_loss(outputs, targets):
    eps = 1e-6
    outputs = torch.clamp(outputs, eps, 1 - eps)
    loss = -torch.sum(targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs)) / outputs.shape[0]
    return loss


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    plt.figure(figsize=(10, 10))
    for y_pred in y_preds:
        fpr, tpr, thresholds = roc_curve(y_trues, y_pred)
        plt.plot(fpr, tpr, label="AUC={:.4f}".format(auc(fpr, tpr)))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.savefig(fpath)
    plt.show()
