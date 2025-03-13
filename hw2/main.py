import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[np.float_],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        new_inputs = np.concatenate((inputs, np.ones(inputs.shape[0]).reshape(-1, 1)), axis=1)
        targets = targets.reshape(-1, 1)
        split = inputs.shape[-1]
        w = np.random.normal(0, 1, split + 1)
        self.weights = w[:split]
        self.intercept = w[split]
        for i in range(self.num_iterations):
            pred, _ = self.predict(inputs)
            gradient = np.sum((pred - targets) * new_inputs, axis=0)
            w = w - self.learning_rate * gradient
            self.weights = w[:split]
            self.intercept = w[split]

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        pred_prob = self.sigmoid((np.sum(self.weights * inputs, axis=1) + self.intercept)).reshape(-1, 1)
        pred_class = np.array([1 if prob > 0.5 else 0 for prob in pred_prob])
        return pred_prob, pred_class

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-x))


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[np.float_],
        targets: t.Sequence[int],
    ) -> None:
        self.m0 = np.mean(inputs[targets == 0], axis=0)
        self.m1 = np.mean(inputs[targets == 1], axis=0)
        self.sb = (self.m1 - self.m0).reshape(-1, 1) @ (self.m1 - self.m0).reshape(1, -1)
        sw = 0
        cov = None
        for i in range(targets.shape[0]):
            if targets[i] == 0:
                cov = (inputs[i] - self.m0).reshape(-1, 1) @ (inputs[i] - self.m0).reshape(1, -1)
            else:
                cov = (inputs[i] - self.m1).reshape(-1, 1) @ (inputs[i] - self.m1).reshape(1, -1)
            sw += cov
        self.sw = sw
        w = np.linalg.inv(sw) @ (self.m1 - self.m0).reshape(-1, 1)
        self.w = w / np.linalg.norm(w)
        self.slope = self.w[1] / self.w[0]

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Sequence[t.Union[int, bool]]:
        proj_m0 = self.w.T @ self.m0.reshape(-1, 1)
        proj_m1 = self.w.T @ self.m1.reshape(-1, 1)
        pred = np.array([1 if abs(self.w.T @ x.reshape(-1, 1) - proj_m1) < abs(self.w.T @ x.reshape(-1, 1) - proj_m0)
                        else 0 for x in inputs])
        return pred

    def plot_projection(self, inputs: npt.NDArray[np.float_]):
        plt.figure(figsize=(8, 6))
        x = np.linspace(-2, 2)
        y = self.slope * x
        plt.plot(x, y)

        pred_class = self.predict(inputs)
        plt.scatter(inputs[pred_class == 0][:, 0], inputs[pred_class == 0][:, 1], c='r')
        plt.scatter(inputs[pred_class == 1][:, 0], inputs[pred_class == 1][:, 1], c='b')
        proj_input = (inputs @ self.w) * self.w.T
        proj_1 = proj_input[pred_class == 0]
        proj_2 = proj_input[pred_class == 1]
        plt.scatter(proj_1[:, 0], proj_1[:, 1], c='r')
        plt.scatter(proj_2[:, 0], proj_2[:, 1], c='b')

        for i in range(inputs.shape[0]):
            plt.plot([inputs[i, 0], proj_input[i, 0]], [inputs[i, 1], proj_input[i, 1]], color='slategrey', alpha=0.5)


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds[:, 0])


def accuracy_score(y_trues, y_preds):
    return float(np.sum(y_trues == y_preds) / y_trues.shape[0])


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-3,  # You can modify the parameters as you want
        num_iterations=1500,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_pred_classes = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_test)
    plt.title(label=f'Projection Line: w={FLD_.slope[0]}, b=0 (Testing data)')
    plt.savefig('record1.jpg')
    plt.show()
    FLD_.plot_projection(x_train)
    plt.title(label=f'Projection Line: w={FLD_.slope[0]}, b=0 (Training data)')
    plt.savefig('record2.jpg')
    plt.show()


if __name__ == '__main__':
    main()
