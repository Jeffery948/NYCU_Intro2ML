import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        new_X = np.concatenate((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)), axis=1)
        y = y.reshape(-1, 1)
        split = X.shape[-1]
        w = np.linalg.inv(new_X.T @ new_X) @ new_X.T @ y
        w = w.reshape(-1)
        self.weights = w[:split]
        self.intercept = w[split]

    def predict(self, X):
        return (np.sum(self.weights * X, axis=1) + self.intercept).reshape(-1, 1)


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        new_X = np.concatenate((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)), axis=1)
        y = y.reshape(-1, 1)
        split = X.shape[-1]
        self.epochs = epochs
        w = np.random.normal(0, 1, split + 1)
        self.weights = w[:split]
        self.intercept = w[split]
        losses = []
        for epoch in range(epochs):
            gradient = -2 * np.mean((y - self.predict(X)) * new_X, axis=0)
            w = w - learning_rate * gradient
            self.weights = w[:split]
            self.intercept = w[split]
            loss = compute_mse(self.predict(X), y)
            losses.append(loss)
        return losses

    def predict(self, X):
        return (np.sum(self.weights * X, axis=1) + self.intercept).reshape(-1, 1)

    def plot_learning_curve(self, losses):
        epoch = np.arange(1, self.epochs + 1)
        plt.plot(epoch, losses, color="blue", label="Train MSE loss")
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig("record.jpg")
        plt.show()


def compute_mse(prediction, ground_truth):
    prediction = prediction.reshape(-1)
    ground_truth = ground_truth.reshape(-1)
    return np.square(prediction - ground_truth).mean()


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=1.92e-4, epochs=350000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Mean prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
