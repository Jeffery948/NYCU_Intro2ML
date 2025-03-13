import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        losses_of_models = []
        x_train = torch.tensor(X_train, dtype=torch.float)
        Y_train = torch.tensor(y_train, dtype=torch.float)
        n_sample = X_train.shape[0]
        self.sample_weights = np.ones(n_sample) / n_sample
        for model in self.learners:
            losses = 0
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            sample_index = np.random.choice(n_sample, n_sample, p=self.sample_weights)
            sample_x = torch.tensor(np.array([X_train[index] for index in sample_index]), dtype=torch.float)
            sample_y = torch.tensor([y_train[index] for index in sample_index], dtype=torch.float)
            for i in range(num_epochs):
                pred_class = model(sample_x).squeeze()
                loss = entropy_loss(pred_class, sample_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            losses_of_models.append(losses)

            pred = (model(x_train).squeeze() > 0.5)
            incorrect = np.array([self.sample_weights[i] if pred[i] != Y_train[i] else 0 for i in range(n_sample)])
            error = np.sum(incorrect)
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            update = np.array([np.exp(-alpha) if pred[i] == Y_train[i] else np.exp(alpha) for i in range(n_sample)])
            self.sample_weights *= update
            self.sample_weights /= sum(self.sample_weights)

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        X = torch.tensor(X, dtype=torch.float)
        final_predictions = torch.zeros(X.shape[0])
        probs = []

        for learner, alpha in zip(self.learners, self.alphas):
            prob = learner(X).squeeze()
            probs.append(prob.detach().numpy())
            predictions = (prob > 0.5)
            final_predictions += alpha * (2 * predictions - 1)

        return (final_predictions > 0).float(), probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        feature_importance = []
        for learner, alpha in zip(self.learners, self.alphas):
            feature_weights = learner.layer.weight.data.abs().numpy()
            importance = feature_weights * alpha
            feature_importance.append(importance)
        feature_importance = np.array(feature_importance)
        feature_importance = np.sum(feature_importance, axis=0)
        return feature_importance.reshape(-1)
