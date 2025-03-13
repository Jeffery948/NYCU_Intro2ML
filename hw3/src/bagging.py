import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Implement your code here"""
        losses_of_models = []
        n_sample = X_train.shape[0]
        sample_weights = np.ones(n_sample) / n_sample
        for model in self.learners:
            losses = 0
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            sample_index = np.random.choice(n_sample, n_sample, p=sample_weights)
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
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        X = torch.tensor(X, dtype=torch.float)
        final_predictions = torch.zeros(X.shape[0])
        probs = []

        for learner in self.learners:
            prob = learner(X).squeeze()
            probs.append(prob.detach().numpy())
            predictions = (prob > 0.5).float()
            final_predictions += predictions

        return (final_predictions > 5).float(), probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        feature_importance = []
        for learner in self.learners:
            feature_weights = learner.layer.weight.data.abs().numpy()
            feature_importance.append(feature_weights)
        feature_importance = np.array(feature_importance)
        feature_importance = np.sum(feature_importance, axis=0)
        return feature_importance.reshape(-1)
