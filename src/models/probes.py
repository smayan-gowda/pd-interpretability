"""
probing classifiers for mechanistic interpretability.

implements linear probes to test what information is encoded in
intermediate layer representations of wav2vec2.
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score


class LinearProbe:
    """
    linear probe for classification or regression tasks.

    used to test whether specific features are linearly decodable from
    neural network representations.
    """

    def __init__(
        self,
        task: str = 'classification',
        regularization: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        args:
            task: 'classification' or 'regression'
            regularization: regularization strength (c for logistic, alpha for ridge)
            max_iter: maximum iterations for optimization
            random_state: random seed
        """
        self.task = task
        self.regularization = regularization
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler = StandardScaler()

        if task == 'classification':
            self.model = LogisticRegression(
                C=regularization,
                max_iter=max_iter,
                random_state=random_state,
                solver='liblinear'
            )
        elif task == 'regression':
            self.model = Ridge(
                alpha=regularization,
                random_state=random_state
            )
        else:
            raise ValueError(f"unknown task: {task}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        fit probe on representations.

        args:
            X: representations [n_samples, n_features]
            y: targets [n_samples]
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        make predictions.

        args:
            X: representations

        returns:
            predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        score predictions.

        args:
            X: representations
            y: targets

        returns:
            accuracy (classification) or r2 (regression)
        """
        preds = self.predict(X)

        if self.task == 'classification':
            return accuracy_score(y, preds)
        else:
            return r2_score(y, preds)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        cross-validate probe.

        args:
            X: representations
            y: targets
            groups: group labels for group-wise cv
            cv: number of folds or cv splitter

        returns:
            metrics dict with mean and std
        """
        X_scaled = self.scaler.fit_transform(X)

        if groups is not None:
            cv_splitter = LeaveOneGroupOut()
            scores = cross_val_score(
                self.model, X_scaled, y,
                cv=cv_splitter,
                groups=groups,
                scoring='accuracy' if self.task == 'classification' else 'r2'
            )
        else:
            scores = cross_val_score(
                self.model, X_scaled, y,
                cv=cv,
                scoring='accuracy' if self.task == 'classification' else 'r2'
            )

        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }


class LayerwiseProber:
    """
    probe multiple layers of a model to identify where information is encoded.

    key tool for mechanistic interpretability: reveals which layers encode
    which features.
    """

    def __init__(
        self,
        task: str = 'classification',
        regularization: float = 1.0
    ):
        """
        args:
            task: 'classification' or 'regression'
            regularization: regularization strength
        """
        self.task = task
        self.regularization = regularization
        self.probes = {}

    def fit_layer(
        self,
        layer_idx: int,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        fit probe for specific layer.

        args:
            layer_idx: layer index
            X: representations from this layer
            y: targets
            groups: group labels for cv

        returns:
            cross-validation results
        """
        probe = LinearProbe(
            task=self.task,
            regularization=self.regularization
        )

        results = probe.cross_validate(X, y, groups=groups)

        self.probes[layer_idx] = probe

        return results

    def probe_all_layers(
        self,
        activations: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        probe all layers in activation tensor.

        args:
            activations: [n_samples, n_layers, hidden_size]
            y: targets [n_samples]
            groups: group labels

        returns:
            results dict mapping layer_idx to metrics
        """
        n_layers = activations.shape[1]

        results = {}

        for layer_idx in range(n_layers):
            layer_acts = activations[:, layer_idx, :]

            results[layer_idx] = self.fit_layer(
                layer_idx, layer_acts, y, groups
            )

        return results

    def get_best_layer(self) -> Tuple[int, float]:
        """
        get layer with highest probing score.

        returns:
            (layer_idx, score)
        """
        if not self.probes:
            raise ValueError("no probes fitted yet")

        scores = {
            idx: probe.model.score(probe.scaler.transform(X), y)
            for idx, probe in self.probes.items()
        }

        best_idx = max(scores, key=scores.get)
        return best_idx, scores[best_idx]


class MultiFeatureProber:
    """
    probe multiple features across layers.

    used to create feature encoding heatmaps showing which layers encode
    which clinical features.
    """

    def __init__(
        self,
        feature_names: List[str],
        task: str = 'regression',
        regularization: float = 1.0
    ):
        """
        args:
            feature_names: list of feature names to probe
            task: 'classification' or 'regression'
            regularization: regularization strength
        """
        self.feature_names = feature_names
        self.task = task
        self.regularization = regularization

        self.results = {name: {} for name in feature_names}

    def probe_feature_across_layers(
        self,
        feature_name: str,
        activations: np.ndarray,
        feature_values: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        probe single feature across all layers.

        args:
            feature_name: name of feature being probed
            activations: [n_samples, n_layers, hidden_size]
            feature_values: target values [n_samples]
            groups: group labels

        returns:
            results dict mapping layer to metrics
        """
        valid_mask = ~np.isnan(feature_values)

        if valid_mask.sum() < 10:
            warnings.warn(f"too few valid samples for {feature_name}")
            return {}

        acts_valid = activations[valid_mask]
        vals_valid = feature_values[valid_mask]
        groups_valid = groups[valid_mask] if groups is not None else None

        n_layers = acts_valid.shape[1]
        layer_results = {}

        for layer_idx in range(n_layers):
            layer_acts = acts_valid[:, layer_idx, :]

            probe = LinearProbe(
                task=self.task,
                regularization=self.regularization
            )

            results = probe.cross_validate(
                layer_acts, vals_valid,
                groups=groups_valid
            )

            layer_results[layer_idx] = results

        self.results[feature_name] = layer_results

        return layer_results

    def probe_all_features(
        self,
        activations: np.ndarray,
        feature_matrix: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        probe all features across all layers.

        args:
            activations: [n_samples, n_layers, hidden_size]
            feature_matrix: [n_samples, n_features]
            groups: group labels

        returns:
            nested dict: feature_name -> layer_idx -> metrics
        """
        for i, feature_name in enumerate(self.feature_names):
            self.probe_feature_across_layers(
                feature_name,
                activations,
                feature_matrix[:, i],
                groups
            )

        return self.results

    def get_encoding_matrix(
        self,
        metric: str = 'mean'
    ) -> np.ndarray:
        """
        get matrix of encoding scores for visualization.

        args:
            metric: which metric to extract ('mean', 'std')

        returns:
            matrix [n_features, n_layers]
        """
        if not self.results or not self.results[self.feature_names[0]]:
            raise ValueError("no probing results available")

        n_features = len(self.feature_names)
        n_layers = len(self.results[self.feature_names[0]])

        matrix = np.zeros((n_features, n_layers))

        for i, feature in enumerate(self.feature_names):
            for layer_idx in range(n_layers):
                if layer_idx in self.results[feature]:
                    matrix[i, layer_idx] = self.results[feature][layer_idx][metric]

        return matrix


def compute_selectivity_score(
    target_accuracy: float,
    control_accuracy: float
) -> float:
    """
    compute selectivity score for probe.

    measures how much better probe performs on target task vs control task.
    high selectivity indicates probe learned task-specific features.

    args:
        target_accuracy: accuracy on target task
        control_accuracy: accuracy on control task

    returns:
        selectivity score
    """
    return max(0, target_accuracy - control_accuracy)


def permutation_test_probe(
    probe: LinearProbe,
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    test probe significance using permutation test.

    args:
        probe: fitted probe
        X: representations
        y: targets
        n_permutations: number of permutations
        random_state: random seed

    returns:
        (observed_score, p_value)
    """
    observed_score = probe.score(X, y)

    rng = np.random.RandomState(random_state)
    null_scores = []

    for _ in range(n_permutations):
        y_perm = rng.permutation(y)

        probe_perm = LinearProbe(
            task=probe.task,
            regularization=probe.regularization
        )
        probe_perm.fit(X, y_perm)

        null_scores.append(probe_perm.score(X, y_perm))

    null_scores = np.array(null_scores)

    p_value = (null_scores >= observed_score).sum() / n_permutations

    return observed_score, p_value
