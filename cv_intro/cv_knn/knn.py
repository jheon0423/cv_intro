import numpy as np
from scipy.spatial.distance import cdist


class KNNClassifier:
    supported_metrics = ("euclidean", "manhattan", "cosine")

    def __init__(self, metric="euclidean"):
        if metric not in self.supported_metrics:
            raise ValueError(f"Unsupported metric: {metric}")
        self.metric = metric
        self.x_train = None
        self.y_train = None
        self.classes = None
        self.y_indices = None
        self.train_squared_norms = None
        self.train_norms = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Expected x shape (n_samples, n_features) and y shape (n_samples,)")
        if len(y) == 0:
            raise ValueError("Training set cannot be empty")
        self.x_train = np.ascontiguousarray(x)
        self.y_train = y
        self.classes, self.y_indices = np.unique(y, return_inverse=True)
        self.train_squared_norms = np.einsum("ij,ij->i", self.x_train, self.x_train)
        self.train_norms = np.sqrt(self.train_squared_norms)
        return self

    def _distance_matrix(self, queries):
        if self.metric == "manhattan":
            return cdist(queries, self.x_train, metric="cityblock")

        similarities = queries @ self.x_train.T
        if self.metric == "euclidean":
            query_squared_norms = np.einsum("ij,ij->i", queries, queries)
            similarities *= -2.0
            similarities += query_squared_norms[:, None]
            similarities += self.train_squared_norms[None, :]
            np.maximum(similarities, 0.0, out=similarities)
            np.sqrt(similarities, out=similarities)
            return similarities

        query_norms = np.linalg.norm(queries, axis=1)
        denominators = query_norms[:, None] * self.train_norms[None, :]
        np.divide(similarities, denominators, out=similarities, where=denominators != 0)
        similarities[denominators == 0] = 0.0
        np.subtract(1.0, similarities, out=similarities)
        np.maximum(similarities, 0.0, out=similarities)
        return similarities

    def predict_many(self, x, k_values, batch_size=128):
        if self.x_train is None:
            raise ValueError("Call fit before predict")
        queries = np.asarray(x, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.x_train.shape[1]:
            raise ValueError("Query feature dimensions must match the training set")
        k_values = sorted(set(int(k) for k in k_values))
        if not k_values or k_values[0] < 1 or k_values[-1] > len(self.y_train):
            raise ValueError("Each k must be in [1, number of training samples]")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        predictions = {k: np.empty(len(queries), dtype=self.classes.dtype) for k in k_values}
        max_k = k_values[-1]
        selected_ks = set(k_values)

        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            distances = self._distance_matrix(np.ascontiguousarray(queries[start:end]))
            neighbor_indices = np.argpartition(distances, kth=max_k - 1, axis=1)[:, :max_k]
            neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
            order = np.argsort(neighbor_distances, axis=1, kind="stable")
            neighbor_indices = np.take_along_axis(neighbor_indices, order, axis=1)
            neighbor_labels = self.y_indices[neighbor_indices]
            votes = np.zeros((end - start, len(self.classes)), dtype=np.int32)
            rows = np.arange(end - start)

            for position in range(max_k):
                votes[rows, neighbor_labels[:, position]] += 1
                k = position + 1
                if k in selected_ks:
                    predictions[k][start:end] = self.classes[np.argmax(votes, axis=1)]

        return predictions

    def predict(self, x, k=5, batch_size=128):
        return self.predict_many(x, [k], batch_size=batch_size)[int(k)]
