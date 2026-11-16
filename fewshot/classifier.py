"""
classifier.py - naming an object from few labelled examples
"""

import json

import numpy as np

UNKNOWN_LABEL = "unknown"


class FewShotClassifier(object):
    def __init__(self, method="prototype", k=3, unknown_threshold=None):
        if method not in ("prototype", "knn"):
            raise ValueError("method must be 'prototype' or 'knn'")
        self.method = method
        self.k = int(k)
        self.unknown_threshold = unknown_threshold
        self.class_names = []
        self.support_embeddings = None # (S, D) every stored example
        self.support_labels = None # (S,)  class index of each example
        self.prototypes = None # (C, D) one averaged vector per class
        self.backbone_spec = None # which model produced the embeddings

    # fit
    def fit(self, embeddings, labels, class_names, backbone_spec=None):
        """Store the labelled examples.

        embeddings : (S, D) float array, L2-normalised (as Backbone.embed returns)
        labels     : length-S sequence of integer class indices
        class_names: list mapping index -> human-readable name
        """
        self.support_embeddings = np.asarray(embeddings, dtype=np.float32)
        self.support_labels = np.asarray(labels, dtype=np.int64)
        self.class_names = list(class_names)
        self.backbone_spec = backbone_spec

        # one prototype per class: mean of that class's embeddings re-normalised to length 1
        protos = []
        for c in range(len(self.class_names)):
            mean_vec = self.support_embeddings[self.support_labels == c].mean(axis=0)
            protos.append(mean_vec / max(np.linalg.norm(mean_vec), 1e-12))
        self.prototypes = np.stack(protos).astype(np.float32)
        return self

    # predict
    def class_scores(self, query_embeddings):
        """Return an (N, C) array: one score per query per class (higher = better)."""
        q = np.asarray(query_embeddings, dtype=np.float32)
        if self.method == "prototype":
            # (N, D) @ (D, C) -> (N, C) cos similarity to each prototype.
            return q @ self.prototypes.T

        # kNN
        sims = q @ self.support_embeddings.T # (N, S)
        k = min(self.k, sims.shape[1])
        # argpartition finds the k largest per row without a full sort
        top_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k] # (N, k)
        scores = np.zeros((q.shape[0], len(self.class_names)), dtype=np.float32)
        for row in range(q.shape[0]):
            for idx in top_idx[row]:
                # all neighbour vote for its class with weight = its similarity.
                scores[row, self.support_labels[idx]] += sims[row, idx]
        return scores / float(k)

    def predict(self, query_embeddings):
        """Classify each query embedding.

        Returns a list of dicts, one per query:
            label       predicted class name (or "unknown")
            class_index index into class_names (-1 for unknown)
            score       score of the winning class
            margin      winning score minus runner-up score; small = uncertain
            scores      {class_name: score} for every class
        """
        scores = self.class_scores(query_embeddings)
        results = []
        for row in scores:
            order = np.argsort(-row)
            best = int(order[0])
            best_score = float(row[best])
            margin = float(best_score - row[order[1]]) if len(order) > 1 else best_score
            is_unknown = (self.unknown_threshold is not None
                          and best_score < self.unknown_threshold)
            results.append({
                "label": UNKNOWN_LABEL if is_unknown else self.class_names[best],
                "class_index": -1 if is_unknown else best,
                "score": round(best_score, 4),
                "margin": round(margin, 4),
                "scores": {name: round(float(s), 4)
                           for name, s in zip(self.class_names, row)},
            })
        return results

    # save/load
    def save(self, path):
        """Write the enrolment to one .npz file so prediction can start instantly.

        The settings travel with the data (as a small JSON string), including
        which backbone made the embeddings. Embeddings from different models
        are NOT comparable, and load() lets callers check for a mismatch.
        """
        meta = {
            "method": self.method,
            "k": self.k,
            "unknown_threshold": self.unknown_threshold,
            "class_names": self.class_names,
            "backbone_spec": self.backbone_spec,
        }
        np.savez_compressed(
            path,
            support_embeddings=self.support_embeddings,
            support_labels=self.support_labels,
            meta=np.array(json.dumps(meta)),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        clf = cls(method=meta["method"], k=meta["k"],
                  unknown_threshold=meta["unknown_threshold"])
        clf.fit(data["support_embeddings"], data["support_labels"],
                meta["class_names"], backbone_spec=meta.get("backbone_spec"))
        return clf
