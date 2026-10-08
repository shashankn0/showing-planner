"""Small trainable models that sit on top of frozen photo embeddings.

Fitting uses scikit-learn; the fitted result is kept as plain NumPy arrays so that
  * prediction needs only NumPy, and
  * models are saved as .npz + JSON, never pickle (loading a pickle can run arbitrary code).

Condition is an *ordered* scale (1 worst .. 5 best), so errors should cost more the further off
they are. Two ordinal approaches are provided and compared on your labels:

  RidgeThresholdHead  regress a number, then learn cut points that turn it into 1..5
  CornLinearHead      CORN-style chain of "is it better than level k, given it is at least k?"
                      linear classifiers; rank-consistent by construction (Shi et al. 2021,
                      arXiv 2111.08851) and gives a probability for each level
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vision.config import ROOMS
from vision.metrics import quadratic_weighted_kappa

ROOM_SCALE = 0.5  # weight of the room-type one-hot next to a unit-length embedding


def make_features(
    embeddings: np.ndarray, rooms: list[str], scale: float = ROOM_SCALE
) -> np.ndarray:
    onehot = np.zeros((len(rooms), len(ROOMS)), dtype=np.float32)
    for i, r in enumerate(rooms):
        onehot[i, ROOMS.index(r) if r in ROOMS else ROOMS.index("other")] = 1.0
    return np.hstack([np.asarray(embeddings, dtype=np.float32), scale * onehot])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def fit_thresholds(pred: np.ndarray, y: np.ndarray) -> list[float]:
    """Cut points between levels that maximize quadratic weighted kappa on (out-of-fold) predictions."""
    cuts = [1.5, 2.5, 3.5, 4.5]

    def score(c: list[float]) -> float:
        k = quadratic_weighted_kappa(y, np.digitize(pred, c) + 1)
        return -1.0 if np.isnan(k) else k

    best = score(cuts)
    for _ in range(3):
        for i in range(4):
            lo = cuts[i - 1] + 0.02 if i else 0.8
            hi = cuts[i + 1] - 0.02 if i < 3 else 5.2
            for cand in np.linspace(lo, hi, 41):
                trial = [*cuts[:i], float(cand), *cuts[i + 1 :]]
                s = score(trial)
                if s > best + 1e-9:
                    best, cuts = s, trial
    return cuts


# ------------------------------------------------------------------ condition heads


@dataclass
class RidgeThresholdHead:
    kind: str = "ridge"
    alpha: float = 0.1
    thresholds: list[float] = field(default_factory=lambda: [1.5, 2.5, 3.5, 4.5])
    coef: np.ndarray | None = None
    intercept: float = 0.0
    boot_coef: np.ndarray | None = None  # (B, d)
    boot_intercept: np.ndarray | None = None  # (B,)

    def fit(
        self, X: np.ndarray, y: np.ndarray, seed: int = 0, n_boot: int = 15
    ) -> RidgeThresholdHead:
        from sklearn.linear_model import Ridge, RidgeCV
        from sklearn.model_selection import KFold, cross_val_predict

        y = np.asarray(y, dtype=float)
        n = len(y)
        if n >= 15:
            self.alpha = float(RidgeCV(alphas=np.logspace(-3, 1, 9)).fit(X, y).alpha_)
        model = Ridge(alpha=self.alpha).fit(X, y)
        self.coef, self.intercept = model.coef_.astype(np.float32), float(model.intercept_)
        if n >= 15:
            oof = cross_val_predict(
                Ridge(alpha=self.alpha), X, y, cv=KFold(5, shuffle=True, random_state=seed)
            )
        else:
            oof = model.predict(X)
        self.thresholds = fit_thresholds(np.clip(oof, 1, 5), y)
        rng = np.random.default_rng(seed)
        boots = [
            Ridge(alpha=self.alpha).fit(X[idx], y[idx])
            for idx in (rng.integers(0, n, n) for _ in range(n_boot))
        ]
        self.boot_coef = np.stack([m.coef_ for m in boots]).astype(np.float32)
        self.boot_intercept = np.array([m.intercept_ for m in boots], dtype=np.float32)
        return self

    def _raw(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32) @ self.coef + self.intercept

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self._raw(X), 1.0, 5.0)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        return np.digitize(self._raw(X), self.thresholds) + 1

    def confidence(self, X: np.ndarray) -> np.ndarray:
        """1 when bootstrap models agree, falling to 0 when they disagree by a full point."""
        if self.boot_coef is None:
            return np.full(len(X), 0.5)
        preds = np.asarray(X, dtype=np.float32) @ self.boot_coef.T + self.boot_intercept
        return np.clip(1.0 - preds.std(axis=1), 0.0, 1.0)

    def to_state(self) -> tuple[dict, dict[str, np.ndarray]]:
        meta = {"kind": self.kind, "alpha": self.alpha, "thresholds": self.thresholds,
                "intercept": self.intercept}  # fmt: skip
        return meta, {"coef": self.coef, "boot_coef": self.boot_coef,
                      "boot_intercept": self.boot_intercept}  # fmt: skip

    @classmethod
    def from_state(cls, meta: dict, arrays: dict[str, np.ndarray]) -> RidgeThresholdHead:
        return cls(alpha=meta["alpha"], thresholds=list(meta["thresholds"]), coef=arrays["coef"],
                   intercept=meta["intercept"], boot_coef=arrays["boot_coef"],
                   boot_intercept=arrays["boot_intercept"])  # fmt: skip


@dataclass
class CornLinearHead:
    kind: str = "corn_linear"
    W: np.ndarray | None = None  # (4, d) weights of the four "better than level k?" steps
    b: np.ndarray | None = None  # (4,)
    const: np.ndarray | None = (
        None  # (4,) fixed probability for steps with too little data, else nan
    )

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> CornLinearHead:
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        d = X.shape[1]
        self.W, self.b = np.zeros((4, d), np.float32), np.zeros(4, np.float32)
        self.const = np.full(4, np.nan, np.float32)
        for i, k in enumerate((1, 2, 3, 4)):
            sub = y >= k
            target = (y[sub] > k).astype(int)
            if sub.sum() >= 6 and target.min() != target.max():
                clf = LogisticRegression(C=30.0, max_iter=2000, random_state=seed).fit(
                    X[sub], target
                )
                self.W[i], self.b[i] = clf.coef_[0], clf.intercept_[0]
            else:  # too little data for this step: smoothed base rate
                self.const[i] = (target.sum() + 1) / (len(target) + 2) if len(target) else 0.5
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(level = 1..5) per row, from the chain P(>1), P(>2 | >=2), ..."""
        X = np.asarray(X, dtype=np.float32)
        cond = _sigmoid(X @ self.W.T + self.b)
        fixed = ~np.isnan(self.const)
        cond[:, fixed] = self.const[fixed]
        gt = np.cumprod(cond, axis=1)  # P(y>1), P(y>2), P(y>3), P(y>4)
        probs = np.empty((len(X), 5))
        probs[:, 0] = 1 - gt[:, 0]
        probs[:, 1:4] = gt[:, :3] - gt[:, 1:4]
        probs[:, 4] = gt[:, 3]
        return np.clip(probs, 0.0, 1.0)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.predict_proba(X) @ np.arange(1, 6), 1.0, 5.0)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1) + 1

    def confidence(self, X: np.ndarray) -> np.ndarray:
        p = np.maximum(self.predict_proba(X), 1e-12)
        return 1.0 - (-(p * np.log(p)).sum(axis=1)) / np.log(5)

    def to_state(self) -> tuple[dict, dict[str, np.ndarray]]:
        return {"kind": self.kind}, {"W": self.W, "b": self.b, "const": self.const}

    @classmethod
    def from_state(cls, meta: dict, arrays: dict[str, np.ndarray]) -> CornLinearHead:
        return cls(W=arrays["W"], b=arrays["b"], const=arrays["const"])


CONDITION_HEADS = {"ridge": RidgeThresholdHead, "corn_linear": CornLinearHead}


def condition_head_from_state(meta: dict, arrays: dict[str, np.ndarray]):  # noqa: ANN201
    return CONDITION_HEADS[meta["kind"]].from_state(meta, arrays)


# ------------------------------------------------------------------ room type and flags


@dataclass
class RoomClassifier:
    classes: list[str] = field(default_factory=list)
    coef: np.ndarray | None = None  # (C, d), or (1, d) when only two room types were seen
    intercept: np.ndarray | None = None

    def fit(self, X: np.ndarray, rooms: list[str], seed: int = 0) -> RoomClassifier:
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=30.0, max_iter=2000, class_weight="balanced", random_state=seed)
        m.fit(X, rooms)
        self.classes = list(m.classes_)
        self.coef, self.intercept = m.coef_.astype(np.float32), m.intercept_.astype(np.float32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilities in ROOMS order (0 for room types never seen in training)."""
        z = np.asarray(X, dtype=np.float32) @ self.coef.T + self.intercept
        if len(self.classes) == 2:
            p1 = _sigmoid(z[:, 0])
            raw = np.stack([1 - p1, p1], axis=1)
        else:
            e = np.exp(z - z.max(axis=1, keepdims=True))
            raw = e / e.sum(axis=1, keepdims=True)
        out = np.zeros((len(X), len(ROOMS)))
        for j, c in enumerate(self.classes):
            out[:, ROOMS.index(c)] = raw[:, j]
        return out

    def to_state(self) -> tuple[dict, dict[str, np.ndarray]]:
        return {"classes": self.classes}, {"coef": self.coef, "intercept": self.intercept}

    @classmethod
    def from_state(cls, meta: dict, arrays: dict[str, np.ndarray]) -> RoomClassifier:
        return cls(list(meta["classes"]), arrays["coef"], arrays["intercept"])


@dataclass
class FlagClassifier:
    coef: np.ndarray | None = None  # (d,)
    intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> FlagClassifier:
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=30.0, max_iter=2000, class_weight="balanced", random_state=seed)
        m.fit(X, y)
        self.coef, self.intercept = m.coef_[0].astype(np.float32), float(m.intercept_[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(X, dtype=np.float32) @ self.coef + self.intercept)

    def to_state(self) -> tuple[dict, dict[str, np.ndarray]]:
        return {"intercept": self.intercept}, {"coef": self.coef}

    @classmethod
    def from_state(cls, meta: dict, arrays: dict[str, np.ndarray]) -> FlagClassifier:
        return cls(arrays["coef"], meta["intercept"])
