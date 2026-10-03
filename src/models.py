"""CPU-friendly models: LightGBM baselines and domain-robust variants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _confidence_weights(conf: np.ndarray, use_conf: bool) -> np.ndarray:
    w = np.ones(len(conf), dtype=np.float64)
    if use_conf:
        c = np.asarray(conf, dtype=np.float64)
        # FORCE labels: 1=high confidence, 2=medium, 3=low → invert
        c = np.clip(c, 1.0, None)
        w = 1.0 / c
        w = w / (w.mean() + 1e-12)
    return w


def _well_balance_weights(wells: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Inverse well-frequency weighting (simple multi-source balance)."""
    wells = np.asarray(wells)
    _, inv, counts = np.unique(wells, return_inverse=True, return_counts=True)
    well_w = 1.0 / counts[inv]
    well_w = well_w / well_w.mean()
    out = base * well_w
    return out / (out.mean() + 1e-12)


@dataclass
class LGBMConfig:
    n_estimators: int = 180
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    # Row bagging disabled: subsample only applies when bagging_freq>0 in LightGBM.
    # Keep subsample=1.0 / bagging_freq=0 so released numbers match the reported runs
    # (an earlier 0.8 subsample without bagging_freq was inert).
    subsample: float = 1.0
    bagging_freq: int = 0
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    n_jobs: int = -1
    random_state: int = 42


class LGBMLithologyModel:
    def __init__(
        self,
        config: LGBMConfig | None = None,
        use_confidence: bool = False,
        use_well_balance: bool = False,
        groupdro_rounds: int = 0,
        groupdro_eta: float = 0.1,
        external_weights: np.ndarray | None = None,
    ):
        self.config = config or LGBMConfig()
        self.use_confidence = use_confidence
        self.use_well_balance = use_well_balance
        self.groupdro_rounds = groupdro_rounds
        self.groupdro_eta = groupdro_eta
        self.external_weights = external_weights
        self.model: lgb.LGBMClassifier | None = None
        self.classes_: np.ndarray | None = None

    def _make_model(self) -> lgb.LGBMClassifier:
        c = self.config
        return lgb.LGBMClassifier(
            n_estimators=c.n_estimators,
            learning_rate=c.learning_rate,
            num_leaves=c.num_leaves,
            min_child_samples=c.min_child_samples,
            subsample=c.subsample,
            subsample_freq=c.bagging_freq,
            colsample_bytree=c.colsample_bytree,
            reg_lambda=c.reg_lambda,
            n_jobs=c.n_jobs,
            random_state=c.random_state,
            class_weight="balanced",
            verbosity=-1,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        conf: np.ndarray,
        wells: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LGBMLithologyModel":
        self.classes_ = np.unique(y)
        w = _confidence_weights(conf, self.use_confidence)
        if self.use_well_balance:
            w = _well_balance_weights(wells, w)
        if self.external_weights is not None:
            ew = np.asarray(self.external_weights, dtype=np.float64)
            if len(ew) != len(w):
                raise ValueError("external_weights length mismatch")
            w = w * ew
            w = w / (w.mean() + 1e-12)

        if self.groupdro_rounds <= 0:
            self.model = self._make_model()
            if X_val is not None and y_val is not None:
                # Early stopping monitors multi-class log-loss on the well-held-out
                # validation split (LightGBM default for multiclass). Macro-F1 is
                # used only for final reporting, not for early stopping.
                # Prefer eval_X/eval_y (LightGBM ≥4.7); fall back to eval_set for 4.3–4.6.
                fit_kw = dict(
                    sample_weight=w,
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                try:
                    self.model.fit(X, y, eval_X=X_val, eval_y=y_val, **fit_kw)
                except TypeError:
                    self.model.fit(X, y, eval_set=[(X_val, y_val)], **fit_kw)
            else:
                self.model.fit(X, y, sample_weight=w)
            return self

        # Approximate GroupDRO: iteratively upweight wells with high error
        unique_wells = np.unique(wells)
        q = np.ones(len(unique_wells), dtype=np.float64) / len(unique_wells)
        well_to_i = {w_: i for i, w_ in enumerate(unique_wells)}

        # cheaper inner rounds, fuller final model
        inner_cfg = LGBMConfig(
            n_estimators=min(120, self.config.n_estimators),
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            bagging_freq=self.config.bagging_freq,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )
        last_model = None
        for r in range(self.groupdro_rounds):
            sample_q = np.array([q[well_to_i[w_]] for w_ in wells], dtype=np.float64)
            sw = w * sample_q
            sw = sw / (sw.mean() + 1e-12)
            cfg = self.config if r == self.groupdro_rounds - 1 else inner_cfg
            model = lgb.LGBMClassifier(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.num_leaves,
                min_child_samples=cfg.min_child_samples,
                subsample=cfg.subsample,
                subsample_freq=cfg.bagging_freq,
                colsample_bytree=cfg.colsample_bytree,
                reg_lambda=cfg.reg_lambda,
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
                class_weight="balanced",
                verbosity=-1,
            )
            model.fit(X, y, sample_weight=sw)
            pred = model.predict(X)
            losses = np.zeros(len(unique_wells), dtype=np.float64)
            for i, ww in enumerate(unique_wells):
                m = wells == ww
                losses[i] = 1.0 - float((pred[m] == y[m]).mean())
            q = q * np.exp(self.groupdro_eta * losses)
            q = q / q.sum()
            last_model = model

        self.model = last_model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(X)


class RFLithologyModel:
    """Random Forest baseline under the same feature pipeline (CPU ERM)."""

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        max_rows: int | None = 120_000,
        max_depth: int = 24,
        min_samples_leaf: int = 20,
    ):
        from sklearn.ensemble import RandomForestClassifier

        self.max_rows = max_rows
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int | None = None) -> "RFLithologyModel":
        rs = self.random_state if seed is None else seed
        if self.max_rows is not None and len(y) > self.max_rows:
            rng = np.random.default_rng(rs)
            idx = rng.choice(len(y), size=self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class XGBLithologyModel:
    """Inductive XGBoost baseline (hist) on the same global feature trunk."""

    def __init__(
        self,
        n_estimators: int = 180,
        random_state: int = 42,
        max_rows: int | None = None,
        learning_rate: float = 0.05,
        max_depth: int = 8,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_rows = max_rows
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        seed: int | None = None,
    ) -> "XGBLithologyModel":
        from collections import Counter

        from xgboost import XGBClassifier

        rs = self.random_state if seed is None else seed
        if self.max_rows is not None and len(y) > self.max_rows:
            rng = np.random.default_rng(rs)
            idx = rng.choice(len(y), size=self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        n_cls = int(len(np.unique(y)))
        # Match LightGBM class_weight=balanced intent.
        counts = Counter(int(v) for v in y)
        n = float(len(y))
        sw = np.array([n / (n_cls * counts[int(v)]) for v in y], dtype=np.float64)
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=1.0,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=n_cls,
            tree_method="hist",
            n_jobs=-1,
            random_state=rs,
            eval_metric="mlogloss",
        )
        fit_kw: dict = {"sample_weight": sw}
        if X_val is not None and y_val is not None:
            fit_kw["eval_set"] = [(X_val, y_val)]
            fit_kw["verbose"] = False
            # xgboost>=2 early stopping via fit kwargs when supported
            try:
                self.model.set_params(early_stopping_rounds=30)
            except Exception:
                pass
        self.model.fit(X, y, **fit_kw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(X)


def coral_align(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """
    Covariance-only CORAL in an already-scaled feature space.

    Whiten source second-order stats then recolour toward the target covariance
    (analytical CORAL map A). Returns ``(Xs @ A, Xt)`` unchanged for Xt.
    This is **not** dual-centering CORAL: callers typically fit a source-only
    ``StandardScaler`` first; target rows are transformed with that same scaler
    (no target re-centering / re-scaling). Inference must apply the scaler only
    and must **not** re-apply A (classifier is trained in target-covariance space).
    """

    def _cov(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        return (x.T @ x) / max(1, len(x) - 1) + eps * np.eye(x.shape[1])

    cs = _cov(Xs)
    ct = _cov(Xt)
    us, ss, _ = np.linalg.svd(cs)
    ut, st, _ = np.linalg.svd(ct)
    ss_sqrt_inv = us @ np.diag(1.0 / np.sqrt(ss + eps)) @ us.T
    st_sqrt = ut @ np.diag(np.sqrt(st + eps)) @ ut.T
    A = ss_sqrt_inv @ st_sqrt
    Xs_aligned = Xs @ A
    return Xs_aligned.astype(np.float32), Xt.astype(np.float32)


class MLPCoralModel:
    """
    Shallow MLP with optional covariance-only / source-scaler CORAL.

    - ``StandardScaler`` is fit on source rows only; target (if any) uses
      ``transform`` in that source-scaled space.
    - When ``use_coral``, ``coral_align`` maps source covariance toward the
      (scaled) target covariance at train time; predict applies the scaler only
      (no re-application of A). This is a weak UDA-lite schedule, not dual-center
      CORAL.
    - Early stopping is sklearn's internal ~10% row-holdout on accuracy
      (``early_stopping=True``), not a well-held-out validation multi_logloss.
    """

    def __init__(
        self,
        hidden=(128, 64),
        max_iter: int = 40,
        random_state: int = 42,
        use_coral: bool = True,
    ):
        self.hidden = hidden
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_coral = use_coral
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=5,
        )
        self._A_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_target_unlabeled: np.ndarray | None = None,
    ) -> "MLPCoralModel":
        # Source-only scaler; target (if present) shares that scale — not dual-center.
        Xs = self.scaler.fit_transform(X)
        self._align_matrix = None
        self._target_ref = X_target_unlabeled
        if self.use_coral and X_target_unlabeled is not None and len(X_target_unlabeled) > 10:
            Xt = self.scaler.transform(X_target_unlabeled)
            # Covariance-only align in source-scaled space; leave target unchanged.
            Xs, _ = coral_align(Xs.astype(np.float64), Xt.astype(np.float64))
            self._align_matrix = True  # mark that CORAL was applied at train time
            self._A_fitted = True
        self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Source-scaler only; do NOT re-apply the train-time CORAL map A.
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)
