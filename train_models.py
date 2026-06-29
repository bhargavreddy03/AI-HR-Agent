"""
train_models.py  —  Upgraded: full academic metrics + 16 graphs
=================================================================
Metrics per model:
  Train Accuracy, Test Accuracy, Precision, Recall, F1, FPR, FNR,
  ROC-AUC, Confusion Matrix, CV Mean/Std, MSE, RMSE, MAE

Graphs (16):
  1.  Train vs Test Accuracy (grouped bar)
  2.  Precision / Recall / F1 (grouped bar)
  3.  FPR vs FNR (grouped bar)
  4.  ROC-AUC comparison (bar)
  5.  Confusion Matrix — best model (heatmap)
  6.  CV Mean vs Test Accuracy with error bars
  7.  Error Metrics — MSE / RMSE / MAE
  8.  ROC Curves — all prob-capable models
  9.  Feature Importance — RandomForest top-20
  10. Actual vs Predicted — best model (DARK THEME — probability-based)
  11. Metric Dashboard (test acc / precision / recall / F1 / AUC)
  12. F1 Score only
  13. FPR only
  14. FNR only
  15. Precision only
  16. Recall only
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score,
    roc_curve, auc as sk_auc,
)
from sklearn.model_selection import cross_val_score

from preprocessing import load_and_preprocess

warnings.filterwarnings("ignore")

MODEL_DIR  = "models"
GRAPHS_DIR = "static/graphs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ── Dark theme palette (used for ALL graphs including #10) ────────────────────
PALETTE = [
    "#4361ee", "#f72585", "#7209b7", "#3a0ca3",
    "#4cc9f0", "#06d6a0", "#ff9f1c", "#ef233c",
]
BG    = "#0f1117"
FG    = "#e0e0f0"
MUTED = "#aaaaaa"
GRID  = "#2a2a3a"


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, reg_alpha=0.5,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
    except ImportError:
        xgb = None

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6,
            min_samples_leaf=5, max_features="sqrt",
            random_state=42, n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            C=0.5, max_iter=2000, random_state=42,
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(C=0.3, max_iter=3000, random_state=42)
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            subsample=0.8, learning_rate=0.05,
            random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=0.01,
            max_iter=400, random_state=42,
        ),
    }
    if xgb is not None:
        models["XGBoost"] = xgb
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fpr_fnr(cm):
    """Return (FPR, FNR) from a 2x2 confusion matrix."""
    try:
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    except Exception:
        fpr, fnr = 0.0, 0.0
    return fpr, fnr


def _compute_roc_auc(y_test, y_prob):
    if y_prob is None:
        return 0.0
    try:
        return roc_auc_score(y_test, y_prob[:, 1])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ModelTrainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, feature_names, class_names):
        self.X_train       = X_train
        self.X_test        = X_test
        self.y_train       = y_train
        self.y_test        = y_test
        self.feature_names = list(feature_names)
        self.class_names   = list(class_names)
        self.results       = {}
        self.models        = {}
        self.best_model_name = None

    # ── Training ─────────────────────────────────────────────────────────────

    def train_all(self):
        print("\n🚀 Training ML Models...\n" + "─" * 60)

        for name, model in get_models().items():
            print(f"  ▸ {name} ...", end=" ", flush=True)

            model.fit(self.X_train, self.y_train)

            y_train_pred = model.predict(self.X_train)
            y_test_pred  = model.predict(self.X_test)
            y_prob = (
                model.predict_proba(self.X_test)
                if hasattr(model, "predict_proba") else None
            )

            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy  = accuracy_score(self.y_test,  y_test_pred)

            avg = "binary" if len(self.class_names) == 2 else "macro"
            precision = precision_score(self.y_test, y_test_pred,
                                        average=avg, zero_division=0)
            recall    = recall_score(self.y_test, y_test_pred,
                                     average=avg, zero_division=0)
            f1        = f1_score(self.y_test, y_test_pred,
                                 average=avg, zero_division=0)

            cm = confusion_matrix(self.y_test, y_test_pred)
            fpr_val, fnr_val = _compute_fpr_fnr(cm)

            auc_val = _compute_roc_auc(self.y_test, y_prob)

            mse  = mean_squared_error(self.y_test, y_test_pred)
            rmse = float(np.sqrt(mse))
            mae  = mean_absolute_error(self.y_test, y_test_pred)

            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, cv=5, scoring="accuracy"
            )

            report = classification_report(
                self.y_test, y_test_pred,
                target_names=self.class_names,
                output_dict=True, zero_division=0,
            )

            self.results[name] = {
                "train_accuracy":        train_accuracy,
                "test_accuracy":         test_accuracy,
                "precision":             precision,
                "recall":                recall,
                "f1_score":              f1,
                "fpr":                   fpr_val,
                "fnr":                   fnr_val,
                "roc_auc":               auc_val,
                "mse":                   mse,
                "rmse":                  rmse,
                "mae":                   mae,
                "cv_mean":               cv_scores.mean(),
                "cv_std":                cv_scores.std(),
                "confusion_matrix":      cm,
                "classification_report": report,
                "y_pred":                y_test_pred,
                "y_prob":                y_prob,
            }
            self.models[name] = model

            print(
                f"Train={train_accuracy:.3f}  Test={test_accuracy:.3f}  "
                f"F1={f1:.3f}  AUC={auc_val:.3f}  "
                f"CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
            )

        self.best_model_name = max(
            self.results, key=lambda k: self.results[k]["test_accuracy"]
        )
        print(
            f"\n✅ Best model: {self.best_model_name} "
            f"(test_acc={self.results[self.best_model_name]['test_accuracy']:.4f})"
        )

    # ── Save models ───────────────────────────────────────────────────────────

    def save_models(self):
        for name, model in self.models.items():
            joblib.dump(
                model,
                f"{MODEL_DIR}/{name.lower().replace(' ', '_')}.pkl",
            )
        joblib.dump(self.best_model_name, f"{MODEL_DIR}/best_model_name.pkl")
        print(f"[✓] All models saved to {MODEL_DIR}/")

    # ── Print summary ─────────────────────────────────────────────────────────

    def print_summary(self):
        print("\n" + "=" * 70)
        print("📋  PER-MODEL METRIC SUMMARY")
        print("=" * 70)
        for name, r in self.results.items():
            print(f"\nModel          : {name}")
            print(f"  Train Accuracy : {r['train_accuracy']:.4f}")
            print(f"  Test Accuracy  : {r['test_accuracy']:.4f}")
            print(f"  Precision      : {r['precision']:.4f}")
            print(f"  Recall         : {r['recall']:.4f}")
            print(f"  F1 Score       : {r['f1_score']:.4f}")
            print(f"  FPR            : {r['fpr']:.4f}")
            print(f"  FNR            : {r['fnr']:.4f}")
            print(f"  ROC-AUC        : {r['roc_auc']:.4f}")
            print(f"  MSE            : {r['mse']:.4f}")
            print(f"  RMSE           : {r['rmse']:.4f}")
            print(f"  MAE            : {r['mae']:.4f}")
            print(f"  CV Mean Acc    : {r['cv_mean']:.4f}")
            print(f"  CV Std         : {r['cv_std']:.4f}")

        print("\n" + "=" * 70)
        print("🏆  RANKED SUMMARY  (sorted by Test Accuracy)")
        print("=" * 70)
        header = (
            f"{'Model':<22} {'TrainAcc':>9} {'TestAcc':>9} "
            f"{'Prec':>7} {'Recall':>7} {'F1':>7} "
            f"{'FPR':>7} {'FNR':>7} {'AUC':>7} "
            f"{'CVMean':>8} {'CVStd':>7}"
        )
        print(header)
        print("─" * len(header))
        for name, r in sorted(
            self.results.items(), key=lambda x: -x[1]["test_accuracy"]
        ):
            print(
                f"{name:<22} "
                f"{r['train_accuracy']:>9.4f} {r['test_accuracy']:>9.4f} "
                f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1_score']:>7.4f} "
                f"{r['fpr']:>7.4f} {r['fnr']:>7.4f} {r['roc_auc']:>7.4f} "
                f"{r['cv_mean']:>8.4f} {r['cv_std']:>7.4f}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Graph utilities — DARK theme
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self, fig, fname):
        path = f"{GRAPHS_DIR}/{fname}"
        fig.savefig(path, bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [graph] ✔ {path}")

    def _style(self, ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color(GRID)
        if title:
            ax.set_title(title, color=FG, fontsize=10, pad=8)
        if xlabel:
            ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, color=MUTED, fontsize=9)

    def _bar_labels(self, ax, bars, fmt=".3f"):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01, f"{h:{fmt}}",
                ha="center", va="bottom", color=FG, fontsize=7,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 16 graphs
    # ─────────────────────────────────────────────────────────────────────────

    def plot_all(self):
        print("\n📊 Generating graphs...\n" + "─" * 60)
        self._plot_train_vs_test_accuracy()      # 1
        self._plot_precision_recall_f1()          # 2
        self._plot_fpr_fnr()                      # 3
        self._plot_roc_auc_bar()                  # 4
        self._plot_confusion_matrix()             # 5
        self._plot_cv_comparison()                # 6
        self._plot_error_metrics()                # 7
        self._plot_roc_curves()                   # 8
        self._plot_feature_importance()           # 9
        self._plot_actual_vs_predicted()          # 10  ← FIXED: uses predict_proba
        self._plot_metric_dashboard()             # 11
        self._plot_single_metric("f1_score",  "F1 Score",  PALETTE[1], "f1_score_comparison.png")
        self._plot_single_metric("fpr",       "FPR",       PALETTE[2], "fpr_comparison.png")
        self._plot_single_metric("fnr",       "FNR",       PALETTE[3], "fnr_comparison.png")
        self._plot_single_metric("precision", "Precision", PALETTE[4], "precision_comparison.png")
        self._plot_single_metric("recall",    "Recall",    PALETTE[5], "recall_comparison.png")
        print("✅ All graphs saved.")

    # 1 ── Train vs Test Accuracy ─────────────────────────────────────────────

    def _plot_train_vs_test_accuracy(self):
        names      = list(self.results.keys())
        train_accs = [self.results[n]["train_accuracy"] for n in names]
        test_accs  = [self.results[n]["test_accuracy"]  for n in names]

        fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG)
        x, w = np.arange(len(names)), 0.35
        b1 = ax.bar(x - w / 2, train_accs, w, label="Train Accuracy",
                    color=PALETTE[0], alpha=0.9)
        b2 = ax.bar(x + w / 2, test_accs,  w, label="Test Accuracy",
                    color=PALETTE[1], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, 1.15)
        ax.legend(facecolor="#1a1a2e", labelcolor=FG)
        self._bar_labels(ax, list(b1) + list(b2))
        self._style(ax, "Train Accuracy vs Test Accuracy", ylabel="Accuracy")
        self._save(fig, "train_vs_test_accuracy.png")

    # 2 ── Precision / Recall / F1 ────────────────────────────────────────────

    def _plot_precision_recall_f1(self):
        names = list(self.results.keys())
        prec  = [self.results[n]["precision"] for n in names]
        rec   = [self.results[n]["recall"]    for n in names]
        f1s   = [self.results[n]["f1_score"]  for n in names]

        fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG)
        x, w = np.arange(len(names)), 0.25
        b1 = ax.bar(x - w, prec, w, label="Precision", color=PALETTE[0], alpha=0.9)
        b2 = ax.bar(x,     rec,  w, label="Recall",    color=PALETTE[1], alpha=0.9)
        b3 = ax.bar(x + w, f1s,  w, label="F1 Score",  color=PALETTE[2], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, 1.2)
        ax.legend(facecolor="#1a1a2e", labelcolor=FG)
        self._bar_labels(ax, list(b1) + list(b2) + list(b3))
        self._style(ax, "Precision, Recall & F1 Score Comparison", ylabel="Score")
        self._save(fig, "precision_recall_f1.png")

    # 3 ── FPR vs FNR ─────────────────────────────────────────────────────────

    def _plot_fpr_fnr(self):
        names = list(self.results.keys())
        fprs  = [self.results[n]["fpr"] for n in names]
        fnrs  = [self.results[n]["fnr"] for n in names]

        fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG)
        x, w = np.arange(len(names)), 0.35
        b1 = ax.bar(x - w / 2, fprs, w, label="FPR (False Positive Rate)",
                    color=PALETTE[3], alpha=0.9)
        b2 = ax.bar(x + w / 2, fnrs, w, label="FNR (False Negative Rate)",
                    color=PALETTE[4], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, max(max(fprs + fnrs) * 1.3, 0.2))
        ax.legend(facecolor="#1a1a2e", labelcolor=FG)
        self._bar_labels(ax, list(b1) + list(b2))
        self._style(ax, "FPR vs FNR Comparison", ylabel="Rate")
        self._save(fig, "fpr_fnr_comparison.png")

    # 4 ── ROC-AUC bar ────────────────────────────────────────────────────────

    def _plot_roc_auc_bar(self):
        names = list(self.results.keys())
        aucs  = [self.results[n]["roc_auc"] for n in names]

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
        bars = ax.bar(names, aucs, color=colors, alpha=0.9)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, 1.15)
        self._bar_labels(ax, bars)
        self._style(ax, "ROC-AUC Comparison (All Models)", ylabel="ROC-AUC")
        self._save(fig, "roc_auc_comparison.png")

    # 5 ── Confusion Matrix ───────────────────────────────────────────────────

    def _plot_confusion_matrix(self):
        cm  = self.results[self.best_model_name]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="magma",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax, linewidths=0.5, linecolor=BG,
            annot_kws={"size": 13, "color": FG},
        )
        ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
        ax.set_ylabel("Actual",    color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED)
        ax.set_title(
            f"Confusion Matrix  —  {self.best_model_name}",
            color=FG, fontsize=11,
        )
        fig.patch.set_facecolor(BG)
        self._save(fig, "confusion_matrix.png")

    # 6 ── CV Mean vs Test Accuracy with error bars ───────────────────────────

    def _plot_cv_comparison(self):
        names = list(self.results.keys())
        cv_m  = [self.results[n]["cv_mean"]       for n in names]
        cv_s  = [self.results[n]["cv_std"]        for n in names]
        t_acc = [self.results[n]["test_accuracy"]  for n in names]

        fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG)
        x, w = np.arange(len(names)), 0.35
        ax.bar(x - w / 2, cv_m, w, label="CV Mean Accuracy",
               color=PALETTE[0], yerr=cv_s, capsize=5, alpha=0.85,
               error_kw={"ecolor": FG, "alpha": 0.7})
        ax.bar(x + w / 2, t_acc, w, label="Test Accuracy",
               color=PALETTE[5], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, 1.18)
        ax.legend(facecolor="#1a1a2e", labelcolor=FG)
        self._style(ax, "Cross-Validation Mean vs Test Accuracy", ylabel="Accuracy")
        self._save(fig, "cv_comparison.png")

    # 7 ── Error Metrics ──────────────────────────────────────────────────────

    def _plot_error_metrics(self):
        names = list(self.results.keys())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=BG)
        fig.patch.set_facecolor(BG)

        for ax, key, label, color in zip(
            axes,
            ["mse", "rmse", "mae"],
            ["MSE", "RMSE", "MAE"],
            [PALETTE[2], PALETTE[3], PALETTE[4]],
        ):
            vals = [self.results[n][key] for n in names]
            bars = ax.bar(names, vals, color=color, alpha=0.85)
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
            self._bar_labels(ax, bars, fmt=".4f")
            self._style(ax, label)

        fig.suptitle("Error Metrics Comparison — MSE / RMSE / MAE",
                     color=FG, fontsize=12, y=1.02)
        plt.tight_layout()
        self._save(fig, "error_metrics.png")

    # 8 ── ROC Curves ─────────────────────────────────────────────────────────

    def _plot_roc_curves(self):
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
        ax.set_facecolor(BG)
        plotted = 0
        for i, (name, res) in enumerate(self.results.items()):
            if res["y_prob"] is None:
                continue
            try:
                fpr, tpr, _ = roc_curve(self.y_test, res["y_prob"][:, 1])
                auc_val     = sk_auc(fpr, tpr)
                ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                        label=f"{name} (AUC = {auc_val:.3f})")
                plotted += 1
            except Exception:
                pass

        if plotted == 0:
            plt.close(fig)
            return

        ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random")
        ax.set_xlabel("False Positive Rate (FPR)", color=MUTED, fontsize=9)
        ax.set_ylabel("True Positive Rate (TPR)", color=MUTED, fontsize=9)
        ax.legend(facecolor="#1a1a2e", labelcolor=FG, fontsize=8, loc="lower right")
        ax.spines[:].set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.set_title("ROC Curves — All Models", color=FG, fontsize=12)
        self._save(fig, "roc_curves.png")

    # 9 ── Feature Importance ─────────────────────────────────────────────────

    def _plot_feature_importance(self):
        rf = self.models.get("RandomForest")
        if rf is None or not hasattr(rf, "feature_importances_"):
            return
        imp   = rf.feature_importances_
        top_n = min(20, len(imp))
        idx   = np.argsort(imp)[::-1][:top_n]
        feats = [
            self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            for i in idx
        ]
        vals = imp[idx]

        fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
        colors  = plt.cm.plasma(np.linspace(0.2, 0.9, top_n))
        ax.barh(feats[::-1], vals[::-1], color=colors)
        ax.set_xlabel("Importance Score", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.set_facecolor(BG)
        ax.set_title(f"Top {top_n} Feature Importances — RandomForest",
                     color=FG, fontsize=11)
        self._save(fig, "feature_importance.png")

    # 10 ── Actual vs Predicted — FIXED: uses predict_proba ───────────────────
    #
    # WHY THE OLD APPROACH WAS WRONG:
    #   Plotting hard labels (0/1) vs actual (0/1) gives only 4 possible
    #   coordinate pairs: (0,0), (0,1), (1,0), (1,1). Even with jitter this
    #   just creates 4 artificial blobs — it carries no diagnostic value.
    #
    # THE FIX:
    #   Use predicted probabilities (predict_proba[:, 1]) on the Y-axis.
    #   Each sample now gets a unique float in [0, 1], producing a continuous
    #   scatter. Correct high-confidence predictions cluster at (0, ~0.0) and
    #   (1, ~1.0); uncertain or wrong predictions drift toward the 0.5 line,
    #   immediately revealing model quality at a glance.
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_actual_vs_predicted(self):
        res    = self.results[self.best_model_name]
        y_test = np.array(self.y_test)
        y_prob = res.get("y_prob")

        if y_prob is not None:
            # ── PRIMARY PATH: use P(class=1) as continuous Y-axis ────────────
            y_axis    = y_prob[:, 1]                     # shape (n_samples,)
            y_label   = "Predicted Probability  (class = 1)"
            use_proba = True
        else:
            # ── FALLBACK: no predict_proba available (rare after calibration) ─
            # Jitter is the only option; label the plot accordingly.
            rng       = np.random.default_rng(42)
            y_axis    = res["y_pred"].astype(float) + rng.uniform(-0.08, 0.08, size=len(res["y_pred"]))
            y_label   = "Predicted Label  (jittered — no probability available)"
            use_proba = False

        # ── colour each point by correctness ─────────────────────────────────
        y_pred_hard = res["y_pred"]
        correct     = (y_pred_hard == y_test)
        pt_colors   = np.where(correct, "#4a9ede", "#f72585")   # blue / pink

        # ── figure ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        ax.scatter(
            y_test, y_axis,
            c=pt_colors,
            alpha=0.70,
            s=32,
            edgecolors="none",
            zorder=3,
        )

        # ── legend proxy artists ──────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#4a9ede", markersize=7,
                   label="Correct prediction"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#f72585", markersize=7,
                   label="Wrong prediction"),
        ]

        if use_proba:
            # Ideal probability reference: actual 0 → prob 0.0, actual 1 → prob 1.0
            ax.plot([0, 1], [0, 1],
                    color="#e05555", linestyle="--", linewidth=1.8,
                    alpha=0.90, zorder=2)

            # Decision boundary at 0.5
            ax.axhline(0.5,
                       color="#ffcc00", linestyle=":", linewidth=1.4,
                       alpha=0.75)

            legend_elements += [
                Line2D([0], [0], color="#e05555", linestyle="--",
                       linewidth=1.8, label="Ideal probability line"),
                Line2D([0], [0], color="#ffcc00", linestyle=":",
                       linewidth=1.4, label="Decision boundary (0.5)"),
            ]

            ax.set_xlim(-0.15, 1.15)
            ax.set_ylim(-0.10, 1.15)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Actual  0\n(Not Suitable)",
                                 "Actual  1\n(Suitable)"],
                               color=MUTED, fontsize=9)
            subtitle = (
                "High-confidence correct preds cluster at corners  |"
                "  Points near 0.5 = uncertain"
            )
        else:
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
            subtitle = (
                "Hard labels shown (no predict_proba available)  |  Jitter applied"
            )

        # ── axes decoration ───────────────────────────────────────────────────
        ax.set_xlabel("Actual Label",  fontsize=12, color=FG, labelpad=8)
        ax.set_ylabel(y_label,         fontsize=11, color=FG, labelpad=8)
        ax.set_title(
            f"Actual vs Predicted  —  {self.best_model_name}",
            fontsize=13, color=FG, pad=14,
        )
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(0.8)
        ax.grid(True, color=GRID, linewidth=0.6, linestyle="-", alpha=0.8)
        ax.set_axisbelow(True)

        ax.legend(
            handles=legend_elements,
            fontsize=8, loc="upper left",
            frameon=True, framealpha=0.85,
            edgecolor=GRID, facecolor=BG, labelcolor=FG,
        )

        fig.text(
            0.5, 0.01, subtitle,
            ha="center", va="bottom", fontsize=9, color=MUTED,
        )

        self._save(fig, "actual_vs_predicted.png")

    # 11 ── Metric Dashboard ──────────────────────────────────────────────────

    def _plot_metric_dashboard(self):
        names   = list(self.results.keys())
        metrics = {
            "Test Accuracy": [self.results[n]["test_accuracy"] for n in names],
            "Precision":     [self.results[n]["precision"]     for n in names],
            "Recall":        [self.results[n]["recall"]        for n in names],
            "F1 Score":      [self.results[n]["f1_score"]      for n in names],
            "ROC-AUC":       [self.results[n]["roc_auc"]       for n in names],
        }
        n_metrics = len(metrics)
        x = np.arange(len(names))
        w = 0.75 / n_metrics

        fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
        for i, (label, vals) in enumerate(metrics.items()):
            offset = (i - n_metrics / 2) * w + w / 2
            ax.bar(x + offset, vals, w,
                   label=label, color=PALETTE[i % len(PALETTE)], alpha=0.88)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, 1.22)
        ax.legend(facecolor="#1a1a2e", labelcolor=FG, fontsize=8,
                  loc="upper right", ncol=3)
        self._style(ax,
                    "Metric Dashboard — Test Acc / Precision / Recall / F1 / AUC",
                    ylabel="Score")
        self._save(fig, "metric_dashboard.png")

    # 12-16 ── Single-metric bar charts ───────────────────────────────────────

    def _plot_single_metric(self, key, label, color, filename):
        names = list(self.results.keys())
        vals  = [self.results[n][key] for n in names]

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
        bars = ax.bar(names, vals, color=color, alpha=0.88)
        ax.set_xticklabels(names, rotation=25, ha="right")
        top = max(vals) if vals else 1.0
        ax.set_ylim(0, min(top * 1.3, 1.15))
        self._bar_labels(ax, bars)
        self._style(ax, f"{label} — Model Comparison", ylabel=label)
        self._save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry-point
# ─────────────────────────────────────────────────────────────────────────────

def train_pipeline(csv_path="resume_dataset_1200.csv"):
    X_train, X_test, y_train, y_test, prep, feature_names = load_and_preprocess(csv_path)
    class_names = ["Not Suitable", "Suitable"]

    trainer = ModelTrainer(
        X_train, X_test, y_train, y_test, feature_names, class_names
    )
    trainer.train_all()
    trainer.save_models()
    trainer.print_summary()
    trainer.plot_all()

    return trainer.results, trainer.best_model_name


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, best = train_pipeline()

    print(f"\n🏆  Best Model  : {best}")
    r = results[best]
    print(f"   Train Acc   : {r['train_accuracy']:.4f}")
    print(f"   Test Acc    : {r['test_accuracy']:.4f}")
    print(f"   Precision   : {r['precision']:.4f}")
    print(f"   Recall      : {r['recall']:.4f}")
    print(f"   F1 Score    : {r['f1_score']:.4f}")
    print(f"   FPR         : {r['fpr']:.4f}")
    print(f"   FNR         : {r['fnr']:.4f}")
    print(f"   ROC-AUC     : {r['roc_auc']:.4f}")
    print(f"   MSE         : {r['mse']:.4f}")
    print(f"   RMSE        : {r['rmse']:.4f}")
    print(f"   MAE         : {r['mae']:.4f}")
    print(f"   CV Mean     : {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")