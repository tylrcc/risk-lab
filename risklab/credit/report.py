from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .. import plotting
from ..portfolio.report import md_table


def roc_chart(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (y, p, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.3, label=f"{name}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#9ca3af")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC, held out set")
    ax.legend(loc="lower right")
    return plotting.save(fig, out)


def ks_chart(y: np.ndarray, p: np.ndarray, out: Path) -> Path:
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(thr, tpr, lw=1.2, label="defaults captured")
    ax.plot(thr, fpr, lw=1.2, label="good loans flagged")
    ax.vlines(thr[i], fpr[i], tpr[i], color="#dc2626", lw=1.2, label=f"KS {tpr[i] - fpr[i]:.3f} at {thr[i]:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("PD threshold")
    ax.set_ylabel("cumulative share")
    ax.set_title("Kolmogorov Smirnov separation")
    ax.legend()
    return plotting.save(fig, out)


def calibration_chart(table: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#9ca3af")
    ax.plot(table["predicted"], table["observed"], marker="o", ms=4, lw=1.2)
    lim = max(table["predicted"].max(), table["observed"].max()) * 1.1
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mean predicted PD")
    ax.set_ylabel("observed default rate")
    ax.set_title("Calibration by PD decile")
    return plotting.save(fig, out)


def distribution_chart(y: np.ndarray, p: np.ndarray, threshold: float, out: Path) -> Path:
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, alpha=0.75, label="repaid", color="#2563eb")
    ax.hist(p[y == 1], bins=bins, alpha=0.75, label="defaulted", color="#dc2626")
    ax.axvline(threshold, color="#111827", lw=1, ls="--", label=f"decision threshold {threshold:.2f}")
    ax.set_xlabel("predicted probability of default")
    ax.set_ylabel("loans")
    ax.set_title("Score distribution by outcome")
    ax.legend()
    return plotting.save(fig, out)


def importance_chart(imp: pd.DataFrame, out: Path, top: int = 15) -> Path:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(d) + 1.2))
    ax.barh(d["feature"], d["importance"], xerr=d["std"], color="#1f2937", height=0.6, error_kw={"lw": 0.8})
    ax.set_xlabel("drop in AUC when feature is shuffled")
    ax.set_title("Permutation importance")
    return plotting.save(fig, out)


def rating_chart(table: pd.DataFrame, out: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.bar(table.index, table["loans"], color="#d1d5db", width=0.6, label="loans")
    ax1.set_ylabel("loans")
    ax1.grid(False)
    ax2 = ax1.twinx()
    ax2.plot(table.index, table["observed_default_rate"], marker="o", color="#dc2626", lw=1.3, label="observed default rate")
    ax2.plot(table.index, table["avg_pd"], marker="s", color="#2563eb", lw=1.0, ls="--", label="average predicted PD")
    ax2.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax2.set_ylim(0, 1)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax1.set_title("Risk rating buckets")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    return plotting.save(fig, out)


def write_summary(metrics: dict[str, dict], cv: dict[str, np.ndarray], ratings: pd.DataFrame,
                  imp: pd.DataFrame, out: Path, config: dict) -> Path:
    table = pd.DataFrame(metrics).T[["auc", "gini", "ks", "brier", "precision", "recall", "approval_rate", "default_rate_approved"]]
    show = table.copy()
    for c in ("auc", "gini", "ks", "brier"):
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    for c in ("precision", "recall", "approval_rate", "default_rate_approved"):
        show[c] = show[c].map(lambda v: f"{v:.1%}")
    show.columns = ["AUC", "Gini", "KS", "Brier", "Precision", "Recall", "Approval rate", "Default rate (approved)"]

    best = max(metrics, key=lambda k: metrics[k]["auc"])
    m = metrics[best]
    lines = ["# Credit risk report", ""]
    lines.append(
        f"Source: {config['source']} ({m['n']} held out loans, base default rate {m['base_rate']:.1%}). "
        f"Test split {config['test_size']:.0%}, seed {config['seed']}. Decision threshold chosen to minimise cost "
        f"with a missed default weighted {config['cost_fn']:g}x a declined good loan."
    )
    lines += ["", md_table(show, "Model"), ""]
    lines.append("5 fold cross validated AUC on the training set: " + ", ".join(
        f"{k} {v.mean():.3f} (sd {v.std():.3f})" for k, v in cv.items()))
    lines += ["", f"## Rating buckets, {best}", ""]
    r = ratings.copy()
    r["avg_pd"] = r["avg_pd"].map(lambda v: f"{v:.1%}")
    r["observed_default_rate"] = r["observed_default_rate"].map(lambda v: f"{v:.1%}")
    r["share"] = r["share"].map(lambda v: f"{v:.1%}")
    r.columns = ["Loans", "Avg PD", "Observed default", "Share"]
    lines += [md_table(r, "Rating"), ""]
    lines += [f"## Top features, {best}", ""]
    top = imp.head(10).copy()
    top["importance"] = top["importance"].map(lambda v: f"{v:.4f}")
    top["std"] = top["std"].map(lambda v: f"{v:.4f}")
    lines += [md_table(top.set_index("feature")[["importance", "std"]], "Feature"), ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
