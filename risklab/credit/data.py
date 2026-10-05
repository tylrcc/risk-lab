"""Loan dataset loaders. Each returns a frame with an `loan_id` column, a
binary `default` column, and feature columns; everything else is dropped."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "default"
ID = "loan_id"

LENDING_CLUB_FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "dti", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies", "fico_range_low",
]
LENDING_CLUB_DEFAULT = {"Charged Off", "Default", "Late (31-120 days)"}
LENDING_CLUB_PAID = {"Fully Paid"}


def load(source: str, path: Path, sample: int | None = None, seed: int = 0) -> pd.DataFrame:
    if source == "german":
        df = load_german(path)
    elif source == "lendingclub":
        df = load_lending_club(path)
    else:
        raise ValueError(f"unknown source {source!r}")
    if sample and sample < len(df):
        df = df.sample(sample, random_state=seed).reset_index(drop=True)
    return df


def load_german(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_lending_club(path: Path) -> pd.DataFrame:
    """Adapter for the Lending Club accepted loans export (2007 to 2018).

    Keeps finished loans only, so the label is known, and parses the handful
    of columns that ship as strings.
    """
    usecols = [c for c in LENDING_CLUB_FEATURES + ["id", "loan_status"]]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    df = df[df["loan_status"].isin(LENDING_CLUB_DEFAULT | LENDING_CLUB_PAID)].copy()
    df[TARGET] = df["loan_status"].isin(LENDING_CLUB_DEFAULT).astype(int)
    df = df.drop(columns="loan_status").rename(columns={"id": ID})

    if "term" in df:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
    for col in ("int_rate", "revol_util"):
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")
    if "emp_length" in df:
        df["emp_length"] = (
            df["emp_length"].astype(str)
            .str.replace("< 1", "0", regex=False)
            .str.extract(r"(\d+)").astype(float)
        )
    return df.reset_index(drop=True)


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ids = df[ID] if ID in df else pd.Series(np.arange(len(df)), name=ID)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[c for c in (ID, TARGET) if c in df])
    return X, y, ids


def column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical
