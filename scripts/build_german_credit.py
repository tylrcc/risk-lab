"""Download the UCI Statlog German Credit data and write it as a readable CSV.

Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
The raw file uses codes like A11, A34; this script decodes them to labels.
"""
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
OUT = Path(__file__).resolve().parents[1] / "data" / "german_credit.csv"

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "dependents", "telephone", "foreign_worker", "target",
]

CODES = {
    "checking_status": {"A11": "negative", "A12": "0 to 200", "A13": "200 plus", "A14": "none"},
    "credit_history": {
        "A30": "no credits taken", "A31": "all paid at this bank", "A32": "existing paid duly",
        "A33": "past delays", "A34": "critical account",
    },
    "purpose": {
        "A40": "car new", "A41": "car used", "A42": "furniture", "A43": "radio tv",
        "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
        "A48": "retraining", "A49": "business", "A410": "other",
    },
    "savings": {"A61": "under 100", "A62": "100 to 500", "A63": "500 to 1000", "A64": "1000 plus", "A65": "unknown"},
    "employment_since": {"A71": "unemployed", "A72": "under 1 year", "A73": "1 to 4 years", "A74": "4 to 7 years", "A75": "7 plus years"},
    "personal_status": {
        "A91": "male divorced", "A92": "female divorced or married", "A93": "male single",
        "A94": "male married or widowed", "A95": "female single",
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings or insurance", "A123": "car or other", "A124": "none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "job": {"A171": "unskilled non-resident", "A172": "unskilled resident", "A173": "skilled", "A174": "management or self-employed"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


def main() -> None:
    with urlopen(URL) as resp:
        df = pd.read_csv(resp, sep=" ", header=None, names=COLUMNS)
    for col, mapping in CODES.items():
        df[col] = df[col].map(mapping)
    df["default"] = (df.pop("target") == 2).astype(int)
    df.insert(0, "loan_id", range(1, len(df) + 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows, default rate {df['default'].mean():.1%})")


if __name__ == "__main__":
    main()
