"""Generate the bundled sample CSVs (deterministic, seed=42).

These are synthetic but realistic datasets used to demo the full pipeline
without users needing their own CSV. They are generated once and committed;
the analysis itself is fully real.

Usage:  python scripts/generate_samples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "samples"
OUT.mkdir(parents=True, exist_ok=True)


def write(name: str, df: pd.DataFrame, meta: dict) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"  {name}.csv  {df.shape[0]} rows x {df.shape[1]} cols")


# ---------------------------------------------------------------- housing
rng = np.random.default_rng(42)
n = 500
sqft = rng.integers(650, 4200, n)
bedrooms = rng.integers(1, 6, n)
bathrooms = np.clip(rng.integers(1, 5, n) + rng.integers(0, 2, n), 1, 4)
age_years = rng.integers(0, 85, n)
lot_size = rng.integers(1500, 22000, n)
garage = rng.integers(0, 3, n)
pool = rng.choice([0, 1], n, p=[0.86, 0.14])
neighborhood = rng.choice(
    ["Downtown", "Suburban", "Rural", "Waterfront"], n, p=[0.30, 0.42, 0.18, 0.10]
)
quality = rng.integers(1, 11, n)
sale_price = (
    85_000
    + 165 * sqft
    + 12_000 * bedrooms
    + 9_500 * bathrooms
    - 1_150 * age_years
    + 6.2 * lot_size
    + 24_000 * garage
    + 17_000 * pool
    + 7_800 * quality
    + rng.normal(0, 32_000, n)
)
sale_price = np.clip(sale_price, 60_000, None).round(0)

housing = pd.DataFrame(
    {
        "sale_price": sale_price,
        "square_feet": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "lot_size": lot_size,
        "garage_spaces": garage,
        "has_pool": pool,
        "neighborhood": neighborhood,
        "quality_rating": quality,
    }
)
# inject ~2.5% missing values in two columns
for col, frac in (("square_feet", 0.02), ("age_years", 0.03)):
    idx = rng.choice(n, size=int(n * frac), replace=False)
    housing.loc[idx, col] = np.nan
# a few exact duplicate rows
housing = pd.concat([housing, housing.iloc[rng.choice(n, 8, replace=False)]], ignore_index=True)
write(
    "housing_sales",
    housing,
    {
        "label": "Housing Sales",
        "description": "508 home sale records with price, size, age, neighborhood and amenities. Predict the sale price (regression). Contains a few duplicate rows and some missing values on purpose.",
        "task": "regression — target: sale_price",
    },
)

# ---------------------------------------------------------------- churn
rng = np.random.default_rng(7)
n = 600
tenure_months = rng.integers(1, 73, n)
monthly_charges = rng.uniform(20, 120, n).round(2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.3, 0.2])
support_tickets = rng.integers(0, 7, n)
age = rng.integers(18, 81, n)
payment_method = rng.choice(["Credit card", "Bank transfer", "Cash", "Paypal"], n)

logit = (
    -4.2
    - 0.028 * tenure_months
    - 0.045 * (contract == "One year").astype(int) * 60
    - 0.06 * (contract == "Two year").astype(int) * 60
    + 0.045 * monthly_charges
    + 0.55 * support_tickets
    - 0.012 * age
)
p_churn = 1 / (1 + np.exp(-logit))
churn = (rng.random(n) < p_churn).astype(int)
churn_df = pd.DataFrame(
    {
        "churned": pd.Series(churn).map({0: "No", 1: "Yes"}),
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "contract_type": contract,
        "support_tickets": support_tickets,
        "age": age,
        "payment_method": payment_method,
    }
)
idx = rng.choice(n, size=15, replace=False)
churn_df.loc[idx, "support_tickets"] = np.nan
write(
    "customer_churn",
    churn_df,
    {
        "label": "Customer Churn",
        "description": "600 customer accounts with tenure, charges, contract type and support activity. Predict whether a customer churns (binary classification).",
        "task": "classification — target: churned",
    },
)

# ---------------------------------------------------------------- customers (clustering)
rng = np.random.default_rng(1234)
centers = [
    (28, 38, 220, 3.5),   # young, low-mid income, low spend, few visits
    (45, 72, 640, 7.2),   # established, higher income, moderate spend
    (62, 95, 1500, 11.4), # affluent seniors, high spend
    (35, 130, 900, 18.6), # high-income, very active shoppers
]
parts = []
labels = []
for i, (ca, ci, cs, cv) in enumerate(centers):
    m = n // 4
    parts.append(
        rng.normal(loc=(ca, ci, cs, cv), scale=(7, 14, cs * 0.25, cv * 0.25), size=(m, 4))
    )
    labels += [i] * m
cust = np.vstack(parts)
cust[:, 0] = np.clip(cust[:, 0], 18, 90).round(0)
cust[:, 1] = np.clip(cust[:, 1], 12, 250).round(0)
cust[:, 2] = np.clip(cust[:, 2], 15, None).round(0)
cust[:, 3] = np.clip(cust[:, 3], 0, None).round(1)
gender = rng.choice(["Female", "Male", "Other"], n, p=[0.47, 0.49, 0.04])
cust_df = pd.DataFrame(
    {
        "age": cust[:, 0].astype(int),
        "annual_income_k": cust[:, 1].astype(int),
        "monthly_spend": cust[:, 2].astype(int),
        "avg_visits_per_month": cust[:, 3],
        "gender": gender,
    }
)
idx = rng.choice(n, size=10, replace=False)
cust_df.loc[idx, "monthly_spend"] = np.nan
write(
    "customer_segments",
    cust_df,
    {
        "label": "Customer Segments",
        "description": "400 customers with age, income, spend and visit frequency. No target column — the agent runs a K-Means cluster search and scores it with the silhouette coefficient.",
        "task": "clustering — no target",
    },
)

print("Samples written to", OUT)
