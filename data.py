"""Synthetic food-delivery orders (inspired by public delivery-time datasets)."""
import numpy as np
import pandas as pd

FEATURES = ["distance_km", "prep_time_min", "num_items", "hour", "is_raining", "rider_experience_months"]
TARGET = "delivery_minutes"


def make_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distance = np.clip(rng.gamma(2.2, 1.6, n_rows), 0.3, 20)             # km from restaurant to customer
    prep = np.clip(rng.normal(15, 5, n_rows), 4, 45)                     # kitchen preparation time
    items = rng.integers(1, 12, n_rows)
    hour = rng.integers(10, 24, n_rows)                                   # open 10:00 - 23:59
    raining = (rng.random(n_rows) < 0.2).astype(int)
    experience = np.clip(rng.exponential(14, n_rows), 0, 96)              # months riding for us

    rush = ((hour >= 12) & (hour <= 13)) | ((hour >= 18) & (hour <= 20))
    speed_min_per_km = 2.6 * np.where(rush, 1.45, 1.0) * np.where(raining == 1, 1.3, 1.0)
    minutes = (
        prep
        + distance * speed_min_per_km
        + 0.4 * items
        + 4.0 * np.exp(-experience / 6)                                   # new riders get lost more
        + 3.0 * (rush & (distance > 6))                                   # long trips in traffic
        + rng.normal(0, 3, n_rows)
    )
    return pd.DataFrame({
        "distance_km": distance.round(2),
        "prep_time_min": prep.round(1),
        "num_items": items,
        "hour": hour,
        "is_raining": raining,
        "rider_experience_months": experience.round(1),
        TARGET: np.clip(minutes, 8, None).round(1),
    })
