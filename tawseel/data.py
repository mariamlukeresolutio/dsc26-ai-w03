"""Synthetic Tawseel parcels. Provided by DSC. Do not edit."""
from __future__ import annotations

import numpy as np
import pandas as pd

AREAS = {  # area: extra log-odds of failure
    "Nasr City": 0.0, "Maadi": -0.3, "Heliopolis": -0.2, "Zamalek": -0.5,
    "Dokki": -0.1, "Mohandessin": -0.1, "Shubra": 0.4, "Imbaba": 0.5,
    "Ain Shams": 0.4, "Helwan": 0.6, "New Cairo": -0.2, "6th of October": 0.2,
    "Sheikh Zayed": -0.1, "Faisal": 0.5, "Haram": 0.4, "Matareya": 0.6,
}
VAGUE = ["ورا الجامع جنب الفرن", "behind the mosque near the bakery",
         "gamb el saydaleya", "قصاد المدرسة", "near the big kiosk", "odam el ahwa"]
PRECISE = ["12 شارع مكرم عبيد الدور 3 شقة 5", "Building 14, Street 9, Apt 22",
           "3omara 7 share3 el nasr dor 2", "45 El Tahrir St, floor 4",
           "عمارة 8 شارع 105 شقة 12", "Villa 31, District 2"]
SLOTS = {"morning": -0.2, "afternoon": 0.0, "evening": 0.35}
NOTES_FAIL = ["customer refused", "not home", "phone off", "wrong address", "العميل رفض"]
NOTES_OK = ["delivered", "تم التسليم", "left with doorman", ""]


def make_parcels(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_customers = max(50, n // 4)
    cust_propensity = rng.normal(0.0, 0.9, n_customers)
    customer = rng.integers(0, n_customers, n)
    area = rng.choice(list(AREAS), n)
    merchant = rng.choice([f"M{m:03d}" for m in range(40)], n)
    slot = rng.choice(list(SLOTS), n, p=[0.35, 0.4, 0.25])
    start = np.datetime64("2026-06-01T08:00")
    minutes = np.sort(rng.integers(0, 120 * 24 * 60, n))
    created = start + minutes.astype("timedelta64[m]")
    cod = np.round(rng.lognormal(6.3, 0.9, n), 0)
    cod[rng.random(n) < 0.04] = np.nan
    items = rng.integers(1, 6, n)
    vague = rng.random(n) < 0.3
    address = np.where(vague, rng.choice(VAGUE, n), rng.choice(PRECISE, n))
    address = np.where(rng.random(n) < 0.03, "", address)

    logit = (-2.4 + cust_propensity[customer]
             + np.array([AREAS[a] for a in area])
             + np.array([SLOTS[s] for s in slot])
             + 0.55 * np.log1p(np.nan_to_num(cod, nan=500.0) / 500.0)
             + 0.9 * vague + 0.6 * (address == ""))
    failed = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    notes = np.where(failed == 1, rng.choice(NOTES_FAIL, n), rng.choice(NOTES_OK, n))
    delivered_at = pd.Series(created + rng.integers(60, 600, n).astype("timedelta64[m]"))
    delivered_at[failed == 1] = pd.NaT
    return pd.DataFrame({
        "parcel_id": [f"P{seed:02d}{i:06d}" for i in range(n)],
        "customer_id": [f"C{c:05d}" for c in customer],
        "merchant_id": merchant, "area": area,
        "created_at": pd.to_datetime(created), "promised_slot": slot,
        "cod_amount": cod, "n_items": items, "address_text": address,
        "courier_notes": notes, "delivered_at": delivered_at, "failed": failed,
    })
