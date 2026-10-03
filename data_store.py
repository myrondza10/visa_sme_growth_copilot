"""
data_store.py — the single data source for SME Growth Co-Pilot
==============================================================
One transaction table in, everything else derived from it:

    merchants, categories, customer overlap (look-alike / network graph),
    RFM segments, contact lists, location demand/competition, peer benchmarks,
    weekly revenue (cashflow).

Usage in the app:
    from data_store import STORE, SME_PROFILES, compute_rfm, REQUIRED_COLUMNS
    STORE.load_csv("transactions.csv")
"""
from __future__ import annotations

import os
import random
import re
from collections.abc import Mapping
from datetime import datetime

import numpy as np
import pandas as pd

try:  # shows as a toast in Gradio; plain Exception when gradio isn't installed
    from gradio import Error as UserError
except ImportError:  # pragma: no cover
    class UserError(Exception):
        pass

# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------
REQUIRED_COLUMNS = ["customer_id", "merchant_name", "merchant_category", "date", "amount_aed"]
OPTIONAL_COLUMNS = [
    "transaction_id", "merchant_area", "payment_method",
    "customer_name", "customer_email", "latitude", "longitude",
]

MIN_TXNS = 30                    # merchants with fewer transactions aren't offered as "Your SME"
MIN_PEERS = 3                    # peers needed in a category to benchmark against
OPP_COMPETITION_WEIGHT = 0.6     # opportunity = demand_index - weight * competitor_index

# What the dataset preview shows (contact details / coordinates stay out of it).
DISPLAY_COLUMNS = ["transaction_id", "customer_id", "merchant_category", "date", "amount_aed", "payment_method"]

# Used only to place areas on the map when the CSV has no latitude/longitude columns.
AREA_COORDS = {
    "jumeirah": (25.2285, 55.2593), "al barsha": (25.1121, 55.2044),
    "dubai marina": (25.0805, 55.1403), "marina": (25.0805, 55.1403),
    "jlt": (25.0693, 55.1465), "business bay": (25.1877, 55.2633),
    "downtown dubai": (25.1972, 55.2744), "deira": (25.2697, 55.3095),
    "al quoz": (25.1367, 55.2268), "mirdif": (25.2159, 55.4235),
    "arabian ranches": (25.0537, 55.2708),
}

FIRST_NAMES = ["Sara", "Ahmed", "Fatima", "James", "Priya", "Omar", "Layla", "Noah",
               "Mei", "Yusuf", "Aisha", "Daniel", "Hana", "Karim", "Zainab", "Liam"]
LAST_NAMES = ["Khan", "Ali", "Smith", "Patel", "Hassan", "Brown", "Chen", "Ibrahim",
              "Kumar", "Nasser", "Rahman", "Lopez"]

def _score_recency(days: float) -> int:
    """Same cutoffs as the segments below (<=15 Champions, >60 Win-Back)."""
    if days <= 15:
        return 4
    if days <= 30:
        return 3
    if days <= 60:
        return 2
    return 1


def _score_frequency(n: float) -> int:
    """Same cutoffs as the segments below (>=6 Champions, <=2 New)."""
    if n >= 6:
        return 4
    if n >= 4:
        return 3
    if n >= 3:
        return 2
    return 1


def _quartile_score(s: pd.Series) -> pd.Series:
    """1-4 quartile score within the given customers (4 = highest). Rank-based, so ties are safe."""
    r = s.rank(method="first", ascending=True)
    return np.ceil(r / max(len(s), 1) * 4).clip(1, 4).astype(int)

# ----------------------------------------------------------------------
# RFM (moved here so benchmarking can reuse it; "today" = last date in the data)
# ----------------------------------------------------------------------
def compute_rfm(df: pd.DataFrame, as_of=None) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of or STORE.as_of or datetime.today())
    d = df.copy()
    if "date_dt" not in d.columns:
        d["date_dt"] = pd.to_datetime(d["date"])

    grouped = (
        d.groupby("customer_id")
        .agg(last_purchase=("date_dt", "max"),
             frequency=("date_dt", "count"),
             monetary=("amount_aed", "sum"))
        .reset_index()
    )
    grouped["recency_days"] = (as_of - grouped["last_purchase"]).dt.days

    def segment(row):
        if row["recency_days"] > 60:
            return "Win-Back (Lapsing)"
        if row["recency_days"] <= 15 and row["frequency"] >= 6:
            return "Champions"
        if row["frequency"] <= 2 and row["recency_days"] <= 20:
            return "New Customers"
        return "Regulars"

    grouped["segment"] = grouped.apply(segment, axis=1)
    grouped["r_score"] = grouped["recency_days"].apply(_score_recency)
    grouped["f_score"] = grouped["frequency"].apply(_score_frequency)
    grouped["m_score"] = _quartile_score(grouped["monetary"])
    grouped["rfm_score"] = (grouped["r_score"].astype(str)
                            + grouped["f_score"].astype(str)
                            + grouped["m_score"].astype(str))
    return grouped


# ----------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------
class DataStore:
    def __init__(self):
        self._all = pd.DataFrame()
        self.source = None
        self.as_of = None
        self.merchants: dict = {}
        self.warnings: list = []
        self._cust_cat = None
        self._aff_cache: dict = {}
        self._overlap_cache: dict = {}

    # ---------------- loading ----------------
    @property
    def ready(self) -> bool:
        return not self._all.empty

    def require(self):
        if not self.ready:
            raise UserError("No data loaded yet — upload a transactions CSV first.")

    def load_csv(self, path: str) -> str:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise UserError(f"Couldn't read that file as CSV: {e}")
        return self.load_dataframe(df, source=os.path.basename(path))

    def load_dataframe(self, df: pd.DataFrame, source: str = "upload") -> str:
        df = df.copy()
        df.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in df.columns]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise UserError(
                "Missing required column(s): " + ", ".join(missing)
                + ". Required: " + ", ".join(REQUIRED_COLUMNS)
            )

        n_in = len(df)
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount_aed"] = pd.to_numeric(df["amount_aed"], errors="coerce")
        for c in ("customer_id", "merchant_name", "merchant_category"):
            df[c] = df[c].astype("string").str.strip()
        df = df.dropna(subset=["customer_id", "merchant_name", "merchant_category", "date_dt", "amount_aed"])
        df = df[df["amount_aed"] > 0]
        for c in ("customer_id", "merchant_name", "merchant_category"):
            df[c] = df[c].astype(str)
        dropped = n_in - len(df)
        if df.empty:
            raise UserError("No usable rows after cleaning (check date format and amount_aed).")

        for c in ("merchant_area", "payment_method", "customer_name", "customer_email"):
            df[c] = df[c].fillna("").astype(str).str.strip() if c in df.columns else ""
        if "transaction_id" in df.columns:
            df["transaction_id"] = df["transaction_id"].astype(str)
        else:
            df["transaction_id"] = ["TXN-%07d" % i for i in range(1, len(df) + 1)]
        for c in ("latitude", "longitude"):
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

        df["amount_aed"] = df["amount_aed"].round(2)
        df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d %H:%M")

        keep = DISPLAY_COLUMNS + ["merchant_name", "merchant_area", "customer_name",
                                  "customer_email", "latitude", "longitude", "date_dt"]
        self._all = df[keep].reset_index(drop=True)
        self.source = source
        self.as_of = self._all["date_dt"].max().to_pydatetime()
        self._rebuild(dropped)
        return self.summary_md()

    def _rebuild(self, dropped: int):
        tx = self._all
        self._aff_cache.clear()
        self._overlap_cache.clear()

        self.merchants = {}
        for name, sub in tx.groupby("merchant_name"):
            areas = sub["merchant_area"][sub["merchant_area"] != ""]
            self.merchants[name] = {
                "category": sub["merchant_category"].mode().iat[0],
                "area": areas.mode().iat[0] if len(areas) else "",
                "avg_ticket": float(sub["amount_aed"].mean()),
                "n_customers": int(sub["customer_id"].nunique()),
                "n_txn": int(len(sub)),
            }
        self._cust_cat = pd.crosstab(tx["customer_id"], tx["merchant_category"]) > 0

        w = []
        if dropped:
            w.append(f"{dropped:,} row(s) dropped (bad date/amount, missing ids, or amount <= 0).")
        if not (tx["merchant_area"] != "").any():
            w.append("`merchant_area` is empty — Location Analytics is unavailable.")
        span_days = (tx["date_dt"].max() - tx["date_dt"].min()).days
        if span_days < 42:
            w.append(f"Only {span_days} days of history — cashflow forecasting needs at least 6 weeks.")
        elif span_days < 540:
            w.append("Under ~18 months of history — the cashflow forecast can use trend + monthly seasonality "
                     "only. Annual patterns (peak season, summer dip, Ramadan/Eid) need 18+ months.")
        by_cat = self.merchants_by_category()
        thin = [c for c, m in by_cat.items() if len(m) <= MIN_PEERS]
        if thin:
            w.append(f"Benchmarking needs >{MIN_PEERS} merchants per category; too few in: " + ", ".join(sorted(thin)) + ".")
        if len(by_cat) < 2:
            w.append("Only one merchant category — Look-Alike Finder has no partners to suggest.")
        self.warnings = w

    def summary_md(self) -> str:
        if not self.ready:
            return "⚠️ No data loaded. Upload a transactions CSV."
        tx = self._all
        lines = [
            f"✅ **{self.source}** — {len(tx):,} transactions · {tx['customer_id'].nunique():,} customers · "
            f"{len(self.merchants):,} merchants · {tx['merchant_category'].nunique()} categories · "
            f"{tx['date_dt'].min():%d %b %Y} → {tx['date_dt'].max():%d %b %Y}"
        ]
        lines += [f"- ⚠️ {w}" for w in self.warnings]
        return "\n".join(lines)

    # ---------------- lookups ----------------
    def sme_names(self) -> list:
        big = [m for m, p in self.merchants.items() if p["n_txn"] >= MIN_TXNS]
        return sorted(big or self.merchants)

    @property
    def profiles(self):
        return _Profiles(self)

    def profile(self, name: str) -> dict:
        self.require()
        if name not in self.merchants:
            raise UserError(f"Unknown merchant '{name}' — pick one from the dropdown.")
        return self.merchants[name]

    def merchants_by_category(self) -> dict:
        out: dict = {}
        for m, p in self.merchants.items():
            out.setdefault(p["category"], []).append(m)
        return out

    def transactions_for(self, name: str) -> pd.DataFrame:
        self.profile(name)
        sub = self._all[self._all["merchant_name"] == name]
        return sub.sort_values("date_dt", ascending=False)[DISPLAY_COLUMNS].reset_index(drop=True)

    def weekly_revenue(self, name: str) -> pd.Series:
        """Revenue per week (index = Monday of the week). The dataset's first week is
        dropped when it is partial; callers drop the final (usually partial) week."""
        self.profile(name)
        sub = self._all[self._all["merchant_name"] == name]
        wk = sub["date_dt"].dt.to_period("W").apply(lambda p: p.start_time)
        s = sub.groupby(wk)["amount_aed"].sum().sort_index()
        first_day = self._all["date_dt"].min().normalize()
        if len(s) and first_day.dayofweek != 0:  # data doesn't start on a Monday -> week 1 is partial
            s = s.iloc[1:]
        return s

    def category_avg_ticket(self, category: str, default: float = 150.0) -> float:
        sub = self._all[self._all["merchant_category"] == category]
        return float(sub["amount_aed"].mean()) if len(sub) else default

    # ---------------- preview ----------------
    def preview(self, n: int = 100) -> pd.DataFrame:
        """Latest `n` rows of the loaded dataset (all merchants). Contact
        details and coordinates are left out."""
        if not self.ready:
            return pd.DataFrame({"Message": ["No data loaded yet — upload a CSV."]})
        cols = ["transaction_id", "customer_id", "merchant_name", "merchant_category",
                "merchant_area", "date", "amount_aed", "payment_method"]
        return (self._all.sort_values("date_dt", ascending=False)[cols]
                .head(int(n)).reset_index(drop=True))

    def preview_caption(self, n: int = 100) -> str:
        if not self.ready:
            return "*No data loaded yet — upload a CSV above.*"
        return (f"Showing the latest **{min(int(n), len(self._all)):,}** of **{len(self._all):,}** rows "
                f"from **{self.source}** — all merchants; these are the rows every tab analyses. "
                "Customer names/emails are hidden here.")

    # ---------------- contacts ----------------
    def directory_for(self, name: str) -> dict:
        """{customer_id: {"name", "email"}}. Uses the CSV's customer_name /
        customer_email where present; fills gaps with a deterministic
        synthetic @example.com contact."""
        self.profile(name)
        sub = self._all[self._all["merchant_name"] == name]
        d = (sub[["customer_id", "customer_name", "customer_email"]]
             .replace("", np.nan).groupby("customer_id").first().fillna(""))
        out = {}
        for cid, row in d.iterrows():
            nm, em = row["customer_name"], row["customer_email"]
            if not (nm and em):
                rnd = random.Random(f"{name}::{cid}")
                first, last = rnd.choice(FIRST_NAMES), rnd.choice(LAST_NAMES)
                slug = re.sub(r"\W", "", cid).lower()
                nm = nm or f"{first} {last}"
                em = em or f"{first.lower()}.{last.lower()}.{slug}@example.com"
            out[cid] = {"name": nm, "email": em}
        return out

    # ---------------- look-alike / affinity ----------------
    def affinity(self, name: str) -> dict:
        """{other_category: share of this merchant's customers who also shop
        in that category}, sorted high -> low."""
        self.profile(name)
        if name not in self._aff_cache:
            custs = self._all.loc[self._all["merchant_name"] == name, "customer_id"].unique()
            share = self._cust_cat.loc[custs].mean().drop(self.merchants[name]["category"], errors="ignore")
            share = share[share > 0].round(2).sort_values(ascending=False)
            self._aff_cache[name] = share.to_dict()
        return self._aff_cache[name]

    def merchant_overlap(self, name: str) -> dict:
        """{merchant: share of this merchant's customers who also shop there}."""
        self.profile(name)
        if name not in self._overlap_cache:
            custs = self._all.loc[self._all["merchant_name"] == name, "customer_id"].unique()
            others = self._all[self._all["customer_id"].isin(custs) & (self._all["merchant_name"] != name)]
            counts = others.groupby("merchant_name")["customer_id"].nunique()
            self._overlap_cache[name] = (counts / max(len(custs), 1)).round(3).to_dict()
        return self._overlap_cache[name]

    def partner_for(self, name: str, category: str):
        """Best real partner in `category`: highest shared-customer overlap."""
        ov = self.merchant_overlap(name)
        cands = [m for m in self.merchants_by_category().get(category, []) if m != name]
        return max(cands, key=lambda m: ov.get(m, 0.0)) if cands else None

    # ---------------- location ----------------
    def area_stats(self, name: str):
        """Returns (DataFrame, home_area). Columns: Area, lat, lon, is_home,
        demand_index (0-100, share of the busiest area's spend),
        competitor_count (other same-category merchants in the area),
        opportunity_score."""
        home = self.profile(name)
        cat, home_area = home["category"], home["area"]
        df = self._all[self._all["merchant_area"] != ""]
        if df.empty or not home_area:
            raise UserError("Location Analytics needs the `merchant_area` column filled in for this merchant.")

        spend = df.groupby("merchant_area")["amount_aed"].sum()
        comp = (df[(df["merchant_category"] == cat) & (df["merchant_name"] != name)]
                .groupby("merchant_area")["merchant_name"].nunique().reindex(spend.index, fill_value=0))
        ll = df.groupby("merchant_area")[["latitude", "longitude"]].median()

        rows = []
        for area in spend.index:
            lat, lon = ll.loc[area, "latitude"], ll.loc[area, "longitude"]
            if pd.isna(lat) or pd.isna(lon):
                lat, lon = AREA_COORDS.get(area.lower(), (np.nan, np.nan))
            if pd.isna(lat):
                continue  # can't place it on a map
            rows.append({
                "Area": area, "lat": float(lat), "lon": float(lon),
                "is_home": area == home_area,
                "demand_index": round(100 * spend[area] / spend.max(), 1),
                "competitor_count": int(comp[area]),
            })
        area_df = pd.DataFrame(rows)
        if not area_df["is_home"].any():
            raise UserError(f"No coordinates for '{home_area}'. Add latitude/longitude columns to the CSV.")
        if len(area_df) < 2:
            raise UserError("Need at least two areas with coordinates to compare locations.")

        comp_idx = 100 * area_df["competitor_count"] / max(area_df["competitor_count"].max(), 1)
        area_df["opportunity_score"] = (area_df["demand_index"] - OPP_COMPETITION_WEIGHT * comp_idx).round(1)
        return area_df, home_area

    # ---------------- benchmarking ----------------
    def merchant_metrics(self, name: str) -> dict:
        self.profile(name)
        sub = self._all[self._all["merchant_name"] == name]
        rfm = compute_rfm(sub)
        total = len(rfm)
        return {
            "avg_ticket_aed": float(sub["amount_aed"].mean()),
            "retention_rate_pct": float(100 * (1 - (rfm["segment"] == "Win-Back (Lapsing)").sum() / total)),
            "visit_frequency": float(rfm["frequency"].mean()),
            "repeat_customer_rate_pct": float(100 * (rfm["frequency"] >= 2).sum() / total),
        }

    def peer_benchmarks(self, name: str):
        """({metric: (mean, std)}, n_peers) across OTHER merchants in the same
        category, or None if there aren't enough peers."""
        cat = self.profile(name)["category"]
        peers = [m for m in self.merchants_by_category().get(cat, [])
                 if m != name and self.merchants[m]["n_txn"] >= MIN_TXNS]
        if len(peers) < MIN_PEERS:
            return None
        table = pd.DataFrame([self.merchant_metrics(m) for m in peers])
        return {k: (float(table[k].mean()), float(table[k].std(ddof=0))) for k in table.columns}, len(peers)


class _Profiles(Mapping):
    """Dict-like view so existing `SME_PROFILES[name]["category"]` and
    `list(SME_PROFILES.keys())` keep working, backed by the store."""

    def __init__(self, store: DataStore):
        self._s = store

    def __getitem__(self, name):
        return self._s.profile(name)

    def __iter__(self):
        return iter(self._s.sme_names())

    def __len__(self):
        return len(self._s.sme_names())


STORE = DataStore()
SME_PROFILES = STORE.profiles