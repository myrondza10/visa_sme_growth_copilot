"""
make_synthetic_data.py — synthetic UAE transactions with realistic, DIFFERENT seasonality
per business, for the SME Growth Co-Pilot.

    python make_synthetic_data.py                       # -> transactions.csv (30 months)
    python make_synthetic_data.py --months 36 --scale 1 # more history / more volume

Every merchant's daily volume = base level x growth trend x day-of-week x annual (heat/tourist)
cycle x calendar events (Ramadan, Eid, DSF, White Friday, New Year, payday, back-to-school...)
x random noise. Each *category* has its own profile and each *merchant* gets its own strength,
growth and area (tourist areas swing harder with the seasons), so no two series look alike.

Customers come from "lifestyle clusters" so the Look-Alike overlap numbers vary by category.
All names/emails are fake (@example.com). Columns match data_store.py.
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--months", type=int, default=36)
ap.add_argument("--end", default=None, help="last date, YYYY-MM-DD (default: today)")
ap.add_argument("--scale", type=float, default=0.33, help="multiplies transaction volume")
ap.add_argument("--customers", type=int, default=10000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="transactions.csv")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)

END = pd.Timestamp(args.end) if args.end else pd.Timestamp(dt.date.today())
START = END - pd.DateOffset(months=args.months)
DAYS = pd.date_range(START.normalize(), END.normalize(), freq="D")
T = len(DAYS)

# ----------------------------------------------------------------------
# Calendar features (Ramadan / Eid dates are approximate Gregorian dates)
# ----------------------------------------------------------------------
RAMADAN = [("2023-03-23", "2023-04-20"), ("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-29"),
           ("2026-02-18", "2026-03-19"), ("2027-02-08", "2027-03-09")]
EID_FITR = ["2023-04-21", "2024-04-10", "2025-03-30", "2026-03-20", "2027-03-10"]   # 3 days
EID_ADHA = ["2023-06-28", "2024-06-16", "2025-06-06", "2026-05-27", "2027-05-16"]   # 4 days


def flag_ranges(ranges):
    f = np.zeros(T, bool)
    for a, b in ranges:
        f |= (DAYS >= pd.Timestamp(a)) & (DAYS <= pd.Timestamp(b))
    return f


def flag_start_len(starts, length, shift=0):
    return flag_ranges([(pd.Timestamp(s) + pd.Timedelta(days=shift),
                         pd.Timestamp(s) + pd.Timedelta(days=shift + length - 1)) for s in starts])


dow = DAYS.dayofweek.values                      # Mon=0 .. Sun=6
dom = DAYS.day.values
doy = DAYS.dayofyear.values
month = DAYS.month.values

RAMADAN_F = flag_ranges(RAMADAN)
EID_F = flag_start_len(EID_FITR, 3) | flag_start_len(EID_ADHA, 4)
PRE_EID_F = flag_start_len(EID_FITR, 7, shift=-7)                    # the week before Eid al-Fitr
DSF_F = ((month == 12) & (dom >= 15)) | ((month == 1) & (dom <= 29))  # Dubai Shopping Festival
NATDAY_F = (month == 12) & (dom >= 1) & (dom <= 3)                   # UAE National Day
SCHOOL_F = ((month == 8) & (dom >= 15)) | ((month == 9) & (dom <= 12))
PAYDAY_F = (dom >= 25) | (dom <= 3)


def white_friday(year):
    d = pd.Timestamp(year=year, month=11, day=30)
    while d.dayofweek != 4:
        d -= pd.Timedelta(days=1)
    return d


WF_F = flag_ranges([(white_friday(y) - pd.Timedelta(days=2), white_friday(y) + pd.Timedelta(days=3))
                    for y in range(START.year, END.year + 1)])
# New-Year resolution bump that decays through Jan/Feb
NY_DECAY = np.where(np.isin(month, [1, 2]), np.exp(-((doy - 1) % 366) / 18.0), 0.0)
# Smooth annual cycle: +1 around mid-January (peak season), -1 around mid-July (summer heat)
HEAT = np.cos(2 * np.pi * (doy - 15) / 365.25)

# ----------------------------------------------------------------------
# Category profiles. Multipliers are relative to a normal day (1.0).
#   dow     Mon..Sun          heat   annual amplitude (+ = winter/tourist peak, - = summer peak)
#   base    txns/day range    ticket mean AED          ny  New-Year decay amplitude
# ----------------------------------------------------------------------
PROFILES = {
    "Food & Beverage": dict(dow=[.85, .85, .9, 1.0, 1.3, 1.4, 1.1], heat=.16, ramadan=.88, eid=1.35, pre_eid=1.0,
                            dsf=1.10, wf=1.0, ny=0, payday=.08, school=1.0, natday=1.25, base=(7, 13), ticket=68),
    "Fitness & Wellness": dict(dow=[1.25, 1.1, 1.1, 1.05, .8, .7, .8], heat=.10, ramadan=.80, eid=.70, pre_eid=1.0,
                               dsf=1.0, wf=1.0, ny=.55, payday=.15, school=1.15, natday=1.0, base=(4, 8), ticket=190),
    "Personal Care": dict(dow=[.8, .85, .9, 1.0, 1.25, 1.4, 1.0], heat=.06, ramadan=1.0, eid=.9, pre_eid=1.5,
                          dsf=1.05, wf=1.0, ny=0, payday=.15, school=1.0, natday=1.1, base=(4, 8), ticket=140),
    "Retail - Fashion": dict(dow=[.8, .8, .85, .95, 1.3, 1.45, 1.15], heat=.10, ramadan=.95, eid=1.1, pre_eid=1.4,
                             dsf=1.35, wf=1.8, ny=0, payday=.15, school=1.25, natday=1.2, base=(4, 8), ticket=260),
    "Grocery": dict(dow=[1.0, .9, .9, .95, 1.2, 1.25, 1.1], heat=-.03, ramadan=1.2, eid=1.2, pre_eid=1.1,
                    dsf=1.0, wf=1.05, ny=0, payday=.12, school=1.05, natday=1.05, base=(10, 18), ticket=95),
    "Entertainment": dict(dow=[.7, .7, .8, .95, 1.4, 1.5, 1.2], heat=.30, ramadan=.80, eid=1.8, pre_eid=1.0,
                          dsf=1.25, wf=1.0, ny=0, payday=.05, school=1.0, natday=1.5, base=(5, 10), ticket=120),
    "Health & Pharmacy": dict(dow=[1.05, 1.0, 1.0, 1.0, .95, .9, .9], heat=.10, ramadan=1.0, eid=.9, pre_eid=1.0,
                              dsf=1.0, wf=1.0, ny=0, payday=.03, school=1.1, natday=1.0, base=(7, 12), ticket=85),
    "Specialty Coffee": dict(dow=[1.1, 1.1, 1.1, 1.05, .95, .95, .9], heat=.18, ramadan=.70, eid=1.1, pre_eid=1.0,
                             dsf=1.05, wf=1.0, ny=0, payday=.05, school=1.1, natday=1.15, base=(12, 20), ticket=38),
    # New categories
    "Spa & Wellness": dict(dow=[.75, .8, .85, .95, 1.3, 1.45, 1.3], heat=.22, ramadan=.85, eid=1.2, pre_eid=1.5,
                           dsf=1.15, wf=1.0, ny=.30, payday=.20, school=1.0, natday=1.1, base=(3, 6), ticket=280),
    "Sportswear": dict(dow=[.8, .8, .9, .95, 1.25, 1.4, 1.1], heat=-.05, ramadan=.90, eid=1.0, pre_eid=1.1,
                       dsf=1.20, wf=1.90, ny=.60, payday=.12, school=1.30, natday=1.15, base=(3, 6), ticket=210),
}
# Ticket size also moves with events (bigger baskets at Eid / Ramadan / sales)
TICKET_EVENT = {"ramadan": 1.10, "eid": 1.15, "dsf": 1.08, "wf": 1.15, "pre_eid": 1.10}

AREAS = {  # (lat, lon, tourist factor: how hard the seasons swing there, demand multiplier)
    "Dubai Marina":      (25.0805, 55.1403, 1.30, 1.25),
    "Downtown Dubai":    (25.1972, 55.2744, 1.35, 1.30),
    "Jumeirah":          (25.2285, 55.2593, 1.25, 1.15),
    "Business Bay":      (25.1877, 55.2633, 1.10, 1.10),
    "JLT":               (25.0693, 55.1465, 1.00, 1.00),
    "Deira":             (25.2697, 55.3095, 1.00, 0.95),
    "Al Quoz":           (25.1367, 55.2268, 0.90, 0.85),
    "Al Barsha":         (25.1121, 55.2044, 0.85, 0.95),
    "Mirdif":            (25.2159, 55.4235, 0.70, 0.90),
    "Arabian Ranches":   (25.0537, 55.2708, 0.70, 0.80),
    "Palm Jumeirah":     (25.1124, 55.1390, 1.40, 1.35),
    "DIFC":              (25.2132, 55.2822, 1.20, 1.20),
}

# (name, area, heat multiplier). A negative multiplier flips the cycle (summer-peak businesses).
MERCHANTS = {
    "Food & Beverage": [
        ("Al Reef Grill",        "Deira",          1),
        ("Saffron Bistro",       "Downtown Dubai", 1.2),
        ("Marina Burger Co",     "Dubai Marina",   1.2),
        ("Zaatar House",         "Mirdif",          .6),
        ("Palm Street Kitchen",  "Jumeirah",        1),
        ("Bistro Coastal",       "Downtown Dubai", 1.1),
        ("Levantine Table",      "Business Bay",    1),
        ("Bean & Batter Café",   "Dubai Marina",   1.1),
        ("The Jumeirah Café",    "Jumeirah",         .9),
        ("Artisan Bakery JLT",   "JLT",             .8),
    ],
    "Fitness & Wellness": [
        ("Iron Oasis Gym",       "Al Quoz",        1),
        ("Zen Studio Yoga",      "Jumeirah",        1),
        ("PulseFit Dubai",       "Business Bay",    1),
        ("CrossFit Barsha",      "Al Barsha",       .8),
        ("Aqua Fit Club",        "Dubai Marina",    1),
        ("Apex Crossfit",        "Al Quoz",         1),
        ("Zen Pilates Room",     "Jumeirah",        1),
        ("PureCore Gym",         "Dubai Marina",    1),
        ("Boutique Gym Al Barsha","Al Barsha",      .8),
        ("The Movement Studio",  "DIFC",           1.1),
    ],
    "Personal Care": [
        ("Velvet Salon",         "Jumeirah",        1),
        ("The Barber Lounge",    "Business Bay",    1),
        ("Glow Beauty Bar",      "JLT",             1),
        ("Nail Atelier",         "Dubai Marina",    1),
        ("Majlis Grooming",      "Mirdif",          1),
        ("Hair & Beauty Salon Marina", "Dubai Marina", 1),
        ("The Grooming Lounge",  "Downtown Dubai",  1),
        ("Lash & Brow Studio",   "Jumeirah",        1),
        ("Men's Grooming Hub",   "DIFC",           1.1),
        ("Beauty Secret",        "Al Barsha",       .9),
    ],
    "Retail - Fashion": [
        ("Souk Threads",         "Deira",           1),
        ("Urban Abaya",          "Al Barsha",       1),
        ("Marina Mode",          "Dubai Marina",   1.2),
        ("Desert Denim",         "Al Quoz",         1),
        ("Kandura & Co",         "Downtown Dubai",  1),
        ("Sandy Lane Boutique",  "Jumeirah",       1.1),
        ("Loom Concept Store",   "Al Quoz",         1),
        ("Style District JLT",   "JLT",             .9),
        ("The Edit Room",        "DIFC",           1.2),
        ("Silk & Thread",        "Palm Jumeirah",  1.3),
    ],
    "Grocery": [
        ("Fresh Basket",         "JLT",             1),
        ("Al Noor Mart",         "Deira",           1),
        ("Green Valley Grocers", "Arabian Ranches", 1),
        ("Corner Cart",          "Al Barsha",       1),
        ("Harvest Hub",          "Mirdif",          1),
        ("Deira Fresh Hub",      "Deira",           1),
        ("Fresh Corner Market",  "Al Barsha",       1),
        ("Organic Food Store",   "Jumeirah",        1),
        ("Daily Pantry",         "Business Bay",   1.1),
        ("Al Wasl Supermarket",  "Jumeirah",        1),
    ],
    "Entertainment": [
        ("Skyline Cinema",       "Business Bay",   -.5),
        ("Desert Karting",       "Al Quoz",        1.2),
        ("Splash Kids Zone",     "Mirdif",         -.9),
        ("Marina Escape Rooms",  "Dubai Marina",    1),
        ("Palm Bowling",         "Jumeirah",        .6),
        ("Golf Simulator Club",  "Al Quoz",         1),
        ("Bowl & Roll JLT",      "JLT",             .7),
        ("VR World UAE",         "Business Bay",    1),
        ("CineLounge Marina",    "Dubai Marina",    1),
        ("Arcade District",      "Downtown Dubai", 1.1),
    ],
    "Health & Pharmacy": [
        ("CityCare Pharmacy",    "Deira",           1),
        ("Wellspring Chemist",   "Al Barsha",       1),
        ("HealthHub Pharmacy",   "JLT",             1),
        ("Family Pharma",        "Arabian Ranches", 1),
        ("Vita Clinic Shop",     "Business Bay",    1),
        ("Marina Health Point",  "Dubai Marina",    1),
        ("Wellness First Pharmacy","Jumeirah",      1),
        ("CityMed Pharmacy",     "Deira",           1),
        ("VitalCare Pharmacy",   "Al Barsha",       1),
        ("Al Shifa Pharmacy",    "Mirdif",          .9),
    ],
    "Specialty Coffee": [
        ("Brew Lab",             "Business Bay",    1),
        ("Cardamom Roasters",    "Jumeirah",       1.2),
        ("Dune Coffee",          "Downtown Dubai", 1.2),
        ("Third Floor Espresso", "JLT",             1),
        ("Qahwa Corner",         "Al Quoz",         .8),
        ("Roast & Grind Jumeirah","Jumeirah",      1.1),
        ("Bean Theory",          "JLT",             1),
        ("Nomad Coffee Roasters","Al Quoz",         .8),
        ("Cortado Corner",       "Downtown Dubai", 1.1),
        ("Espresso Lab",         "Business Bay",   1.1),
    ],
    "Spa & Wellness": [
        ("Hammams & Care Marina","Dubai Marina",   1.3),
        ("Serenity Spa House",   "Jumeirah",       1.2),
        ("Still Water Spa",      "Arabian Ranches", 1),
        ("Calm Studio",          "Business Bay",    1),
        ("The Spa at Marina",    "Palm Jumeirah",  1.4),
        ("Aqua Spa",             "Dubai Marina",   1.2),
        ("Rejuvenate Wellness",  "DIFC",           1.1),
        ("Oasis Hammam",         "Deira",           .9),
        ("Bloom Spa",            "Jumeirah",       1.1),
        ("Azure Beauty Retreat", "Downtown Dubai", 1.2),
    ],
    "Sportswear": [
        ("Stride Lab",           "Business Bay",    1),
        ("Court & Track",        "Downtown Dubai",  1),
        ("Rally Sports Co.",     "Dubai Marina",   1.2),
        ("Peak Gear Outlet",     "Mirdif",          .7),
        ("Motion Athletics UAE", "Al Barsha",       .9),
        ("Sprint Zone",          "JLT",             1),
        ("Active Life Store",    "Jumeirah",        1),
        ("The Locker Room",      "DIFC",           1.1),
        ("Endurance Sports",     "Al Quoz",         .9),
        ("Urban Active",         "Palm Jumeirah",  1.3),
    ],
}

# Lifestyle clusters drive which merchants share customers (-> look-alike overlap)
CLUSTERS = {
    "Wellness":  {"Fitness & Wellness": 1, "Health & Pharmacy": .8, "Specialty Coffee": .8,
                  "Personal Care": .5, "Grocery": .4, "Spa & Wellness": .9, "Sportswear": .6},
    "Lifestyle": {"Retail - Fashion": 1, "Entertainment": .9, "Food & Beverage": .9,
                  "Personal Care": .7, "Specialty Coffee": .5, "Spa & Wellness": .7, "Sportswear": .5},
    "Household": {"Grocery": 1, "Health & Pharmacy": .8, "Food & Beverage": .7,
                  "Entertainment": .4, "Retail - Fashion": .3},
    "Active":    {"Sportswear": 1, "Fitness & Wellness": .9, "Spa & Wellness": .6,
                  "Specialty Coffee": .5, "Health & Pharmacy": .4},
}
HOURS = {
    "Food & Beverage":   [12, 13, 14, 19, 20, 21, 22],
    "Fitness & Wellness":[6, 7, 8, 17, 18, 19, 20],
    "Personal Care":     [11, 12, 14, 15, 16, 17, 18],
    "Retail - Fashion":  [12, 14, 16, 18, 19, 20, 21],
    "Grocery":           [8, 10, 12, 17, 18, 19, 20],
    "Entertainment":     [14, 16, 18, 19, 20, 21, 22],
    "Health & Pharmacy": [9, 11, 13, 17, 18, 19, 21],
    "Specialty Coffee":  [7, 8, 8, 9, 10, 15, 16],
    "Spa & Wellness":    [10, 11, 12, 14, 15, 16, 17],
    "Sportswear":        [11, 12, 14, 16, 18, 19, 20],
}
PAYMENT = ["Visa Credit", "Visa Debit", "Visa Contactless", "Visa Chip & PIN",
           "Mastercard", "Apple Pay (Visa)", "Visa Digital Wallet", "Cash"]
FIRST = ["Sara", "Ahmed", "Fatima", "James", "Priya", "Omar", "Layla", "Noah", "Mei", "Yusuf",
         "Aisha", "Daniel", "Hana", "Karim", "Zainab", "Liam", "Rania", "Tariq", "Elena", "Raj",
         "Amira", "Sami", "Maya", "Rohan", "Khalid", "Nadia", "Hassan", "Leila", "Adam", "Sofia"]
LAST  = ["Khan", "Ali", "Smith", "Patel", "Hassan", "Brown", "Chen", "Ibrahim", "Kumar", "Nasser",
         "Rahman", "Lopez", "Farouk", "Haddad", "Novak", "Sharma", "Al Rashid", "Malik", "Nguyen", "Osei"]

# ----------------------------------------------------------------------
# Customer universe
# ----------------------------------------------------------------------
U = args.customers
cust_ids    = np.array([f"C{i:05d}" for i in range(1, U + 1)])
cust_names  = np.array([f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(U)])
cust_emails = np.array([f"{n.lower().replace(' ', '.')}.{i:05d}@example.com"
                        for i, n in enumerate(cust_names, 1)])
cluster_names = list(CLUSTERS)
primary   = rng.integers(0, len(cluster_names), U)
secondary = np.where(rng.random(U) < .4, rng.integers(0, len(cluster_names), U), -1)


def category_weights(cat):
    w = np.full(U, 0.06)
    for arr in (primary, secondary):
        for k, name in enumerate(cluster_names):
            w[arr == k] = np.maximum(w[arr == k], CLUSTERS[name].get(cat, 0.06))
    return w ** 1.5


# ----------------------------------------------------------------------
# Generate each merchant
# ----------------------------------------------------------------------
frames = []
for cat, merchants in MERCHANTS.items():
    P = PROFILES[cat]
    cat_w = category_weights(cat)
    dow_w = np.array(P["dow"])[dow]
    for name, area, heat_mult in merchants:
        lat0, lon0, tourist, demand = AREAS[area]
        base     = rng.uniform(*P["base"]) * demand * args.scale
        growth   = rng.uniform(-.06, .40)          # per-year trend
        strength = rng.uniform(.7, 1.3)             # how seasonal this merchant is
        heat_amp = P["heat"] * heat_mult * tourist * strength

        trend = (1 + growth) ** (np.arange(T) / 365.25)
        mult  = (trend * dow_w * (1 + heat_amp * HEAT) * (1 + P["payday"] * PAYDAY_F)
                 * (1 + P["ny"] * NY_DECAY))
        for flag, key in ((RAMADAN_F, "ramadan"), (EID_F, "eid"), (PRE_EID_F, "pre_eid"),
                          (DSF_F, "dsf"), (WF_F, "wf"), (SCHOOL_F, "school"), (NATDAY_F, "natday")):
            mult = mult * np.where(flag, 1 + (P[key] - 1) * strength, 1.0)
        ticket_mult = np.ones(T)
        for flag, key in ((RAMADAN_F, "ramadan"), (EID_F, "eid"), (DSF_F, "dsf"),
                          (WF_F, "wf"), (PRE_EID_F, "pre_eid")):
            ticket_mult = ticket_mult * np.where(flag, TICKET_EVENT[key], 1.0)

        lam    = base * mult * rng.gamma(25, 1 / 25, T)   # day-level noise (overdispersion)
        counts = rng.poisson(np.clip(lam, 0, None))
        n      = int(counts.sum())
        day_idx = np.repeat(np.arange(T), counts)

        # Customer pool + lifecycle (joiners, churners, loyal heavy visitors)
        pool_n = int(np.clip(base * 120, 500, 2500))
        pool   = rng.choice(U, pool_n, replace=False, p=cat_w / cat_w.sum())
        prop   = rng.gamma(.9, 1.0, pool_n)
        join   = np.where(rng.random(pool_n) < .55, -10**6, rng.integers(0, T, pool_n))
        churn  = np.where(rng.random(pool_n) < .22,
                          np.maximum(join, 0) + rng.exponential(450, pool_n), 10**6)
        pick   = rng.choice(pool_n, n, p=prop / prop.sum())
        for _ in range(10):                            # re-draw customers not active that day
            bad = ~((join[pick] <= day_idx) & (day_idx <= churn[pick]))
            if not bad.any():
                break
            pick[bad] = rng.choice(pool_n, int(bad.sum()), p=prop / prop.sum())
        keep         = (join[pick] <= day_idx) & (day_idx <= churn[pick])
        pick, day_idx = pick[keep], day_idx[keep]
        n  = len(pick)
        cu = pool[pick]

        amount = (rng.lognormal(np.log(P["ticket"] * rng.uniform(.8, 1.25)) - .5 * .45**2, .45, n)
                  * ticket_mult[day_idx])
        hours = rng.choice(HOURS[cat], n)
        ts    = (DAYS[day_idx]
                 + pd.to_timedelta(hours, unit="h")
                 + pd.to_timedelta(rng.integers(0, 60, n), unit="m"))
        frames.append(pd.DataFrame({
            "customer_id":       cust_ids[cu],
            "merchant_name":     name,
            "merchant_category": cat,
            "merchant_area":     area,
            "date":              ts,
            "amount_aed":        amount.round(2),
            "payment_method":    rng.choice(PAYMENT, n),
            "customer_name":     cust_names[cu],
            "customer_email":    cust_emails[cu],
            "latitude":          round(lat0 + rng.normal(0, .0015), 5),
            "longitude":         round(lon0 + rng.normal(0, .0015), 5),
        }))

df = (pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True))
df.insert(0, "transaction_id", [f"TXN-{i:07d}" for i in range(1, len(df) + 1)])
df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")
df.to_csv(args.out, index=False)
print(f"{len(df):,} transactions · {df.customer_id.nunique():,} customers "
      f"· {df.merchant_name.nunique()} merchants "
      f"· {df.date.min()[:10]} → {df.date.max()[:10]} → {args.out}")