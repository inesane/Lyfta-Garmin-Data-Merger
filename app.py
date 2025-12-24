import streamlit as st
import pandas as pd
from fitparse import FitFile
from io import BytesIO
import matplotlib.pyplot as plt
from collections import defaultdict
import zipfile
import os
import glob

st.set_page_config(page_title="Lyfta × Garmin", layout="wide")

# =====================================================
# Utilities
# =====================================================

def load_fit_bytes(uploaded_file):
    if isinstance(uploaded_file, str):
        with open(uploaded_file, "rb") as f:
            return f.read()

    data = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".fit"):
        return data

    if name.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(data)) as z:
            fits = [f for f in z.namelist() if f.lower().endswith(".fit")]
            if len(fits) != 1:
                raise ValueError("ZIP must contain exactly one .fit file")
            return z.read(fits[0])

    raise ValueError("Unsupported file type")

# =====================================================
# Lyfta
# =====================================================

def parse_lyfta_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    if "Superset id" in df.columns:
        df["Superset id"] = df["Superset id"].astype("Int64")
    return df

def extract_lyfta_sets_for_date(df, date):
    same_day = df[df["Date"].dt.date == date.date()]
    if same_day.empty:
        raise ValueError("No Lyfta workout on same date")

    sets = []
    for _, r in same_day.iterrows():
        sets.append({
            "exercise": r["Exercise"],
            "weight": r["Weight"],
            "reps": r["Reps"],
            "superset_id": r.get("Superset id")
        })
    return sets

def reorder_lyfta_sets_for_execution(sets):
    result = []
    i = 0
    n = len(sets)

    while i < n:
        sid = sets[i]["superset_id"]
        if pd.isna(sid):
            result.append(sets[i])
            i += 1
            continue

        block = []
        j = i
        while j < n and not pd.isna(sets[j]["superset_id"]) and sets[j]["superset_id"] == sid:
            block.append(sets[j])
            j += 1

        by_ex = defaultdict(list)
        for s in block:
            by_ex[s["exercise"]].append(s)

        max_len = max(len(v) for v in by_ex.values())
        for k in range(max_len):
            for ex in by_ex:
                if k < len(by_ex[ex]):
                    result.append(by_ex[ex][k])

        i = j

    return result

# =====================================================
# Garmin
# =====================================================

def extract_activity_start(fit_bytes):
    fit = FitFile(BytesIO(fit_bytes))
    for msg in fit.get_messages("session"):
        for f in msg:
            if f.name == "start_time":
                return pd.to_datetime(f.value)
    raise ValueError("No start_time")

def parse_hr_df(fit_bytes):
    fit = FitFile(BytesIO(fit_bytes))
    rows = []
    for r in fit.get_messages("record"):
        row = {}
        for f in r:
            if f.name == "timestamp":
                row["timestamp"] = pd.to_datetime(f.value)
            elif f.name == "heart_rate":
                row["heart_rate"] = f.value
        if "timestamp" in row:
            rows.append(row)
    return pd.DataFrame(rows)

def extract_active_sets(fit_bytes):
    fit = FitFile(BytesIO(fit_bytes))
    sets = []
    for msg in fit.get_messages("set"):
        f = {x.name: x.value for x in msg}
        if str(f.get("set_type")).lower() == "active":
            start = pd.to_datetime(f["start_time"])
            dur = f.get("duration", 20)
            end = start + pd.to_timedelta(float(dur), unit="s")
            sets.append({"start": start, "end": end})
    return sorted(sets, key=lambda x: x["start"])

# =====================================================
# Merge
# =====================================================

def merge_garmin_lyfta(lyfta_sets, garmin_sets, hr_df):
    lyfta_sets = reorder_lyfta_sets_for_execution(lyfta_sets)
    merged = []

    for i, g in enumerate(garmin_sets):
        if i >= len(lyfta_sets):
            break

        s, e = g["start"], g["end"]
        slice_df = hr_df[(hr_df["timestamp"] >= s) & (hr_df["timestamp"] <= e)]
        if slice_df.empty:
            continue

        ly = lyfta_sets[i]
        merged.append({
            "set_index": i,
            "exercise": ly["exercise"],
            "weight": ly["weight"],
            "reps": ly["reps"],
            "start": s,
            "end": e,
            "avg_hr": slice_df["heart_rate"].mean(),
            "max_hr": slice_df["heart_rate"].max(),
            "samples": len(slice_df)
        })

    return pd.DataFrame(merged)

# =====================================================
# Plots
# =====================================================

def plot_full(hr_df, merged_df):
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(hr_df["timestamp"], hr_df["heart_rate"])
    ymin = hr_df["heart_rate"].min() - 5

    for _, r in merged_df.iterrows():
        ax.axvspan(r["start"], r["end"], alpha=0.25)
        mid = r["start"] + (r["end"] - r["start"]) / 2
        ax.text(mid, ymin, r["set_index"] + 1, ha="center", va="top", fontsize=8)

    st.pyplot(fig)

def plot_set(hr_df, row):
    s, e = row["start"], row["end"]
    d = hr_df[(hr_df["timestamp"] >= s) & (hr_df["timestamp"] <= e)]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(d["timestamp"], d["heart_rate"])
    ax.set_title(f'{row["exercise"]} | {row["weight"]}×{row["reps"]}')
    st.pyplot(fig)

# =====================================================
# UI
# =====================================================

st.title("🏋️ Lyfta × Garmin Set-Level HR")

lyfta_file = st.sidebar.file_uploader("Lyfta CSV", type=["csv"])
fit_files = st.sidebar.file_uploader(
    "Garmin FIT / ZIP files",
    type=["fit","zip"],
    accept_multiple_files=True
)

use_demo = st.sidebar.checkbox("Use demo data", value=not (lyfta_file and fit_files))

if use_demo:
    demo_dir = "demo"

    # pick the first CSV found in the demo folder (if any)
    csv_candidates = [os.path.join(demo_dir, f) for f in os.listdir(demo_dir) if f.lower().endswith(".csv")]
    lyfta_file = csv_candidates[0] if csv_candidates else None

    # collect all .fit files in the demo folder (case-insensitive)
    fit_candidates = [os.path.join(demo_dir, f) for f in os.listdir(demo_dir) if f.lower().endswith(".fit")]
    fit_files = sorted(fit_candidates)

if lyfta_file and fit_files:
    try:
        lyfta_df = parse_lyfta_csv(lyfta_file)

        workouts = []
        for f in fit_files:
            b = load_fit_bytes(f)
            start = extract_activity_start(b)
            workouts.append({
                "label": start.strftime("%Y-%m-%d %H:%M"),
                "start": start,
                "bytes": b
            })

        selected = st.selectbox(
            "Select workout",
            workouts,
            format_func=lambda x: x["label"]
        )

        hr_df = parse_hr_df(selected["bytes"])
        garmin_sets = extract_active_sets(selected["bytes"])
        lyfta_sets = extract_lyfta_sets_for_date(lyfta_df, selected["start"])
        merged_df = merge_garmin_lyfta(lyfta_sets, garmin_sets, hr_df)

        st.subheader("Sets")
        st.dataframe(merged_df)

        st.subheader("Per-Set Heart Rate Analysis")

        for _, row in merged_df.iterrows():
            with st.expander(
                f'Set {row["set_index"] + 1}: {row["exercise"]}',
                expanded=False
            ):
                # --- Summary metrics ---
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Exercise", row["exercise"])
                c2.metric("Load", f'{row["weight"]} × {row["reps"]}')
                c3.metric("Avg HR", f'{row["avg_hr"]:.1f}')
                c4.metric("Max HR", f'{row["max_hr"]:.0f}')

                # --- Plot ---
                plot_set(hr_df, row)

        st.subheader("Full workout HR")
        plot_full(hr_df, merged_df)

    except Exception as e:
        st.error(str(e))
else:
    st.info("Upload data or enable demo mode")
