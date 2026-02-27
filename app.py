import streamlit as st
import pandas as pd
from fitparse import FitFile
from io import BytesIO
import matplotlib.pyplot as plt
from collections import defaultdict
import zipfile
import os

st.set_page_config(page_title="Workout × Garmin", layout="wide")

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

# =====================================================
# Strong
# =====================================================

def parse_strong_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def extract_strong_sets_for_date(df, date):
    same_day = df[df["Date"].dt.date == date.date()]
    if same_day.empty:
        raise ValueError("No Strong workout on same date")

    sets = []
    for _, r in same_day.iterrows():
        sets.append({
            "exercise": r["Exercise Name"],
            "weight": float(r["Weight"]) if pd.notna(r["Weight"]) else 0.0,
            "reps": float(r["Reps"]) if pd.notna(r["Reps"]) else 0,
            "superset_id": None
        })
    return sets

# =====================================================
# Common set reordering
# =====================================================

def reorder_sets_for_execution(sets):
    """Interleave superset exercises into execution order.
    Straight sets (superset_id is None/NaN) pass through unchanged.
    """
    result = []
    i = 0
    n = len(sets)

    while i < n:
        sid = sets[i]["superset_id"]
        if pd.isna(sid):
            result.append(sets[i])
            i += 1
            continue

        # Collect the full superset block (consecutive sets sharing this sid)
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

def merge_workout(workout_sets, garmin_sets, hr_df):
    workout_sets = reorder_sets_for_execution(workout_sets)
    merged = []

    for i, g in enumerate(garmin_sets):
        if i >= len(workout_sets):
            break

        s, e = g["start"], g["end"]
        slice_df = hr_df[(hr_df["timestamp"] >= s) & (hr_df["timestamp"] <= e)]
        if slice_df.empty:
            continue

        ws = workout_sets[i]
        merged.append({
            "set_index": i,
            "exercise": ws["exercise"],
            "weight": ws["weight"],
            "reps": ws["reps"],
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
    fig, ax = plt.subplots(figsize=(14, 5))
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
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(d["timestamp"], d["heart_rate"])
    ax.set_title(f'{row["exercise"]} | {row["weight"]}×{row["reps"]}')
    st.pyplot(fig)

# =====================================================
# UI
# =====================================================

st.title("🏋️ Workout × Garmin Set-Level HR")

# Sidebar: app selector
selected_app = st.sidebar.selectbox("Workout App", ["Lyfta", "Strong"])

workout_file = st.sidebar.file_uploader(f"{selected_app} CSV", type=["csv"])
fit_files = st.sidebar.file_uploader(
    "Garmin FIT / ZIP files",
    type=["fit", "zip"],
    accept_multiple_files=True
)

# Demo mode only available for Lyfta
if selected_app == "Lyfta":
    use_demo = st.sidebar.checkbox("Use demo data", value=not (workout_file and fit_files))
else:
    use_demo = False

if use_demo:
    demo_dir = "demo"
    workout_file = os.path.join(demo_dir, "demo_lyfta.csv")
    fit_candidates = [
        os.path.join(demo_dir, f)
        for f in os.listdir(demo_dir)
        if f.lower().endswith(".fit")
    ]
    fit_files = sorted(fit_candidates)

if workout_file and fit_files:
    try:
        # Parse the workout CSV
        if selected_app == "Lyfta":
            workout_df = parse_lyfta_csv(workout_file)
        else:
            workout_df = parse_strong_csv(workout_file)

        # Load all FIT files and extract start times
        workouts = []
        for f in fit_files:
            b = load_fit_bytes(f)
            start = extract_activity_start(b)
            workouts.append({
                "label": start.strftime("%Y-%m-%d %H:%M"),
                "start": start,
                "bytes": b
            })

        # Pre-merge all workouts for cross-workout analysis (no superset reorder)
        all_merged_parts = []
        for w in workouts:
            try:
                hr_df_w = parse_hr_df(w["bytes"])
                garmin_sets_w = extract_active_sets(w["bytes"])
                if selected_app == "Lyfta":
                    ws = extract_lyfta_sets_for_date(workout_df, w["start"])
                else:
                    ws = extract_strong_sets_for_date(workout_df, w["start"])
                merged_w = merge_workout(ws, garmin_sets_w, hr_df_w)
                if not merged_w.empty:
                    merged_w["workout_date"] = w["start"].date()
                    merged_w["workout_label"] = w["label"]
                    all_merged_parts.append(merged_w)
            except Exception:
                pass  # Skip workouts with no matching date

        all_merged_df = (
            pd.concat(all_merged_parts, ignore_index=True)
            if all_merged_parts
            else pd.DataFrame()
        )

        tab1, tab2 = st.tabs(["Workout View", "Exercise History"])

        # ----------------------------------------------------------------
        # Tab 1 — Workout View
        # ----------------------------------------------------------------
        with tab1:
            selected = st.selectbox(
                "Select workout",
                workouts,
                format_func=lambda x: x["label"]
            )
            selected_label = selected["label"]

            hr_df = parse_hr_df(selected["bytes"])
            garmin_sets = extract_active_sets(selected["bytes"])

            if selected_app == "Lyfta":
                extracted_sets = extract_lyfta_sets_for_date(workout_df, selected["start"])
            else:
                extracted_sets = extract_strong_sets_for_date(workout_df, selected["start"])

            # Strong: optional superset configuration
            if selected_app == "Strong":
                with st.expander("Configure Supersets (optional)", expanded=False):
                    st.caption(
                        "Assign the same integer to exercises that form a superset. "
                        "Leave blank for straight sets."
                    )
                    unique_exercises = list(dict.fromkeys(s["exercise"] for s in extracted_sets))
                    config_df = pd.DataFrame({
                        "Exercise": unique_exercises,
                        "Superset Group": [None] * len(unique_exercises)
                    })
                    edited = st.data_editor(
                        config_df,
                        column_config={
                            "Superset Group": st.column_config.NumberColumn(
                                min_value=1,
                                step=1,
                                help="Same number = superset. Leave blank for straight sets."
                            )
                        },
                        key=f"superset_{selected_label}",
                        hide_index=True
                    )
                    superset_map = {
                        row["Exercise"]: int(row["Superset Group"])
                        for _, row in edited.iterrows()
                        if pd.notna(row["Superset Group"])
                    }
                    for s in extracted_sets:
                        s["superset_id"] = superset_map.get(s["exercise"])

            merged_df = merge_workout(extracted_sets, garmin_sets, hr_df)

            st.subheader("Sets")
            st.dataframe(merged_df)

            st.download_button(
                label="Download merged data as CSV",
                data=merged_df.to_csv(index=False),
                file_name=f"workout_{selected_label.replace(':', '-').replace(' ', '_')}.csv",
                mime="text/csv"
            )

            st.subheader("Per-Set Heart Rate Analysis")
            for _, row in merged_df.iterrows():
                with st.expander(
                    f'Set {row["set_index"] + 1}: {row["exercise"]}',
                    expanded=False
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Exercise", row["exercise"])
                    c2.metric("Load", f'{row["weight"]} × {row["reps"]}')
                    c3.metric("Avg HR", f'{row["avg_hr"]:.1f}')
                    c4.metric("Max HR", f'{row["max_hr"]:.0f}')
                    plot_set(hr_df, row)

            st.subheader("Full workout HR")
            plot_full(hr_df, merged_df)

            # Cross-exercise comparison for this workout
            st.subheader("Cross-Exercise Comparison")
            wk_summary_df = merged_df.copy()
            wk_summary_df["volume"] = wk_summary_df["weight"] * wk_summary_df["reps"]
            summary = (
                wk_summary_df.groupby("exercise")
                .agg(
                    avg_hr=("avg_hr", "mean"),
                    max_hr=("max_hr", "max"),
                    sets=("set_index", "count"),
                    volume=("volume", "sum")
                )
                .reset_index()
            )
            st.dataframe(summary)

            col1, col2 = st.columns(2)
            chart_height = max(3, len(summary) * 0.5)
            with col1:
                st.subheader("Avg HR by Exercise")
                fig_cx1, ax_cx1 = plt.subplots(figsize=(6, chart_height))
                ax_cx1.barh(summary["exercise"], summary["avg_hr"])
                ax_cx1.set_xlabel("Avg HR (bpm)")
                plt.tight_layout()
                st.pyplot(fig_cx1)
            with col2:
                st.subheader("Max HR by Exercise")
                fig_cx2, ax_cx2 = plt.subplots(figsize=(6, chart_height))
                ax_cx2.barh(summary["exercise"], summary["max_hr"])
                ax_cx2.set_xlabel("Max HR (bpm)")
                plt.tight_layout()
                st.pyplot(fig_cx2)

        # ----------------------------------------------------------------
        # Tab 2 — Exercise History
        # ----------------------------------------------------------------
        with tab2:
            if all_merged_df.empty:
                st.info(
                    "No merged data available. "
                    "Ensure your workout CSV dates match the uploaded Garmin FIT files."
                )
            else:
                all_exercises = sorted(all_merged_df["exercise"].unique())
                selected_exercise = st.selectbox("Select exercise", all_exercises)

                ex_df = all_merged_df[all_merged_df["exercise"] == selected_exercise].copy()
                ex_df = ex_df.sort_values(["workout_date", "set_index"])

                # Epley 1RM estimate per set
                ex_df["est_1rm"] = ex_df["weight"] * (1 + ex_df["reps"] / 30)

                st.dataframe(
                    ex_df[["workout_label", "set_index", "weight", "reps", "avg_hr", "max_hr", "est_1rm"]]
                    .rename(columns={
                        "workout_label": "Workout",
                        "set_index": "Set #",
                        "est_1rm": "Est. 1RM"
                    })
                )

                # Aggregate per day — date as "YYYY-MM-DD" string for clean x-axis
                hr_by_date = (
                    ex_df.groupby("workout_date")["avg_hr"]
                    .mean()
                    .reset_index()
                    .sort_values("workout_date")
                )
                hr_by_date["date_str"] = hr_by_date["workout_date"].astype(str)

                orm_by_date = (
                    ex_df.groupby("workout_date")["est_1rm"]
                    .max()
                    .reset_index()
                    .sort_values("workout_date")
                )
                orm_by_date["date_str"] = orm_by_date["workout_date"].astype(str)

                st.subheader("Avg HR over time")
                fig_hr, ax_hr = plt.subplots(figsize=(10, 3))
                ax_hr.plot(hr_by_date["date_str"], hr_by_date["avg_hr"], marker="o")
                ax_hr.set_ylabel("Avg HR (bpm)")
                ax_hr.set_xlabel("Date")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig_hr)

                st.subheader("Est. 1RM over time (strength progression)")
                fig_orm, ax_orm = plt.subplots(figsize=(10, 3))
                ax_orm.plot(orm_by_date["date_str"], orm_by_date["est_1rm"], marker="o", color="tab:orange")
                ax_orm.set_ylabel("Est. 1RM (kg)")
                ax_orm.set_xlabel("Date")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig_orm)


    except Exception as e:
        st.error(str(e))
else:
    st.info("Upload data or enable demo mode")
