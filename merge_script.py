#!/usr/bin/env python3
"""
merge_garmin_lyfta.py

Usage:
    python merge_garmin_lyfta.py /path/to/lyfta.csv /mnt/data/21159045593_ACTIVITY.fit

This script:
- Parses Lyfta CSV (format: Title,Date,Duration,Exercise,"Superset id",Weight,Reps,...)
- Parses the Garmin FIT file (extracts timestamps and heart_rate records).
- Tries to map sets from Garmin to Lyfta sets (1:1).
- If Garmin has set/lap markers, use them. Else use HR peak-detection. Else equal-split.
- Outputs merged_workout.csv and prints summary.
"""

import sys
import os
from datetime import datetime, timedelta
import math
import json

try:
    import pandas as pd
    import numpy as np
    from fitparse import FitFile
    from scipy.signal import find_peaks
except Exception as e:
    print("Missing dependency. Run: pip install fitparse pandas numpy scipy")
    raise e

# ---------- Utilities ----------

def parse_lyfta_csv(path):
    """
    Parse Lyfta CSV into ordered list of sets.
    Expects CSV header like the sample user provided.
    Returns: workout_info (dict), sets (list of dicts)
    """
    df = pd.read_csv(path)
    # Normalize columns (strip quotes etc.)
    # Assumes every row is a set for the same workout (common Title and Date)
    if df.empty:
        raise ValueError("Lyfta CSV empty.")
    # Get workout-level fields from first row
    wdate_raw = df.loc[0, 'Date']
    # Try parse date in common formats
    try:
        workout_start = pd.to_datetime(wdate_raw)
    except Exception:
        workout_start = pd.to_datetime(wdate_raw, infer_datetime_format=True)
    workout_info = {
        'title': df.loc[0, 'Title'],
        'start_time': workout_start.to_pydatetime(),
        'duration_text': df.loc[0, 'Duration'] if 'Duration' in df.columns else None,
        'n_sets': len(df)
    }
    sets = []
    for i, row in df.iterrows():
        s = {
            'set_index': int(i),  # zero-based index
            'exercise': row.get('Exercise'),
            'weight': row.get('Weight'),
            'reps': row.get('Reps'),
            'set_type': row.get('Set Type') if 'Set Type' in row.index else None,
            # additional fields can be included as needed
        }
        sets.append(s)
    return workout_info, sets

def parse_fit_file(fit_path):
    """
    Parse FIT file and extract:
    - activity_start (datetime)
    - activity_total_elapsed (seconds) if available
    - hr_df: DataFrame with timestamp and heart_rate
    - garmin_sets: list of dicts {start, end} if available from 'lap' or 'event' messages
    """
    fitfile = FitFile(fit_path)

    # Default containers
    records = []
    activity_start = None
    total_elapsed = None
    garmin_sets = []

    # First pass: get session / file_id time
    for msg in fitfile.get_messages():
        name = msg.name
        if name == 'file_id':
            # file_id often contains 'time_created'
            for d in msg:
                if d.name == 'time_created' and d.value:
                    activity_start = d.value if activity_start is None else activity_start
        elif name == 'session':
            # session often stores start_time and total_elapsed_time
            for d in msg:
                if d.name == 'start_time' and d.value:
                    activity_start = d.value
                if d.name in ('total_elapsed_time','total_timer_time') and d.value:
                    total_elapsed = float(d.value)
        elif name == 'lap':
            # lap messages sometimes mark set boundaries (not guaranteed)
            lap_start = None
            lap_end = None
            for d in msg:
                if d.name == 'start_time':
                    lap_start = d.value
                if d.name == 'timestamp':
                    lap_end = d.value
                if d.name in ('start_time', 'timestamp') and d.value:
                    pass
            if lap_start and lap_end:
                garmin_sets.append({'start': lap_start, 'end': lap_end})
        # don't break; continue to collect records below

    # Second pass: collect record messages (timestamp + heart_rate)
    fitfile = FitFile(fit_path)  # reopen to iterate again cleanly
    for record in fitfile.get_messages('record'):
        rec = {}
        for d in record:
            rec[d.name] = d.value
        # We only care about timestamp and heart_rate
        if 'timestamp' in rec and ('heart_rate' in rec or 'heart_rate' in rec.keys()):
            # Some fitparse returns heart_rate as 'heart_rate'
            hr = rec.get('heart_rate', None)
            if hr is None:
                # Try variants
                hr = rec.get('heart_rate', None)
            records.append({'timestamp': rec['timestamp'], 'heart_rate': hr})

    if not records:
        raise ValueError("No HR records found in FIT file.")

    hr_df = pd.DataFrame(records)
    hr_df = hr_df.dropna(subset=['timestamp']).drop_duplicates(subset=['timestamp'])
    hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp'])
    hr_df = hr_df.sort_values('timestamp').reset_index(drop=True)

    # If hr column has NaNs, forward-fill or drop
    if hr_df['heart_rate'].isna().all():
        raise ValueError("FIT HR records exist but heart_rate values are all missing.")
    hr_df['heart_rate'] = hr_df['heart_rate'].ffill().bfill().astype(float)

    # Estimate activity_start if still None
    if activity_start is None:
        activity_start = hr_df['timestamp'].iloc[0]

    # If total_elapsed not found, compute from HR timestamps
    if total_elapsed is None:
        total_elapsed = (hr_df['timestamp'].iloc[-1] - hr_df['timestamp'].iloc[0]).total_seconds()

    return {
        'activity_start': pd.to_datetime(activity_start).to_pydatetime(),
        'total_elapsed': float(total_elapsed),
        'hr_df': hr_df,
        'garmin_sets': garmin_sets
    }

# ---------- Set segmentation helpers ----------

def detect_sets_from_hr(hr_df, expected_n_sets, min_peak_distance_seconds=30, min_peak_prominence=6):
    """
    Try to detect sets by finding HR peaks.
    Returns list of windows [{'start': dt, 'end': dt, 'peak': dt}, ...] length <= expected_n_sets
    Algorithm:
    - Resample HR to 1s
    - Smooth with rolling median
    - Find peaks with prominence
    - Build windows around peaks until HR drops near baseline
    """
    df = hr_df.copy().set_index('timestamp')
    # resample to 1s
    df = df.resample('1S').mean().interpolate()
    hr = df['heart_rate'].values
    times = df.index.to_pydatetime()

    if len(hr) < 5:
        return []

    baseline = np.median(hr[:max(1, min(30, len(hr)) )])  # first 30s median as baseline
    # Determine threshold: baseline + prominence
    prominence = min_peak_prominence
    # min distance in samples
    distance = int(min_peak_distance_seconds)
    peaks, props = find_peaks(hr, distance=distance, prominence=prominence)
    # If too many peaks, keep strongest expected_n_sets
    if len(peaks) == 0:
        return []

    # Sort peaks by prominence descending then pick top expected_n_sets and sort by time
    prominences = props.get('prominences', np.ones(len(peaks)))
    peak_info = sorted(zip(peaks, prominences), key=lambda x: -x[1])
    selected_peaks = sorted([p for p, _ in peak_info[:expected_n_sets]])
    # Build windows by expanding left/right until hr falls near baseline + 1/3 prominence or until midpoint with neighbors
    windows = []
    for idx, p in enumerate(selected_peaks):
        peak_time = times[p]
        # Determine threshold to stop expansion
        thr = baseline + 0.33 * prominences[list(peaks).index(p)] if len(prominences)>0 else baseline + 3
        # expand left
        left = p
        while left > 0 and hr[left] > thr:
            left -= 1
        # expand right
        right = p
        while right < len(hr)-1 and hr[right] > thr:
            right += 1
        start = times[left]
        end = times[right]
        windows.append({'start': start, 'end': end, 'peak': peak_time})
    # If we detected fewer windows than expected, don't fail — return what we have
    return windows

def equal_split_windows(activity_start, activity_end, n):
    """
    Split time between activity_start and activity_end into n equal windows.
    """
    total = (activity_end - activity_start).total_seconds()
    if n <= 0 or total <= 0:
        return []
    windows = []
    for i in range(n):
        s = activity_start + timedelta(seconds=(i * total / n))
        e = activity_start + timedelta(seconds=((i+1) * total / n))
        windows.append({'start': s, 'end': e, 'peak': s + (e - s)/2})
    return windows

# ---------- Merge logic ----------

def build_set_windows(fit_parsed, lyfta_sets):
    """
    Return list of windows (length == len(lyfta_sets)) mapping to each Lyfta set.
    Strategy:
     1. If garmin_sets present and count matches or close -> map sequentially.
     2. Else try detect_sets_from_hr (peak detection).
     3. Else equal split.
    """
    n = len(lyfta_sets)
    garmin_sets = fit_parsed.get('garmin_sets', [])
    hr_df = fit_parsed['hr_df']
    activity_start = fit_parsed['activity_start']
    activity_end = activity_start + timedelta(seconds=fit_parsed['total_elapsed'])

    # 1. Garmin-detected laps/sets
    if garmin_sets and len(garmin_sets) >= n:
        # If more than needed, pick those within activity range and nearest start times
        windows = []
        # Sort garmin sets by start
        garmin_sorted = sorted(garmin_sets, key=lambda x: x['start'])
        # Map first n
        for i in range(n):
            g = garmin_sorted[i]
            windows.append({'start': pd.to_datetime(g['start']).to_pydatetime(),
                            'end': pd.to_datetime(g['end']).to_pydatetime(),
                            'peak': pd.to_datetime(g['start']).to_pydatetime()})
        return windows

    # 2. Try peak detection
    detected = detect_sets_from_hr(hr_df, expected_n_sets=n)
    if detected and len(detected) >= 1:
        # If we detected fewer than n, try to supplement by nearest equal-split windows
        if len(detected) == n:
            return detected
        else:
            # If fewer, try to augment: compute remaining by equal split on remaining intervals
            # A simple approach: if detected < n, fallback to equal split (more robust)
            pass

    # 3. Fallback: equal split
    return equal_split_windows(activity_start, activity_end, n)

def compute_metrics_for_window(hr_df, window):
    """
    Compute avg_hr, max_hr, duration_seconds for given window (start,end).
    hr_df has timestamp and heart_rate.
    """
    s = pd.to_datetime(window['start'])
    e = pd.to_datetime(window['end'])
    mask = (hr_df['timestamp'] >= s) & (hr_df['timestamp'] <= e)
    slice_df = hr_df.loc[mask]
    if slice_df.empty:
        # If no samples in window, try nearest sample by time
        nearest_idx = (hr_df['timestamp'] - s).abs().idxmin()
        sample = hr_df.loc[nearest_idx]
        avg_hr = float(sample['heart_rate'])
        max_hr = float(sample['heart_rate'])
        duration = (e - s).total_seconds()
        hr_curve = [{'ts': str(sample['timestamp']), 'hr': sample['heart_rate']}]
        return {'avg_hr': avg_hr, 'max_hr': max_hr, 'duration': duration, 'hr_curve': hr_curve}
    avg_hr = float(slice_df['heart_rate'].mean())
    max_hr = float(slice_df['heart_rate'].max())
    duration = (e - s).total_seconds()
    hr_curve = [{'ts': str(r['timestamp']), 'hr': float(r['heart_rate'])} for _, r in slice_df.iterrows()]
    return {'avg_hr': avg_hr, 'max_hr': max_hr, 'duration': duration, 'hr_curve': hr_curve}

# ---------- Main runner ----------

def main(lyfta_csv_path, fit_path, out_csv='merged_workout.csv'):
    if not os.path.exists(lyfta_csv_path):
        raise FileNotFoundError(f"Lyfta CSV not found: {lyfta_csv_path}")
    if not os.path.exists(fit_path):
        raise FileNotFoundError(f"FIT not found: {fit_path}")

    print("Parsing Lyfta CSV...")
    workout_info, lyfta_sets = parse_lyfta_csv(lyfta_csv_path)
    print(f"Lyfta workout start: {workout_info['start_time']}, sets: {len(lyfta_sets)}")

    print("Parsing FIT file...")
    fit_parsed = parse_fit_file(fit_path)
    print(f"Garmin activity start: {fit_parsed['activity_start']}, elapsed_seconds: {fit_parsed['total_elapsed']:.0f}")

    # Build windows
    print("Constructing set windows...")
    windows = build_set_windows(fit_parsed, lyfta_sets)
    if not windows or len(windows) != len(lyfta_sets):
        # If windows length mismatched, try best effort: equal split forced
        print("Warning: Detected windows don't match expected number of sets. Using equal-split fallback.")
        activity_end = fit_parsed['activity_start'] + timedelta(seconds=fit_parsed['total_elapsed'])
        windows = equal_split_windows(fit_parsed['activity_start'], activity_end, len(lyfta_sets))

    # Compute per-set metrics and assemble merged rows
    merged_rows = []
    hr_df = fit_parsed['hr_df']
    for idx, set_rec in enumerate(lyfta_sets):
        w = windows[idx]
        metrics = compute_metrics_for_window(hr_df, w)
        merged = {
            'set_index': idx,
            'exercise': set_rec.get('exercise'),
            'weight': set_rec.get('weight'),
            'reps': set_rec.get('reps'),
            'set_start': w['start'],
            'set_end': w['end'],
            'set_duration_s': metrics['duration'],
            'avg_hr': metrics['avg_hr'],
            'max_hr': metrics['max_hr'],
            'hr_curve_json': json.dumps(metrics['hr_curve'])
        }
        merged_rows.append(merged)

    out_df = pd.DataFrame(merged_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved merged CSV -> {out_csv}")

    # Print overall workout info
    all_avg = out_df['avg_hr'].mean()
    all_max = out_df['max_hr'].max()
    total_sets = len(out_df)
    workout_duration_seconds = fit_parsed['total_elapsed']
    print("\n==== Workout Summary ====")
    print(f"Title: {workout_info.get('title')}")
    print(f"Start (Lyfta): {workout_info.get('start_time')}")
    print(f"Start (Garmin): {fit_parsed.get('activity_start')}")
    print(f"Workout duration (sec): {workout_duration_seconds:.0f}")
    print(f"Total sets: {total_sets}")
    print(f"Avg HR across sets: {all_avg:.1f} bpm")
    print(f"Max HR across sets: {all_max:.0f} bpm")

    # Print per-set table
    print("\nPer-set table (index, exercise, reps, weight, duration_s, avg_hr, max_hr):")
    for _, r in out_df.iterrows():
        print(f"{int(r.set_index):2d} | {r.exercise:25.25} | reps={r.reps} | wt={r.weight} | dur={r.set_duration_s:.0f}s | avg={r.avg_hr:.1f} | max={r.max_hr:.0f}")

    return out_df

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python merge_garmin_lyfta.py /path/to/lyfta.csv /path/to/activity.fit")
        print("Example: python merge_garmin_lyfta.py lyfta.csv /mnt/data/21159045593_ACTIVITY.fit")
        sys.exit(1)
    lyfta_csv = sys.argv[1]
    fit_file = sys.argv[2]
    main(lyfta_csv, fit_file)
