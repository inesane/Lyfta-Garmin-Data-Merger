# 🏋️ Workout × Garmin Set-Level HR

Merge **Lyfta or Strong workout set data** with **Garmin activity heart-rate data** to get **set-level physiological insights**—including per-set HR curves, averages, peaks, a full workout timeline, cross-exercise comparison, and exercise history across multiple workouts.

This tool is designed for people who track **sets/reps/weights** in **Lyfta or Strong**, track **heart rate & activity data** using **Garmin**, and want **per-set heart-rate analysis**.

Set-wise heart rate data can be used to track peak heart rate for each set and analyze how heart rate response changes across successive sets. Faster or higher HR spikes for the same exercise and load can indicate accumulating systemic fatigue or insufficient recovery between sets. By examining heart rate recovery during rest intervals, this data can help determine whether rest periods are adequate and identify the optimal rest duration needed to sustain performance.

Additionally, trends in set-wise heart rate over multiple workouts can be used to assess conditioning and recovery status. Lower heart rate responses for the same workload may indicate improved cardiovascular efficiency, and elevated responses may signal under-recovery or excessive training load.

---

## Live Demo (Hosted)

You can use the app directly here hosted using Streamlit Community Cloud (no setup required):

**https://workout-garmin-data-merger.streamlit.app/**

The demo also includes sample Lyfta data so you can explore the UI without uploading anything.

---

## Features

- Upload a **Lyfta CSV** or **Strong CSV** and **one or more Garmin `.fit` files**
- Automatically:
  - Match Garmin activities to workouts **by date**
  - Detect **active sets** from Garmin (ignores rest sets)
  - Reorder sets correctly when **supersets** are used (Lyfta: via exported superset IDs; Strong: via in-app configuration)
- **Workout View tab** — per-workout drill-down:
  - **Sets table** with exercise, weight, reps, avg HR, max HR
  - **Download merged data as CSV**
  - **Per-set HR graphs** (one collapsible expander per set)
  - **Full workout HR timeline** with set regions highlighted and labeled
  - **Cross-exercise comparison** for the selected workout:
    - Summary table: exercise → avg HR, max HR, sets, total volume
    - Side-by-side horizontal bar charts: avg HR and max HR by exercise
- **Exercise History tab** — track a single exercise across all uploaded workouts:
  - Per-set table with date, weight, reps, avg HR, max HR, estimated 1RM
  - Avg HR over time (line chart)
  - Estimated 1RM over time using the **Epley formula** (`weight × (1 + reps/30)`), taking the best set per day as the strength progression metric

---

## Exporting Your Data

### 🔹 Export from Lyfta (CSV)

- In the app:
  **Profile → Settings → Export Data**
- Or directly visit:
  https://my.lyfta.app/settings/export-data

Export the data as **CSV**.

> The app expects the standard Lyfta CSV format, including:
> - `Date`
> - `Exercise`
> - `Weight`
> - `Reps`
> - `Superset id` (optional)

---

### 🔹 Export from Strong (CSV)

- In the app:
  **Settings → Export Strong Data**

Export as **CSV**.

> The app expects the standard Strong CSV format, including:
> - `Date`
> - `Exercise Name`
> - `Weight`
> - `Reps`
> - `Set Order`

> **Note:** Strong does not export superset information. The app provides an optional in-app superset configurator — assign the same integer group to exercises that were performed as a superset, and the app will reorder sets accordingly before matching with Garmin.

---

### 🔹 Export from Garmin (.fit)

Garmin activities must be exported **per activity** as `.fit` files.

Steps:
1. Go to:
   https://connect.garmin.com/modern/activities
2. Open the activity you want
3. Click the **⚙️ settings icon** (top-right)
4. Select **Export File**
5. Upload the downloaded `.fit` file

> ℹ️ Bulk Garmin exports (via Garmin's "Export Your Data") do **not reliably include all `.fit` activity files**.
> For now, the app supports:
> - Single `.fit` uploads
> - Or `.zip` files containing **exactly one `.fit`**

---

## How the Matching Works

1. **Garmin activity date** is extracted from the `.fit` file
2. The app selects the **workout on the same calendar date** from the uploaded CSV
3. Workout sets are:
   - Kept sequential by default
   - **Reordered for supersets** using round-robin interleaving
4. Garmin **active sets** are matched **1-to-1** with workout sets in order
5. HR data is sliced per set window and analyzed

**Note:** The number of sets recorded in your workout app and in Garmin for a given workout should match.

---

## Tabs

### Workout View
Select a Garmin activity from the dropdown. The page shows:
- Full sets table with HR metrics
- CSV export button
- Per-set expandable HR graphs
- Full workout HR timeline
- Cross-exercise summary table and bar charts for the selected workout

### Exercise History
Select any exercise that appears across your uploaded workouts. The page shows:
- A table of every set logged for that exercise with HR and estimated 1RM
- Avg HR over time (line chart, one point per workout day)
- Estimated 1RM over time (line chart, best Epley 1RM per day)

---

## Current Limitations

- Garmin `.fit` files must be uploaded one activity at a time. The "Export All Data" feature that Garmin provides does not give a consolidated list of all `.fit` files from your activities. A bulk download script may be added in the future.
- Only **heart rate** data from Garmin is used
- Assumes **one workout per day** per CSV source

---

## Future Ideas

- Support Garmin bulk ZIP exports
- Add HR zone analysis per set (time spent in each zone per set)
- Interactive plots (Plotly)
- Implement set auto-detect based on HR peaks so you don't need to manually start/stop sets on the watch
- Hevy and other workout app integration

---

## License

MIT License — feel free to fork, modify, and build on top of this.

---
