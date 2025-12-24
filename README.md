# 🏋️‍♂️ Lyfta × Garmin Data Merger and Visualization Tool

Merge **Lyfta workout set data** with **Garmin activity heart-rate data** to get **set-level physiological insights**—including per-set HR curves, averages, peaks, and a full workout timeline.

This tool is designed for people who:
- Track **sets/reps/weights** in **Lyfta**
- Track **heart rate & activity data** using **Garmin**
- Want **per-set heart-rate analysis**, not just a single workout blob

---

## Live Demo (Hosted)

You can use the app directly here (no setup required):

**https://lyfta-garmin-data-merger.streamlit.app/**

The demo also includes sample data so you can explore the UI without uploading anything.

---

## Features

- Upload **one Lyfta CSV** and **one or more Garmin `.fit` files**
- Automatically:
  - Match Garmin activities to Lyfta workouts **by date**
  - Detect **active sets** from Garmin (ignores rest sets)
  - Reorder Lyfta sets correctly when **supersets** are used
- View:
  - **Full workout HR timeline** with set regions highlighted and labeled
  - **Per-set HR graphs** (one graph per exercise set)
  - **Per-set metrics**:
    - Exercise name
    - Weight × reps
    - Average HR
    - Max HR
    - Sample count
- Supports **multiple Garmin activities** → select a workout and drill down

---

## Exporting Your Data

### 🔹 Export from Lyfta (CSV)

You can export your workout data from Lyfta as a CSV:

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

### 🔹 Export from Garmin (.fit)

Garmin activities must be exported **per activity** as `.fit` files.

Steps:
1. Go to:  
   https://connect.garmin.com/modern/activities
2. Open the activity you want
3. Click the **⚙️ settings icon** (top-right)
4. Select **Export File**
5. Upload the downloaded `.fit` file

> ℹ️ Bulk Garmin exports (via Garmin’s “Export Your Data”) do **not reliably include all `.fit` activity files**.  
> For now, the app supports:
> - Single `.fit` uploads  
> - Or `.zip` files containing **exactly one `.fit`**

---

## How the Matching Works

1. **Garmin activity date** is extracted from the `.fit` file
2. The app selects the **Lyfta workout on the same calendar date**
3. Lyfta sets are:
   - Kept sequential by default
   - **Reordered intelligently for supersets** using round-robin execution
4. Garmin **active sets** are matched **1-to-1** with Lyfta sets
5. HR data is sliced per set window and analyzed

This avoids false HR assignment and preserves real execution order.

---

## Visualizations

### 1. Full Workout Timeline
- Continuous heart-rate curve
- Each set shown as a shaded region
- Exercise names labeled under the curve

### 2. Per-Set Drill-Down
For each set:
- Exercise name
- Load (weight × reps)
- Avg HR / Max HR
- Individual HR graph for that set only

This makes it easy to answer questions like:
- *Which sets spike my HR the most?*
- *Does HR drop across later sets of the same exercise?*
- *How do supersets affect cardiovascular load?*

---

## Current Limitations

- Garmin `.fit` files must be uploaded **one activity at a time**. The "Export All Data" feature that garmin provides does not give you a consolidated list of all .fit files from your activities from what I've seen. I may write a script to automatically download all .fit files from your garmin connect dashboard in the future.
- Only **heart rate** and numeric Garmin record fields are visualized
- Assumes **one Lyfta workout per day**

---

## Future Ideas

- Support Garmin bulk ZIP exports
- Add HR zone analysis per set
- Compare same exercise across workouts
- Export merged data as CSV / JSON
- Interactive plots (Plotly)
- Strength-specific effort metrics
- Implement set auto-detect based on heartrate peaks so that you don't need to constantly stop and start new sets on the watch while working out
- Integrate with the other major workout platforms such as Hevy and Strong and incorporate their export structures

---

## License

MIT License — feel free to fork, modify, and build on top of this.

---