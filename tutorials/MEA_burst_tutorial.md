# MEA Burst Detection GUI Tutorial (`MEA_burst_detect_v9C8.py`)

This tutorial explains how to use the Jupyter-based GUI in `MEA_burst_detect_v9C8.py` to:

- Load multichannel MEA **epochs** from a `.fif` file (MNE `Epochs`)
- Run **cycle-by-cycle burst detection** (ByCycle-style logic) on a selected epoch
- Visualize bursts and cycle metrics for Eye1 and Eye2 channels
- Identify **global bursts** shared across multiple channels
- Export single-epoch or all-epoch results to Excel

---

## 1. Requirements

Install the required packages (ideally in a virtual environment):

```bash
pip install mne numpy matplotlib pandas ipywidgets
pip install jupyterlab
```

The script also uses:

- `collections.defaultdict`
- `IPython.display` (`display`)
- `matplotlib.gridspec`, `matplotlib.patches`
- The standard library (`os`, etc. if needed)

Make sure:

- You can run **Jupyter Notebook / JupyterLab**.
- `ipywidgets` is enabled so sliders and buttons render properly.

---

## 2. What the Script Expects and Produces

### Input: MNE Epochs `.fif` file

The GUI expects a `.fif` file containing an **MNE `Epochs` object**:

- Loaded via:

  ```python
  epochs = mne.read_epochs(fif_path, preload=True)
  ```

- Shape: `(n_epochs, n_channels, n_times)`
- Sampling frequency taken from `epochs.info['sfreq']`

The code assumes the first 16 channels correspond to:

- **Eye1**: channels 1–8
- **Eye2**: channels 9–16

After loading, the script creates two **synthetic average channels**:

- Index `0`: **Synthetic Eye1 Average (Ch 1–8)**
- Index `17`: **Synthetic Eye2 Average (Ch 9–16)**

So internally you have 18 channels:

```text
0   Synthetic Eye1 Average (Ch 1–8)
1–8 Original Eye1 channels
9–16 Original Eye2 channels
17  Synthetic Eye2 Average (Ch 9–16)
```

### Output: Burst and Global-Burst Tables

The core detection function, `plot_burst_cycles(...)`, returns two `pandas.DataFrame` objects:

1. **`df_bursts`** – per-channel bursts in the current epoch:

   Each row describes one **burst** in one channel, with columns such as:

   - `ChannelIndex`, `ChannelLabel`
   - `BurstID`
   - `Start_s`, `End_s`, `Duration_s`
   - `NumCycles`
   - `MeanFreq`, `StdFreq`
   - `MeanAmp`, `StdAmp`
   - `MeanAmpCons`, `StdAmpCons`
   - `MeanPerCons`, `StdPerCons`
   - `MeanMono`, `StdMono`
   - `IsGlobal`, `GlobalID`, `NumChannelsInGlobal`
   - `SynthEye1InGlobal`, `SynthEye2InGlobal`
   - `GlobalCoveragePct`
   - `Eye1DistantOverlap`, `Eye1NearOverlap`,
     `Eye2DistantOverlap`, `Eye2NearOverlap`

2. **`df_global`** – aggregated **global bursts**:

   Each row describes one global burst that spans multiple channels:

   - `GlobalID`
   - `Start_s`, `End_s`, `Duration_s`
   - `NumChannels`
   - `Channels` (comma-separated channel indices)
   - `SynthEye1Overlap`, `SynthEye2Overlap`
   - `MeanFreq`, `StdFreq`
   - `MeanAmp`, `StdAmp`
   - `MeanAmpCons`, `StdAmpCons`
   - `MeanPerCons`, `StdPerCons`
   - `MeanMono`, `StdMono`
   - `TotalCycles`

These tables are displayed in the notebook for single-epoch plots and can be exported to Excel (single epoch or all epochs) via the GUI.

---

## 3. How to Run the GUI

1. Start Jupyter in the folder containing `MEA_burst_detect_v9C8.py`:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. In a new notebook, run:

   ```python
   %run MEA_burst_detect_v9C8.py
   ```

3. The GUI will appear with:

   - A **FIF Path** text box
   - A **Load Data** button
   - Plot controls (epoch, channel group, detection thresholds)
   - Export controls for Excel

If nothing appears, check the cell output for errors and verify that the `.fif` path and dependencies are correct.

---

## 4. Step-by-Step: Detecting and Visualizing Bursts

### 4.1 Load MEA Epochs

At the top of the GUI:

1. Enter the full path to your `.fif` file in **“FIF Path:”**  
   (e.g. `D:\data\my_recording_epochs.fif`)

2. Click **“Load Data”**.

The script will:

- Read the epochs with MNE
- Add synthetic Eye1/Eye2 average channels
- Create channel names
- Update the **Epoch** slider range

A short message prints:

- Number of epochs
- Number of channels
- Samples per epoch
- Sampling frequency

### 4.2 Choose Epoch and Channel Group

Under the plotting controls:

- **`Epoch` slider**  
  Select which epoch to analyze.

- **`Channels` dropdown** (channel group):

  - **Eye 1 (Synthetic + Ch 1–8)** → channels `[0, 1–8]`
  - **Eye 2 (Ch 9–16 + Synthetic)** → channels `[9–17]`

- **`Display` dropdown**:

  - **`paged`** – show up to 4 channels at a time and use the **`Page`** slider.
  - **`all`** – show all selected channels in one figure.

- **`Topo Mode` dropdown**:

  - **Combined** / **Separated** – controls how channels are mapped onto the MEA topographic mini-map (right-hand panel).

### 4.3 Time Window and Amplitude Limits

- **`Tmin (s)` / `Tmax (s)`**  
  Time window (in seconds) within the epoch to analyze and plot (e.g. `0.0–5.0 s`).

- **`Amp Min` / `Amp Max`**  
  Y-axis limits for the time series plots. Set based on typical microvolt range so traces fit nicely.

### 4.4 Cycle Detection Parameters

These sliders control the ByCycle-style cycle selection:

- **`Amp Frac`** (`amp_fraction`)  
  Minimum absolute amplitude for a cycle to be considered.

- **`Amp Cons`** (`amp_consistency`)  
  Minimum amplitude consistency (0–1) across neighboring half-cycles.

- **`Period Cons`** (`period_consistency`)  
  Minimum period consistency (0–1) across neighboring cycles.

- **`Monotonicity`** (`monotonicity_byc`)  
  Minimum fraction of strictly rising (rise) / strictly falling (decay) steps in the cycle.

- **`Min Cycles`** (`min_n_cycles`)  
  Minimum **number of consecutive valid cycles** required to form a burst.

- **`Freq Min` / `Freq Max`**  
  Frequency band (Hz) in which cycles are considered valid bursts (e.g. `10–18 Hz`).

- **`Min Phase`** (`min_phase_duration`)  
  Currently not used in the detection logic (reserved for future extensions).

- **`Shared Chs`** (`min_shared_channels`)  
  Minimum number of channels that must be simultaneously bursting to count a **global burst**.

### 4.5 Cycle Parameter for Color Coding

- **`Cycle Param` dropdown** controls which metric is used for color-coding individual cycles:

  - `Frequency (Hz)` → `frequency`
  - `Amplitude` → `amp`
  - `Amplitude Consistency` → `amp_cons`
  - `Period Consistency` → `period_cons`
  - `Monotonicity` → `monotonicity_byc`

The selected metric is:

- Mapped to a color scale (viridis colormap) over each cycle region.
- Printed as a small numeric label within each cycle span.

### 4.6 Plotting Bursts

- Optionally toggle:

  - **Show Amplitude**
  - **Show Frequency**

  (These are currently placeholders in the UI; burst detection and cycle coloring run the same regardless.)

- Click **“Plot Bursts”**.

For each selected channel, the GUI:

1. Runs `plot_burst_cycles(...)` on the chosen epoch/time window.
2. Plots the raw signal in black.
3. Overlays:

   - Colored spans for each valid cycle (based on `Cycle Param`),
   - Markers for trough (red), peak (blue), half-rise (green), half-decay (magenta),
   - Blue segments highlighting bursts (≥ `Min Cycles` cycles),
   - Burst labels with median ± SD frequency.

4. Displays two tables in the output area:

   - **All bursts (per channel)** — `df_bursts`
   - **Global bursts (aggregated)** — `df_global`

5. If topography is enabled, it also shows a simplified **MEA topographic map** on the right.

Use the **`Page`** slider (if in paged mode) to browse through blocks of up to 4 channels.

---

## 5. Exporting Results to Excel

### 5.1 Export Single-Epoch Results

In the **single-epoch export** section:

- **“Excel Path:”** – set a full file path, e.g.:

  ```text
  C:/temp/burst_export_epoch0.xlsx
  ```

- Click **“Export to Excel”**.

This writes a **4-sheet Excel file**:

1. **Eye1** – all bursts for Eye1-related channels (indices 0–8) in the current epoch.
2. **Eye2** – all bursts for Eye2-related channels (indices 9–17) in the current epoch.
3. **Global** – all global bursts (`df_global`) in the current epoch.
4. **Details** – a single-row summary with:

   - Detection parameters:
     - `AmpFraction`, `AmpConsistency`, `PeriodConsistency`,
       `Monotonicity`, `MinCycles`, `FreqMin`, `FreqMax`,
       `SharedChannels`
   - `TotalEpochs` in the loaded file (for context)
   - `TotalEye1Bursts`, `TotalEye2Bursts`
   - `GlobalBurstsEye1`, `GlobalBurstsEye2`, `GlobalBurstsTOTAL`

If no data is available (e.g., you haven’t plotted yet), the GUI will print a message instead of exporting.

### 5.2 Process All Epochs and Export

In the **all-epoch export** section:

- Click **“Process All Epochs & Export”**.

The GUI will:

1. Loop over **all epochs**.
2. For each epoch:
   - Run burst detection for Eye1 channels (`[0–8]`) with `skip_plot=True`.
   - Run burst detection for Eye2 channels (`[9–17]`) with `skip_plot=True`.
3. Collect:

   - All Eye1 bursts across epochs → `Eye1` sheet
   - All Eye2 bursts across epochs → `Eye2` sheet
   - All global bursts (Eye1 + Eye2) → `Global` sheet

4. Build a **Details** sheet (similar to single-epoch export) with:

   - Detection parameters
   - `TotalEpochs`
   - Total number of bursts for each eye
   - Separate counts of Eye1 and Eye2 global bursts
   - Total global bursts across both eyes

Progress is printed in the output area as percentages; when finished, the file is written to the path in **“Excel Export Path:”**.

---

## 6. Notes and Limitations

- The GUI is designed for **retina MEA** layouts with Eye1 (Ch1–8) and Eye2 (Ch9–16) plus synthetic averages.
- Time-series units and ranges (for `Amp Min/Max`) depend on how your `.fif` was created (e.g. µV).
- `Show Amplitude` and `Show Frequency` checkboxes are currently **non-critical**; they are passed through but not yet used to toggle specific plot elements.
- Global bursts are defined purely by **temporal overlap** and the `Shared Chs` threshold across channels; no spatial weighting is applied.
- Excel exports can yield large files if you process many epochs with lenient thresholds.
- If you change detection parameters, re-click **“Plot Bursts”** (for single epoch) or **“Process All Epochs & Export”** to regenerate updated results.

