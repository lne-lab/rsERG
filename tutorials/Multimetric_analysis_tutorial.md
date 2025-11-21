# RSEGViewer Tutorial  
`Cycle_Hilb_Spect_pACF_v9C2Nv2E_env50_piecewise_consistency_autoGUI_artToggle_promPctGuardFinal.py`

This tutorial explains how to use the **RSEGViewer** GUI to inspect and quantify narrow-band oscillatory bursts in single-channel rsERG data stored as MNE `.fif` epochs.

RSEGViewer is implemented in:

- Main module:  
  `Cycle_Hilb_Spect_pACF_v9C2Nv2E_env50_piecewise_consistency_autoGUI_artToggle_promPctGuardFinal.py`
- Helper module:  
  `epoch_plotter_v8_GUI.py` (provides `compute_pacf_stability_over_time` and related utilities)

The GUI class you will interact with is **`RSEGViewer`**.

---

## 1. Purpose and Overview

RSEGViewer provides an interactive Jupyter GUI that lets you:

- Load an **MNE Epochs `.fif`** file (e.g. single-channel rsERG).
- Choose a **channel** and **epoch** to inspect.
- Band-pass filter around a chosen band (e.g. 2–4, 5–7, …, 32–34 Hz).
- Compute a **Hilbert envelope** and detect **candidate burst segments**.
- Gate segments using:
  - Envelope percentile threshold,
  - **pACF stability** over time,
  - **Wavelet power** in-band vs flanks,
  - **Cycle-by-cycle features** (ByCycle-inspired),
  - Optional **artifact masks** and **PSD 1/f peak gate**.
- Visualize all of this in a multi-panel figure for a single epoch.
- Batch process **all kept epochs for one channel** and export an **Excel file** summarizing segment-level and epoch-level metrics.

Use RSEGViewer when you want to cross-validate burst detection heuristics for rsERG “hum” activity and export a reproducible summary of burst vs non-burst segments across epochs.

---

## 2. Requirements and Expected Inputs

### 2.1 Python Environment

RSEGViewer is designed to run inside a Jupyter notebook. You will need at least:

```bash
pip install mne numpy scipy matplotlib pandas ipywidgets openpyxl
```

Make sure:

- Jupyter Notebook or JupyterLab is installed.
- **ipywidgets** is enabled so sliders and buttons render properly.
- The helper file `epoch_plotter_v8_GUI.py` is on your Python path (e.g. in the same folder as your notebook and main module).

### 2.2 Input Data

**Required input:** an MNE `Epochs` object saved as `.fif`:

- Shape: `(n_epochs, n_channels, n_times)`
- Sampling frequency (`sfreq`) is taken from `epochs.info['sfreq']`.
- Typically you analyze **one rsERG channel** at a time, but the file may contain multiple channels; you choose which one via a channel slider in the GUI.
- In the manuscript, 5-s epochs were used and passed into RSEGViewer.

**Recommended preprocessing:**

- Basic pre-cleaning and band-limiting (e.g. 0.5–45 Hz) before creating the `.fif` file.
- Optional prior artifact cleaning is fine; RSEGViewer’s multi-criterion artifact logic is an additional layer.

### 2.3 Output

When you click **“Process All Epochs”** for a given channel, RSEGViewer writes an Excel workbook:

- Location: **same directory** as the input `.fif`.
- File name pattern:  
  `"<basename>__ch<channel>_segments.xlsx"`  
  e.g. `recording_baseline__ch0_segments.xlsx`

The workbook contains three sheets:

1. **`data`** – One row per segment (burst or fail), with columns such as:
   - `Epoch`, `SegIdx`, `Start_s`, `End_s`, `Duration_s`, `Band_Hz`
   - `Valid_Cycles`, `PACF_OK`, `Power_OK`, `Artifact`
   - `Env_Frac`, `PACF_Frac`, `Wave_Frac`, `Env_OK`
   - `OneF_Z`, `OneF_OK`
   - `Status` = `"burst"` or `"fail"` (final classification)

2. **`summary`** – A single row describing:
   - File name, channel index, sampling rate
   - Band-pass settings (Low/High Hz, filter order)
   - Envelope percentile, `Min_Cycles`
   - Cycle thresholds (`AmpFrac`, `AmpCons`, `PerCons`, `Mono`)
   - pACF parameters (window, step, lag range, stability threshold)
   - Wavelet flank width and overlap thresholds
   - Artifact cleaner configuration (mode, thresholds, presets)
   - Counts and percentages of burst vs fail segments

3. **`agreement`** – One row per epoch, summarizing agreement between envelope vs pACF segmentation:
   - `Jaccard`, `Precision`, `Recall`, `F1`, `SegMed` (median segment length), etc.

---

## 3. Launching RSEGViewer in Jupyter

1. Place these files in the same folder as your notebook (or anywhere on your `PYTHONPATH`):

   - `Cycle_Hilb_Spect_pACF_v9C2Nv2E_env50_piecewise_consistency_autoGUI_artToggle_promPctGuardFinal.py`
   - `epoch_plotter_v8_GUI.py`

2. In a Jupyter notebook cell:

   ```python
   from Cycle_Hilb_Spect_pACF_v9C2Nv2E_env50_piecewise_consistency_autoGUI_artToggle_promPctGuardFinal import RSEGViewer

   viewer = RSEGViewer()
   ```

3. Instantiating `RSEGViewer()` builds and displays the full widget GUI.

You should see:

- A **FIF path** text box plus **Load File** and **Process All Epochs** buttons.
- Sliders for **channel**, **epoch**, and **time window**.
- Panels for:
  - Band-pass and envelope settings
  - Cycle thresholds (ByCycle-inspired)
  - pACF parameters
  - Wavelet gating and overlap thresholds
  - Artifact rejection
  - PSD 1/f gate and axis controls

A large output area below the controls will show multi-panel figures and log text.

---

## 4. Interactive Usage (Single Epoch)

### 4.1 Load File and Choose Channel / Epoch

1. Enter the full path to your `.fif` file in the **FIF Path** text box, for example:

   ```text
   C:/path/to/your_data/my_recording_epochs.fif
   ```

   or on Unix-like systems:

   ```text
   /home/username/data/my_recording_epochs.fif
   ```

2. Click **Load File**:
   - RSEGViewer calls `mne.read_epochs(...)` with `preload=True`.
   - Internal state is initialized (number of epochs, sampling rate, etc.).
   - Channel and epoch sliders become active.

3. Use:
   - The **channel slider** to select which channel to analyze.
   - The **epoch slider** to select the epoch index.
   - The **time window** controls (`time_start`, `time_end`) to focus on a sub-window (in seconds) within that epoch, if desired.

Every time you change these, the viewer re-runs the analysis and updates the multi-panel visualization.

---

### 4.2 Band-pass, Envelope, and Cycle Thresholds

#### Band-pass and envelope

Key controls:

- **Band-pass** – `Low Hz`, `High Hz`, and filter order:
  - Sets the band used for Hilbert envelope, pACF, wavelet, and cycle detection.
  - In the paper, RSEGViewer was used to scan a series of **narrow bands**:
    - 2–4, 5–7, 8–10, 11–13, 14–16, 17–19, 20–22, 23–25, 26–28, 29–31, 32–34 Hz.

- **Min Cycles**
  - Minimum number of consecutive valid cycles required for a segment to be eligible as a burst.

- **Envelope percentile** (`Env %ile`)
  - Percentile of the Hilbert envelope in the current epoch used as a threshold.
  - Samples above this threshold are considered “high envelope” and contribute to `Env_OK` and `Env_Frac`.

#### Cycle thresholds (ByCycle-inspired)

Cycle features are computed in a ByCycle-inspired way for each trough–to–trough cycle in the band-passed signal. The main thresholds are:

- **`AmpFrac`** – minimum absolute amplitude per cycle (in the signal’s native units).
- **`AmpCons`** – amplitude consistency:
  - Ratios of consecutive rise/decay amplitudes; 1.0 means identical amplitudes, 0.5 means one cycle is twice the amplitude of the other.
- **`PerCons`** – period consistency:
  - Ratios of consecutive cycle periods; higher values demand more uniform inter-cycle timing.
- **`Mono`** – monotonicity:
  - Fraction of steps in rise and decay that have the expected sign (up during rise, down during decay).

Additional optional cycle QC:

- **`Period CV`** and **`Amp CV`**
  - Coefficient-of-variation style checks across cycles in a segment; used to filter segments with internally inconsistent cycles.

Cycles that fail one or more thresholds are marked invalid and do not count towards forming a “burst” of length `Min Cycles`.

#### Band-specific tuning of PerCons and AmpCons

Because power drops with frequency due to 1/f scaling, and narrowband filtering can make noise appear smoother, **single large outliers can mimic bursts**, especially at higher frequencies. To control this, **period consistency** and **amplitude consistency** were explicitly tuned as a function of band in the manuscript.

**Period consistency (`PerCons`)**

For a band `[f_min, f_max]`, the center frequency is

\[
f_c = rac{f_{	ext{min}} + f_{	extmax}}{2}.
\]

RSEGViewer uses a period consistency threshold that corresponds to allowing roughly ±1 Hz deviation from the center:

\[
	ext{PerCons}_	ext{thresh} pprox rac{f_c - 1}{f_c}.
\]

Examples:

- 8–10 Hz band → \( f_c = 9 \) Hz → \( 	ext{PerCons} = 8/9 pprox 0.889 \)
- 32–34 Hz band → \( f_c = 33 \) Hz → \( 	ext{PerCons} = 32/33 pprox 0.970 \)

Higher bands therefore impose **stricter** period consistency, reflecting the expectation that true narrowband high-frequency bursts have very similar periods.

**Amplitude consistency (`AmpCons`)**

Amplitude consistency is defined as a ratio between the smaller and larger cycle amplitudes:

- `AmpCons = 1.0` → two cycles have identical amplitude.
- `AmpCons = 0.5` → one cycle is twice the amplitude of the other.

Across bands, the manuscript used an **increasing threshold**:

- Start around **0.25** for the 2–4 Hz band.
- Increase to **0.30** for the 5–7 Hz band.
- Continue in increments of ~0.05.
- End around **0.85** for the 32–34 Hz band.

This reflects that:

1. Higher-frequency bursts are expected to have **more uniform envelopes** over their shorter durations.
2. Stricter amplitude consistency at higher bands helps suppress motion-like spikiness and filter-induced pseudo-regularity.

**Practical tip:**  
Whenever you change the band, re-tune `PerCons` and `AmpCons` based on the single-epoch plots:

- Use the (f_c−1)/f_c heuristic for `PerCons` as a starting point.
- Increase `AmpCons` for higher frequencies; relax only if clearly rhythmic segments are being rejected.

---

### 4.3 pACF Stability and Wavelet Gating

RSEGViewer uses **partial autocorrelation (pACF)** and **wavelet power** as additional, band-specific evidence for oscillatory structure.

#### pACF stability

Controls include:

- Window length (`PACF Win`, in seconds)
- Step size between windows (`PACF Step`, seconds)
- Lag range (`Min Lag`, `Max Lag`, `n Lags`)
- Stability threshold (`PACF Stab Th`)

Internally, RSEGViewer calls `compute_pacf_stability_over_time` (from `epoch_plotter_v8_GUI.py`) to compute pACF-based stability scores across time. A segment is considered `PACF_OK` if the pACF stability exceeds the threshold over a sufficient fraction of its samples (`PACF Overlap %`).

An **auto-consistency** option uses the current band to propose reasonable default pACF parameters.

#### Wavelet gating

Wavelet gating compares in-band power to neighboring **flank** bands:

- `Flank Hz` – distance in Hz between the band and its flanks.
- `Wave Pow %` – percentile threshold for in-band power relative to flanks.
- `Wave Prom %` – percentile threshold for a prominence-like ratio (band vs flanks).
- `Wave Overlap %` – required fraction of samples in a segment that must satisfy wavelet criteria to be `Power_OK`.

Together, pACF and wavelet gates help distinguish genuine narrowband hum from smooth, non-oscillatory noise.

---

### 4.4 Artifact Cleaning and 1/f Peak Gate

RSEGViewer implements a **multi-criterion artifact rejection** similar to the manuscript, plus an optional within-epoch spike mask.

#### 4.4.1 Epoch-level artifact rejection (criteria i–v)

For each 5-s epoch, RSEGViewer computes time- and frequency-domain metrics using **robust z-scores** (median/MAD). The logic matches these five criteria:

1. **Large amplitude deflections (i)**  
   - Detect amplitude |z| excursions lasting ≥ ~10 ms that exceed a hard threshold (e.g. |z| ≥ 8).  
   - Epochs with such runs are rejected.

2. **Sharp, fast spikes in the derivative (ii)**  
   - Compute z-scores of the first difference (signal derivative).  
   - Detect |z| excursions lasting ≥ ~8–10 ms above a hard threshold (e.g. |z| ≥ 8).  
   - Epochs with such derivative spikes are rejected.

3. **Low-frequency drift power (iii)**  
   - Compute Welch PSD, integrate power in a drift band (e.g. 0.5–1 Hz).  
   - Convert to dB and compute robust z across epochs; epochs with drift power z ≥ threshold (e.g. z ≥ 8) are rejected.

4. **Broadband 1/f power surges (iv)**  
   - Fit a simple 1/f aperiodic component to the PSD in a broader band.  
   - Evaluate the fit at 20 Hz, compute z across epochs (the “intercept” z).  
   - Epochs with intercept z >= threshold (e.g. z ≥ 8) are rejected.

5. **Repeated moderate excursions (v)**  
   - Track the **fraction** of samples where amplitude |z| exceeds a softer threshold (e.g. |z| ≥ 5) and a **rate** of derivative events where derivative |z| ≥ 5.  
   - Epochs with many moderate events (≥ about three excursions) are rejected even if no single excursion hits the hard thresholds in (i) or (ii).

In RSEGViewer:

- You can run artifact rejection in **fixed** mode (explicit numeric thresholds) or **quantile** mode (thresholds derived from chosen quantiles across epochs).
- In **fixed mode**, you can directly set soft and hard z thresholds, required durations, and drift/1/f cutoffs to match the manuscript as closely as you like.
- In **quantile mode**, RSEGViewer converts the continuous metrics (`amp_frac`, `dor_rate`, drift power, intercept at 20 Hz) into thresholds based on user-selected quantiles; epochs above those quantiles are rejected.

**Optional rescue loop**

RSEGViewer can optionally run a **“rescue” loop**:

- If enabled, and the initial thresholds would reject too many epochs (i.e. the kept fraction drops below a configurable minimum), the viewer gradually **relaxes** the quantile thresholds until a minimum keep fraction is restored or a cap is reached.
- If disabled, no rescue is performed: the artifact thresholds are applied as-is, and all epochs that fail them remain rejected.

The final keep/reject decision is stored internally as an **epoch mask**:

- Only kept epochs are available in the epoch slider and in **Process All Epochs**.

#### 4.4.2 Within-epoch local spike masking (optional)

Beyond deciding which epochs to keep, RSEGViewer can apply a **local spike mask** inside each kept epoch:

- A short sliding window is used to compute local robust z-scores for amplitude and derivative.
- Samples that exceed the local spike thresholds are marked as artifacts.
- A small temporal padding is added around each spike to capture the full transient.
- When **“Use artifact rejection”** is enabled:
  - Overlap with this local mask contributes to the `Artifact` flag for each segment.
  - Artifact-contaminated samples reduce the effective overlap fractions for envelope, pACF, and wavelet (“how much of this segment is clean and passes each gate?”).

Masked regions are shown as shaded spans in the raw/filtered trace panels so you can visually inspect that spikes are being captured without erasing genuine oscillations.

#### 4.4.3 PSD 1/f peak gate

A separate **1/f peak gate** can be enabled:

- The PSD is fitted with a simple 1/f model in a specified frequency range.
- Residuals are computed in the band of interest (band-pass region).
- A z-score (`OneF_Z`) of that residual amplitude in-band is compared to a threshold (`1/f Z thr`).
- Segments failing this gate are marked `OneF_OK = False`.

This is **not** a full FOOOF implementation; it’s a lightweight check that segments classified as bursts also show an in-band excess above the aperiodic 1/f background.

---

### 4.5 Figure Layout (Single Epoch)

Each time the GUI updates (after changing parameters, channel, epoch, or time window), RSEGViewer draws a multi-panel figure that typically includes:

- **pACF stability vs time**  
  - Time-resolved pACF stability curves with segments highlighted.

- **Envelope vs pACF overlay**  
  - Binary segmentations from envelope and pACF.
  - Summary metrics (Jaccard, precision, recall, F1, median segment length).

- **Raw + filtered traces**  
  - Raw signal and band-passed signal.
  - Valid/invalid segments highlighted.
  - Artifact regions shaded (if local mask enabled).

- **Cycle panel**  
  - Filtered signal with trough, peak, and mid-rise/mid-decay markers.
  - Useful to assess whether cycle thresholds behave as expected.

- **Segment classification (Q panel)**  
  - Visual summary of segment-level gates (`Valid_Cycles`, `PACF_OK`, `Power_OK`, `Env_OK`, `OneF_OK`, `Artifact`, final `Status`).

- **Hilbert envelope panel**  
  - Envelope vs threshold (Env %ile).
  - Candidate segments where envelope remains high.

- **Wavelet spectra**  
  - Extended time–frequency plot.
  - Band-restricted view for the band of interest.

- **PSD + 1/f fit**  
  - Power spectrum for the epoch.
  - 1/f fit and residuals, with band indicated.
  - In-band residual z-score (`OneF_Z`).

- **Segment table**  
  - A per-segment summary for the current epoch.

Use this visual summary to check that your thresholds are behaving sensibly before running the full batch export.

---

## 5. Batch Processing and Excel Export

Once you are satisfied with the settings for a given **channel** and band:

1. Ensure epoch-level artifact rejection has been run (if you are using it) and that the **kept epochs** make sense.
2. Set the **channel slider** to the channel you want to export.
3. Click **Process All Epochs**.

RSEGViewer will then:

1. Determine which epoch indices to process (kept epochs only if artifact rejection is active; all epochs otherwise).
2. For each epoch:
   - Apply the same analysis used for the current single-epoch view:
     - Band-pass, envelope, cycle detection, pACF, wavelet, artifact masks, 1/f gate.
   - Generate:
     - A list of per-segment dictionaries (`rows`) with all gates evaluated.
     - A small per-epoch agreement summary (`agree`) comparing envelope vs pACF segmentation.
3. Concatenate all `rows` into the **`data`** table.
4. Build the **`summary`** table (one row) from the current GUI settings and global counts.
5. Build the **`agreement`** table (one row per epoch).
6. Write all three tables to an Excel file next to your `.fif` for that channel.

Progress and a final summary are printed in the notebook output (e.g. number of segments exported, burst vs fail ratio, file path).

If Excel writing fails (e.g. missing `openpyxl` or permissions issues), an error message is shown instead.

---

## 6. Notes and Limitations

- **Single-channel at export**  
  `Process All Epochs` operates on the **currently selected channel** only. To analyze additional channels, change the channel slider and run again (producing one Excel file per channel).

- **ByCycle-inspired, not a direct reimplementation**  
  Cycle detection and metrics (amplitude, period, monotonicity, consistency) are conceptually aligned with ByCycle but implemented locally. The thresholds (`AmpFrac`, `AmpCons`, `PerCons`, `Mono`) act directly on these custom metrics. Advanced ByCycle features (e.g. detailed symmetry metrics, full group-level interfaces) are not implemented here.

- **Band-dependent thresholds**  
  In the manuscript’s narrow 2–4 … 32–34 Hz scans, `PerCons` and `AmpCons` are explicitly tuned per band to reflect expected regularity and to suppress 1/f-driven pseudo-regularity. If you analyze different bands or data types, you should re-optimize these thresholds based on the single-epoch plots.

- **Artifact detection is heuristic and configurable**  
  Multi-criterion artifact rejection (criteria i–v) plus optional local spike masking are tuned for rsERG-style recordings. When applied to other preparations, the mode (fixed vs quantile), thresholds, and optional rescue loop should be revisited and validated visually.

- **Rescue loop is optional**  
  Automatic “rescue” of epochs (relaxing thresholds if too few are kept) is optional. If you disable it, RSEGViewer uses your artifact thresholds as-is and will not attempt to restore a minimum keep fraction.

- **FOOOF is not used here**  
  The PSD 1/f gate uses a simple aperiodic fit and z-scored residuals; it is not a full FOOOF peak fitting routine.

- **Multiplicative gating**  
  Many criteria (cycles, pACF, wavelet, envelope, artifacts, 1/f) are combined. Overly strict settings can easily eliminate almost all segments. If you see very few bursts:
  - Start with lenient presets,
  - Relax `AmpCons`, `PerCons`, `Mono`, and pACF / overlap thresholds,
  - Temporarily disable the 1/f gate and/or artifact rejection to see which component is constraining the most.

RSEGViewer is intended as a transparent, tunable pipeline for rsERG burst detection; the GUI and exported Excel files are designed to make these thresholds and decisions explicit and easy to audit.
