# PSD Cleaning & Export GUI Tutorial (`psd_clean_v1b.py`)

This tutorial explains how to use the Jupyter-based GUI in `psd_clean_v1b.py` to **review, clean, and export PSD data** that was previously computed (e.g. with `psd_compute.py`).

---

## 1. Requirements

Install the required packages (ideally in a virtual environment):

```bash
pip install ipywidgets ipyfilechooser matplotlib numpy
pip install python-pptx  # optional, only needed for PPTX export
pip install jupyterlab
```

The script also uses:

- `pickle`
- `math`
- `os`
- `IPython.display` (`display`, `clear_output`)

Make sure `ipywidgets` are enabled in your Jupyter environment (recent JupyterLab versions usually work out of the box).

---

## 2. What the Script Expects and Produces

### Input: PSD `.pkl` file

A pickle file containing PSDs per channel, e.g.:

```python
psd_results = {
    "Ch1": {
        "psd":   np.ndarray of shape (n_epochs, n_freqs),
        "freqs": np.ndarray of shape (n_freqs,)
    },
    "Ch2": {...},
    ...
}
```

Typically this is the output from `psd_compute.py`.

### Output: Cleaned PSD + Figures

The GUI lets you:

1. **Plot PSDs with automatic epoch exclusion**, based on:
   - Low-frequency band outliers (e.g. 1–3 Hz),
   - Repeated extreme peaks in test bands (e.g. 7–20 Hz).

2. **Export figures**:
   - As individual image files (`.png`, `.svg`, etc.),
   - Or combined into a single PowerPoint file (`.pptx`, if `python-pptx` is installed).

3. **Export a cleaned PSD `.pkl` file**:
   - Contains only the **kept epochs** for each channel.
   - Same structure as the original `psd_results`, but with fewer epochs.

---

## 3. How to Run the GUI

1. Start Jupyter in the folder containing `psd_clean_v1b.py`:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. Open a new notebook and run:

   ```python
   %run psd_clean_v1b.py
   ```

3. A GUI titled  
   **“PSD Plotter GUI (Threshold-based Exclusion + Cleaned Export)”**  
   will appear in the notebook output.

If it does not appear, check the cell output for errors and confirm that `ipywidgets` are installed.

---

## 4. Step-by-Step: Inspecting and Cleaning PSDs

### 4.1 Load PSD

- In the **“Load PSD”** panel:
  - Click **Load PSD Pickle**.
  - Use the file chooser to select your `.pkl` PSD file.
- The log area will print:
  - Confirmation that the file was loaded,
  - A list of channel names (e.g. `Ch1`, `Ch2`, ...),
  - Which are also added to the **Select Channels** box.

> Note: The code assumes channel names like `"Ch1"`, `"Ch2"`, ... for numeric sorting. If yours differ, they will still load, but the sort may be non-numerical.

### 4.2 Choose Channels

- In **“Select Channels”**, highlight one or more channels (Ctrl/Cmd-click or Shift-click).
- These are the channels that will be plotted and cleaned.

### 4.3 Set Exclusion Parameters

In **“Plotting Options”**, you can control:

- **LowBandMin / LowBandMax**  
  Frequency range (e.g. `1–3 Hz`) used to detect low-frequency outliers.

- **LowBand Thr**  
  Threshold multiplier for the low band (e.g. `3.0` = 3× the mean PSD in that band).

- **TestBand Thr**  
  Multiplier for test bands (e.g. `10.0` = 10× the mean PSD in the band).

- **Test Bands**  
  A text field containing tuples like:
  ```text
  (7,9),(9,11),(11,13),(13,15),(15,17),(17,19),(19,21)
  ```
  Each band is checked for extreme peaks. If a trace is extreme in enough bands, it is excluded.

> The script compares each epoch’s PSD to the **mean PSD across epochs** and marks suspicious traces as “excluded”.

### 4.4 Plotting Options & Layout

- **Show Kept Traces / Show Excluded Traces**  
  Toggle whether to draw each group.

- **Show Original Mean / Show New Mean**  
  - Original = mean over all epochs,
  - New = mean over only the kept epochs.

- **Show Vertical Lines / Vertical Lines (Hz)**  
  Add reference lines at specific frequencies (e.g. `10,15`).

- **Axis Ranges** (`X Min`, `X Max`, `Y Min`, `Y Max`)  
  Optional manual limits for frequency and PSD axes.

- **Font Sizes**  
  Control title, axis, legend, and tick label sizes.

- **Columns per Row**  
  Controls how many subplots per row (e.g. 4).

When ready, click **Plot PSDs**.  
The GUI will:

- Apply the exclusion rules per channel,
- Plot kept vs excluded traces,
- Compute and show new vs original mean PSDs,
- Store the kept epoch indices for each channel (used later for the cleaned export).

A short message will confirm how many figures were generated and that kept indices are recorded.

---

## 5. Exporting Figures and Cleaned PSD

### 5.1 Export Figures

In the **“Export Figures”** section:

1. Use the **file chooser** to specify an output path and filename, e.g.:
   - `MyPsdPlots.png`
   - `MyPsdPlots.svg`
   - `MyPsdPlots.pptx`

2. Click **Export Figures**.

Behavior:

- If the extension is **`.ppt` / `.pptx`**:
  - A PowerPoint file is created.
  - Each figure becomes a separate slide with an embedded image.
  - Requires `python-pptx` to be installed.

- For other extensions (`.png`, `.jpg`, `.svg`, ...):
  - Each figure is saved as a separate file:
    - `MyPsdPlots_1.png`
    - `MyPsdPlots_2.png`
    - etc.

The log area prints the paths of all exported files.

### 5.2 Export Cleaned PSD

In the **“Export Cleaned PSD”** section:

1. Use the file chooser to select or type a `.pkl` filename, e.g.:
   - `psd_results_cleaned.pkl`

2. Click **Export Cleaned PSD**.

The script will:

- Use the stored kept indices for each channel,
- Build a new dictionary with only the kept epochs:

  ```python
  cleaned_data = {
      "Ch1": {"psd": psd[kept_ix], "freqs": freqs},
      "Ch2": {...},
      ...
  }
  ```

- Save this to the specified `.pkl` file.

Only channels that were **plotted** (and thus had kept indices computed) are included.

---

## 6. Notes and Limitations

- This GUI is meant to work **after** PSD computation (e.g. with `psd_compute.py`).
- Exclusion logic is **threshold-based** and uses the average PSD as reference; it is not a statistical test.
- If you change exclusion parameters, you must **re-plot** before exporting cleaned PSDs.
- If no channels are selected or no figures are plotted, exports will be skipped with a message rather than an error.
- You can version and track your cleaning by:
  - Saving the cleaned `.pkl` with a suffix (e.g. `_v1`, `_v2`),
  - Keeping a note of the parameter choices used in your notebook.

