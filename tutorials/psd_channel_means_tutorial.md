# Channel-Mean PSD GUI Tutorial (`psd_channel_means_v1b.py`)

This tutorial explains how to use the Jupyter-based GUI in `psd_channel_means_v1b.py` to:

- Load PSD data (from a `.pkl` file, e.g. output of `psd_compute.py` or `psd_clean_v1b.py`)
- Compute **per-channel mean PSDs** (before and after outlier exclusion)
- Group channels into **Eye1 (Ch1–Ch8)** and **Eye2 (Ch9–Ch16)**
- Compute and plot **group means** for each eye
- Export figures and final “mean-of-means” data

---

## 1. Requirements

Install the required packages (ideally in a virtual environment):

```bash
pip install ipywidgets ipyfilechooser matplotlib numpy
pip install jupyterlab
pip install python-pptx  # optional, only needed for PPTX export
```

The script also uses:

- `pickle`, `os`, `math`
- `IPython.display` (`display`, `clear_output`)

Make sure widgets are enabled in your Jupyter environment (recent JupyterLab versions usually work out of the box).

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

This is typically the output from `psd_compute.py` (raw PSDs) or `psd_clean_v1b.py` (cleaned PSDs).

Internally, the GUI:

- Computes **original channel mean PSDs** across epochs.
- Applies an **exclusion rule** (low-band and test-band thresholds) to drop outlier epochs.
- Computes **new means** from the kept epochs.

### Output

1. **Figures**:
   - “Screening” plots for Eye1 and/or Eye2, showing per-channel means.
   - “Final group” plots with selected channels and group mean overlays.
   - A “Final Means of Means Only” plot with Eye1/Eye2 group means.

2. **Exported files** (via GUI):
   - Image files (`.png`, `.svg`, `.jpg`, etc.) or PowerPoint (`.pptx`) with the plotted figures.
   - A data file (`.pkl` or `.xlsx`) containing the final Eye1/Eye2 **mean-of-means** arrays and their frequency vectors.

---

## 3. How to Run the GUI

1. Start Jupyter in the folder containing `psd_channel_means_v1b.py`:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. In a new notebook, run:

   ```python
   %run psd_channel_means_v1b.py
   ```

3. A GUI titled  
   **“Enhanced PSD Plotter GUI with ipyfilechooser Exports”**  
   will appear in the notebook output.

If it does not appear, check the cell output for errors and confirm that `ipywidgets` and `ipyfilechooser` are installed.

---

## 4. Step-by-Step: From Channel Means to Group Means

### 4.1 Load PSD

- Under **“1) Load PSD Pickle”**:
  - Click **Load PSD Pickle**.
  - Use the file chooser to select your PSD `.pkl` file.
- The log panel will:
  - Confirm successful loading,
  - List available channels (e.g. `Ch1`–`Ch16`),
  - Populate the **Select Channels** widget.

> Channels are sorted numerically if named `Ch1`, `Ch2`, etc.

### 4.2 Select Channels and Eyes

- Under **“2) Select Channels”**, highlight one or more channels.
- In **“Plotting Options”**, the checkboxes:
  - **Show Eye1 (Ch1–8)** – controls whether Eye1 subplot is used.
  - **Show Eye2 (Ch9–16)** – controls whether Eye2 subplot is used.

The script automatically assigns:

- **Eye1** = channels with numbers 1–8 (e.g. `Ch1`–`Ch8`)
- **Eye2** = channels with numbers 9–16 (e.g. `Ch9`–`Ch16`)

### 4.3 Exclusion Parameters

In **“Plotting Options”**, two numeric fields control epoch exclusion:

- **Low Band Threshold**  
  Scales the mean PSD in a low band (fixed at 1–3 Hz).  
  If an epoch exceeds `threshold × mean` in that band, it is excluded.

- **Test Band Threshold**  
  Used for a set of test bands (default: `(7,9),(9,11),...,(19,21)` Hz).  
  If an epoch repeatedly exceeds `threshold × mean` across these bands, it is excluded.

The test-band ranges can be edited in the **“Test Bands”** text field.

These parameters are applied when computing the **“new”** channel means.

### 4.4 Plot Screening PSDs

- Optionally adjust:
  - Axis limits (`X-axis Min/Max`, `Y-axis Min/Max`),
  - Font sizes (title, axes, legend, ticks).
- Click **Plot PSDs**.

The GUI will:

1. Compute original and new channel means for all loaded channels.
2. Create **screening plots**:
   - Eye1 subplot (if enabled and Eye1 channels are selected),
   - Eye2 subplot (if enabled and Eye2 channels are selected).
3. For each eye, it overplots:
   - Original means (solid lines),
   - New means (dashed lines), depending on the toggles.

These screening plots are meant to help you visually decide which channels to include in the final group mean.

### 4.5 Choose Channels for Final Group Means

After plotting, the GUI shows **selection widgets** for Eye1 and Eye2:

- Each has a multi-select list of that eye’s channels.
- By default, all available channels for that eye are selected.

Click **Plot Final Group Means**:

- For each eye, the script:
  - Uses the **new means** (after exclusion),
  - Computes the **group mean** across the selected channels.
- It then:
  - Plots final Eye1/Eye2 channel means and overlays the group mean (black line) in a new figure.
  - Creates a separate **“Final Means of Means Only”** figure with only the Eye1/Eye2 group means.

The final Eye1/Eye2 group mean arrays and their frequency axes are stored internally for export:

- `Eye1_MeanOfMeans`, `Eye1_freqs`
- `Eye2_MeanOfMeans`, `Eye2_freqs`

---

## 5. Exporting Figures and Final Means

There are two export sections.

### 5.1 Export Subplot Figures (Screening + Final)

Under **“Export Subplots (Screening/Final) Figures”**:

1. Use the file chooser to set a base filename and extension, e.g.:
   - `MyPsdMeans.png`
   - `MyPsdMeans.svg`
   - `MyPsdMeans.pptx`

2. Click **Export Figures**.

Behavior:

- If the extension is **`.ppt` / `.pptx`**:
  - All current figures (screening, final, means-only) are added as slides in a PowerPoint file.
- Otherwise (e.g. `.png`, `.svg`):
  - Each figure is saved as a separate file:
    - `MyPsdMeans_1.png`
    - `MyPsdMeans_2.png`
    - etc.

The export log lists all saved files.

### 5.2 Export Final Means Data/Figure

Under **“Export Final Means Plot/Data”**:

1. Use the file chooser to select or type a filename with one of the supported extensions:
   - `.pkl` – pickle of `final_data_dict`:
     - Contains Eye1/Eye2 mean-of-means arrays and freqs.
   - `.xlsx` / `.xls` – Excel file with each key as a column.
   - Image formats (`.png`, `.svg`, `.jpg`, `.jpeg`) – the “Final Means of Means Only” figure.
   - `.ppt` / `.pptx` – PowerPoint with that final figure on a slide.

2. Click **Export Final Means**.

The GUI will save either the final data, the figure, or both, depending on the chosen extension.

---

## 6. Notes and Limitations

- This GUI is designed as a **post-processing step** after PSD computation and (optionally) cleaning.
- The Eye1/Eye2 mapping is hard-coded to `Ch1–Ch8` and `Ch9–Ch16`. If your naming scheme differs, you may need to adapt the code.
- Exclusion is **threshold-based** and uses the average PSD as reference; it is not a formal statistical test.
- You must re-run **Plot PSDs** and **Plot Final Group Means** after changing thresholds or channel selections before exporting.
- Export will politely fail with a log message (not a hard error) if required plots or data have not been generated.

