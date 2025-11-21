# FOOOF Analysis GUI Tutorial (`excel_to_1darray_to_FOOOF_v1A11_agg.py`)

This tutorial explains how to use the Jupyter-based GUI in `excel_to_1darray_to_FOOOF_v1A11_agg.py` to:

- Load PSD data from one or more **Excel** (`.xlsx`) files  
- Run **FOOOF** fits on each PSD
- Plot the fits (all spectra or those closest to the mean)
- Optionally export figures, a pickle of all FOOOFGroup objects, and an Excel summary table

---

## 1. Requirements

Install the required packages (ideally in a virtual environment):

```bash
pip install pandas numpy ipywidgets matplotlib fooof scipy tqdm
pip install python-pptx   # optional, only needed for PPTX export
pip install jupyterlab
```

The script also uses:

- `pickle`, `os`, `re`, `math`
- `tqdm.notebook` (for progress bars)
- `tkinter` (built-in with most Python installs) for file dialogs

Make sure:

- You run this **locally** (not on a remote server without GUI), because `tkinter` opens native file/directory dialogs.
- `ipywidgets` are enabled in your Jupyter environment (recent JupyterLab/Notebook versions usually work out of the box).

---

## 2. What the Script Expects and Produces

### Input: Excel PSD files

The module supports three **analysis modes**, selected in the dropdown labeled **“Analysis Mode”**:

1. **Grouped Mice**  
   - Reads the **first sheet** in each `.xlsx` file.  
   - Assumes:
     - **Column A** = frequency  
     - **Columns B+** = PSD vectors (one column per group/mouse/eye).  
   - Returns a dictionary per file:
     ```python
     {
       "freq": freq_array,
       "Mouse1_Eye1": psd_array,
       "Mouse1_Eye2": psd_array,
       ...
     }
     ```

2. **Individual Mice** (old approach; “Eye Averages”)  
   - Reads the sheet named **"Eye Averages"**.  
   - Expects:
     - Column A: Frequencies  
     - Column B: `"Eye1 Average PSD"`  
     - Column C: `"Eye2 Average PSD"`  
   - Returns:
     ```python
     {
       "freq": freq_array,
       "Eye1 Average PSD": eye1_psd,
       "Eye2 Average PSD": eye2_psd
     }
     ```

3. **Individual Mice Means** (new approach)  
   - Reads the **first sheet**.  
   - Expects:
     - Column A: PSD for Eye1  
     - Column B: Frequency  
     - Column C: PSD for Eye2  
     - Column D: Frequency (redundant; ignored)  
   - Returns:
     ```python
     {
       "freq": freq_array,
       "Eye1": eye1_psd,
       "Eye2": eye2_psd
     }
     ```

Each Excel file becomes one entry in a global dictionary `psd_data_dict_all`.

### Output: FOOOF fits and exports

For every PSD column (except `"freq"`), the script:

- Builds a **FOOOFGroup** object.
- Fits the PSD in a specified frequency range.
- Drops fits with **R² below a threshold**.
- Stores all FOOOFGroup objects in a dictionary:

```python
fg_dict = {
  "ColumnName_FileSuffix": FOOOFGroup(...),
  ...
}
```

Depending on the export options you choose, the GUI can then:

1. **Plot fits** (all or “closest to mean” examples) in Jupyter.  
2. **Export figures** in:
   - `PNG`, `SVG`, `JPEG`, or
   - `PPTX` (PowerPoint; one slide per figure).
3. **Save FOOOFGroup dictionary** as a pickle:
   - `fooof_groups.pkl`
4. **Export a summary Excel**:
   - `fooof_results.xlsx` with columns:
     - `dict_key`, `fit_index`, `r_squared`,
     - `aperiodic_params`, `peak_params`.
5. **Display a summary table** of all fits inside the notebook.

---

## 3. How to Run the GUI

1. Start Jupyter in the folder containing `excel_to_1darray_to_FOOOF_v1A11_agg.py`:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. In a new notebook cell, run:

   ```python
   %run excel_to_1darray_to_FOOOF_v1A11_agg.py
   ```

3. A GUI titled something like  
   **“FOOOF analysis and export GUI”** (ending in “Process and Analyze”)  
   will appear in the notebook output.

If it does not appear, check the cell output for errors and ensure `ipywidgets` and `fooof` are installed correctly.

---

## 4. Step-by-Step: Running FOOOF on Your PSDs

### 4.1 Select Excel Files

- Click **“Select Excel Files”**.
- A native file browser opens (Tk dialog).
- Choose one or more `.xlsx` files and confirm.
- The selected file paths are printed in the notebook output.

### 4.2 Select Output Directory

- Click **“Select Output Directory”**.
- Choose a folder where you want to save:
  - Figures,
  - `fooof_groups.pkl`,
  - `fooof_results.xlsx` (if enabled).
- The chosen path is printed.

> If you enable any export options but forget to choose a directory, the script will warn you and not proceed.

### 4.3 Choose Analysis Mode

Use the **“Analysis Mode”** dropdown:

- `Grouped Mice`
- `Individual Mice`
- `Individual Mice Means`

Make sure it matches how your Excel files are structured (see Section 2).

### 4.4 Configure FOOOF Parameters

Under **“Configure FOOOF Parameters”**:

- **Freq Min / Freq Max**  
  - Frequency range (e.g. `4–45 Hz`) within which FOOOF is fit.

- **Amplitude Threshold**  
  - `min_peak_height` used by FOOOF (e.g. `0.2`).

- **R² Threshold**  
  - Fits with `r_squared` below this are dropped.

- **Max Peaks**  
  - `max_n_peaks` allowed in each fit.

- **Fitting Mode**  
  - `"knee"` or `"fixed"` aperiodic mode.

- **Peak Width Min / Max**  
  - `peak_width_limits` in Hz (e.g. `2–10 Hz`).

These are passed directly into `FOOOFGroup`.

### 4.5 Optional Plot Settings

You can further refine the plots:

- **X Axis Min / Max**  
  - Plotting window in Hz.

- **Y Axis Min / Max**  
  - PSD axis limits (arbitrary units of the model power).

- **Show Grid**  
  - Check to add a grid behind the plot.

- **X Tick Font Size**  
  - Slider controlling size of x-axis tick labels.

- **Include R² in Plot Titles**  
  - Adds `R²: xx` to each plot title.

- **Include Peak Params Table**  
  - After each plot, prints a table of peaks (center frequency, amplitude, FWHM) if present.

### 4.6 Export Options

- **Export Figures**  
  - If checked, figures are saved to disk.
  - Otherwise, they are only shown in the notebook.

- **Use Closest-to-Mean Plotting**  
  - If checked and there are multiple fits in a FOOOFGroup, the script:
    - Computes the mean spectrum.
    - Finds the **10 spectra closest** to this mean (Euclidean distance).
    - Plots those as representative examples.
  - If unchecked, **all** fits in each FOOOFGroup are plotted.

- **Choose Figure Export Formats**  
  - Check one or more of:
    - `PNG`, `SVG`, `JPEG`, `PPT`.  
  - If `PPT` is selected **and** `python-pptx` is installed:
    - A PowerPoint file is created with each figure on its own slide.

---

## 5. Exporting FOOOF Results and Tables

### 5.1 Process and Analyze

Click **“Process and Analyze”**:

1. The script validates:
   - Excel files selected,
   - Output directory (if exports enabled),
   - Chosen figure formats.

2. It processes the Excel files according to **Analysis Mode**.

3. It runs FOOOF fits:

   ```python
   fg_dict = run_fooof_analysis(
       psd_data_dict_all,
       freq_range=[freq_min, freq_max],
       amp_threshold=amp_threshold,
       r2_threshold=r2_threshold,
       max_peaks=max_peaks,
       fitting_mode=fitting_mode,
       peak_width_limits=[peak_width_min, peak_width_max],
   )
   ```

4. For each entry in `fg_dict` (each PSD column):

   - Plots either:
     - “Closest to mean” examples, or
     - All PSD fits.
   - Applies axis, grid, and font settings.
   - Exports figures in the selected formats (if enabled).

### 5.2 Save FOOOFGroup Dictionary

- **Export as Pickle** checkbox:  
  - If checked, saves a `fooof_groups.pkl` file in the chosen output directory.
  - You can reload it later:

    ```python
    import pickle
    with open("fooof_groups.pkl", "rb") as f:
        fg_dict = pickle.load(f)
    ```

### 5.3 Export Excel Summary

- **Export as Excel** checkbox:  
  - If checked, creates `fooof_results.xlsx` in the output directory.
  - Each row contains:
    - `dict_key` (column+file label),
    - `fit_index`,
    - `r_squared`,
    - `aperiodic_params` (as a string),
    - `peak_params` (as a string).

You can then filter/sort these results in Excel or import them into Python/R for further analysis.

### 5.4 Display Export Table in Notebook

- **Display Export Table** checkbox:  
  - If checked, prints a pandas DataFrame summary of all FOOOF fits at the end of the run.

---

## 6. Notes and Limitations

- The GUI is designed for **post-hoc spectral analysis**:
  - It does **not** compute PSDs itself; it assumes PSDs are already in Excel.
- Each FOOOFGroup here typically receives **one PSD** (shape `(1, n_freqs)`), but the code is written to handle multiple fits per group if present.
- Fits with poor R² are dropped before export and plotting.
- Axis limits are not auto-scaled by the script beyond what Matplotlib normally does; if plots look odd, try relaxing or adjusting `Y Axis Min/Max`.
- If you change FOOOF parameters or export options, simply click **“Process and Analyze”** again to rerun and regenerate all figures and tables.

You can add this file to your repository as `excel_to_fooof_tutorial.md` and link it from your main `README.md`.
