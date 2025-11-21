# PSD GUI Tutorial (`psd_compute.py`)

This tutorial explains how to use the Jupyter-based GUI in `psd_compute.py` to compute and visualize power spectral density (PSD) from MNE `Epochs` stored in `.fif` files.

---

## 1. Requirements

Install the required Python packages (ideally in a virtual environment):

```bash
pip install mne ipywidgets ipyfilechooser tqdm scipy matplotlib joblib
pip install jupyterlab
```

The script also uses:

- `numpy`
- `pickle`
- `scipy.signal`
- `matplotlib`
- `tqdm.notebook`
- `ipywidgets`
- `ipyfilechooser`
- `mne`

Make sure widgets are enabled in your Jupyter environment. In most recent JupyterLab versions this works out of the box.

---

## 2. What the Script Expects

**Input**

- A pre-filtered MNE `Epochs` object saved as `.fif` or `.fif.gz`.
- Shape: `(n_epochs, n_channels, n_times)`.

The script reads this with:

```python
mne.read_epochs(path_to_fif, preload=True)
```

**Output**

- A `.pkl` file containing a dictionary of PSDs per channel:

```python
psd_results = {
    "channel_name": {
        "psd":   np.ndarray of shape (n_epochs, n_freqs),
        "freqs": np.ndarray of shape (n_freqs,)
    },
    ...
}
```

Channels that fail (e.g. all-NaN data) are stored as:

```python
{"psd": None, "freqs": None}
```

and are excluded from plotting.

---

## 3. How to Run the GUI

1. Start Jupyter in the directory containing `psd_compute.py`:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. Create a new notebook and run:

   ```python
   %run psd_compute.py
   ```

3. The GUI will appear in the notebook output.  
   If nothing appears, check that:
   - The cell ran without errors.
   - `ipywidgets` is properly installed/enabled.

---

## 4. Step-by-Step: Using the GUI

The GUI is organized into four main actions.

### 4.1 Select the Input `.fif` File

- Use the **“Input .fif File”** chooser.
- Navigate to your filtered `.fif` / `.fif.gz` file and click it.
- The selected path is shown under the chooser.

If the path is invalid, the log/output area will print an error when you try to compute PSDs.

---

### 4.2 Set PSD Parameters

Two widgets control the Welch PSD computation:

- **Window Length (s)**  
  Segment length for Welch (`nperseg`). Must be positive and shorter than the epoch duration.

- **Overlap (%)**  
  Percentage overlap between consecutive windows, from `0` to `<100`.

Internally:

```python
sfreq   = epochs.info["sfreq"]
nperseg = int(window_length_s * sfreq)
noverlap = int(nperseg * (overlap_percent / 100))
```

If `nperseg` exceeds the number of samples in an epoch, the script will stop and print a clear error.

---

### 4.3 Choose Output File and Compute PSD

1. Use the **“Output Pickle File”** chooser to set the output `.pkl` path.
   - The filename must end with `.pkl`.

2. Click **“Compute PSD”**.

For each channel:

- Data from all epochs is extracted.
- PSD is computed using `scipy.signal.welch` with a Hamming window.
- Frequencies are restricted to 0–100 Hz.
- A progress bar shows status across channels.
- Any failures (e.g. NaN-only channels) are reported.

At the end:

- The progress bar turns green (`success`).
- A summary is printed (number of channels with valid PSDs).
- The channel selector for plotting is automatically populated.

---

### 4.4 Plot PSDs

Once PSDs are computed:

1. **Select Channels**  
   In **“Select Channels”**, choose one or more channels.

2. **Pick a Plot Mode**

   - **“Mean & Positive Std”**  
     Plots the mean PSD and a shaded region from `mean` to `mean + std`.

   - **“Individual Traces & Mean”**  
     Plots each epoch’s PSD in light gray with the mean in blue.

3. **Set Axis Limits (Optional)**

   - `X Min (Hz)` / `X Max (Hz)` define the frequency range.
   - `Y Min (V²/Hz)` / `Y Max (V²/Hz)` define the PSD range.

   The script checks that `min < max` for each axis.  
   After computation, `Y Max` is initialized to a bit above the global maximum PSD value.

4. **Click “Plot PSD”**

   - One figure is created per selected channel.
   - The selected plot mode and axis limits are applied.

If PSDs have not been computed or channels are empty, a message is printed instead of raising an error.

---

## 5. Reusing the Saved PSDs

You can load the `.pkl` file in any Python script or notebook:

```python
import pickle

with open("psd_results.pkl", "rb") as f:
    psd_results = pickle.load(f)

ch_name = "CH1"
psd   = psd_results[ch_name]["psd"]   # shape: (n_epochs, n_freqs)
freqs = psd_results[ch_name]["freqs"]
```

From here you can:

- Compute band-limited power
- Fit oscillatory peaks
- Average across epochs or channels
- Feed the PSDs into any downstream analysis pipeline

---

## 6. Notes and Limitations

- The GUI is intended for **Jupyter Notebook/JupyterLab** and relies on `ipywidgets`.
- Input files must be MNE `Epochs` objects saved as `.fif` / `.fif.gz` with valid `sfreq` and channel info.
- The current implementation restricts output to **0–100 Hz**. Use axis controls to zoom into narrower bands.
- Channels with invalid PSDs are kept in `psd_results` but marked with `None` and hidden from plotting.

