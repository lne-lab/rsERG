# rsERG
This code allows analysis on rsERG data

# rsERG Analysis Pipeline – Nature Submission

This repository contains the Python tools used in the accompanying manuscript to analyze resting-state electroretinography (rsERG) recordings from multichannel corneal arrays.

All analysis is run in **Jupyter Notebook / JupyterLab** using interactive GUIs built with `ipywidgets` and `ipyfilechooser`.


There are three main pipelines:

1. **Spectral analysis (PSD + FOOOF)**
   - `psd_compute.py`
   - `psd_clean_v1b.py`
   - `psd_channel_means_v1b.py`
   - `excel_to_1darray_to_FOOOF_v1A11_agg.py`

2. **Cycle-by-cycle burst analysis on the MEA**
   - `MEA_burst_detect_v9C8.py`

3. **Multimetric single-channel burst analysis**
   - `Cycle_Hilb_Spect_pACF_v9C2Nv2E_env50_piecewise_consistency_autoGUI_artToggle_promPctGuardFinal.py`

All three pipelines can be accessed in the same Jupyter Notebook called Resting_State_ERG_Pipeline, which is accessible within the same repository as the python modules. 

---

## 1. Requirements

These scripts are designed to run with:

- Python ≥ 3.9  
- Jupyter Notebook / JupyterLab

Core packages (non-exhaustive, but required):

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `mne`
- `ipywidgets`
- `ipyfilechooser`
- `openpyxl`
- `tqdm`
- `fooof` (for spectral parametrization)
- `python-pptx` (optional; for PPT export in `psd_channel_means_v1b.py`)

Enable widgets in Jupyter, e.g.:

```bash
jupyter nbextension enable --py widgetsnbextension
