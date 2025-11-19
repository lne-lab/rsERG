# rsERG Analysis Pipeline 

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
   - `epoch_plotter_v8_GUI.py`

All three pipelines can be accessed in the same Jupyter Notebook called Resting_State_ERG_Pipeline, which is accessible within the same repository as the python modules. 

---

## 1. Requirements

These scripts are designed to run with:

- Python ≥ 3.9  
- Jupyter Notebook / JupyterLab

Core packages:

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
  
## 2. Setting up a Python virtual environment and JupyterLab (Bash)

The easiest way for reviewers to run the notebooks and modules is to use a local Python virtual environment in the same folder as this repository.

1. **Download / clone this repository**

   ```bash
   # Option A: clone from git
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_FOLDER>

   # Option B: download as ZIP, unzip, then:
   cd <REPOSITORY_FOLDER>
   
## 3. Create and activate a virtual environment


python3 -m venv .venv
source .venv/bin/activate    # on macOS / Linux / WSL

## 4. Install dependencies (including JupyterLab)

A requirements.txt file is provided:


pip install -r requirements.txt


## 5. Launch JupyterLab

From the same folder (with the virtual environment still activated):

```bash
jupyter lab
