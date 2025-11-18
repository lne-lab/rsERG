import os
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib.patches import Rectangle
from epoch_plotter_v8_GUI import compute_pacf_stability_over_time
from IPython.display import display, clear_output
from scipy.signal import welch

# ===== Adaptive Consistency (piecewise amplitude, ±1 Hz period) =====
def _per_cons_thr_delta_hz(f, delta_hz=1.0):
    """Return period-consistency ratio capturing ±delta_hz around f: min(f/(f+Δ), (f-Δ)/f)."""
    try:
        f = float(f)
    except Exception:
        return 0.0
    if not np.isfinite(f) or f <= max(delta_hz, 1e-9):
        return 0.0
    return min(f / (f + delta_hz), (f - delta_hz) / f)


# Piecewise amplitude-consistency thresholds (editable)
# Each tuple = (low_hz, high_hz, amp_cons_threshold)
_AMP_CONS_BANDS = [
    (5.0,  7.0,  0.30),
    (8.0,  10.0, 0.35),
    (11.0, 13.0, 0.45),
    (14.0, 16.0, 0.55),
    (17.0, 19.0, 0.60),
    (20.0, 22.0, 0.65),
    (23.0, 25.0, 0.70),
    (26.0, 28.0, 0.75),
    (29.0, 31.0, 0.80),
    (32.0, 34.0, 0.85),
]

def _amp_cons_thr_piecewise(f, default_low=0.30, default_high=0.90):
    """Map instantaneous frequency to a piecewise amp-cons threshold; looser at low bands, tighter at high bands."""
    try:
        f = float(f)
    except Exception:
        return default_low
    if not np.isfinite(f):
        return default_low
    for lo, hi, thr in _AMP_CONS_BANDS:
        if lo <= f <= hi:
            return float(thr)
    # Outside explicit bands: extrapolate by clamping to ends
    if f < _AMP_CONS_BANDS[0][0]:
        return float(_AMP_CONS_BANDS[0][2])
    if f > _AMP_CONS_BANDS[-1][1]:
        return float(_AMP_CONS_BANDS[-1][2])
    return float(default_low)

# =====================================================================
# Artifact cleaner (pre-filter), adapted from artifact_cleaner_fif_v9
# =====================================================================
def _robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else 1e-12

def _amp_abs_z(x: np.ndarray) -> np.ndarray:
    s = _robust_mad(x)
    return np.abs((x - np.median(x)) / s)

def _derivative_abs_z(x: np.ndarray, fs: float) -> np.ndarray:
    dx = np.diff(x, prepend=x[0])
    s = _robust_mad(dx)
    return np.abs(dx / s)

def _any_cluster_over(mask: np.ndarray, fs: float, min_ms: float) -> bool:
    if not np.any(mask):
        return False
    min_len = max(1, int(round((min_ms / 1000.0) * fs)))
    run = 0
    for v in mask:
        if v:
            run += 1
            if run >= min_len:
                return True
        else:
            run = 0
    return False

def _aperiodic_intercept_at(power_f: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float, f_at: float) -> float:
    m = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(m):
        return np.nan
    f = freqs[m]
    P = power_f[m]
    x = np.log10(np.clip(f, 1e-9, None))
    y = np.log10(np.clip(P, 1e-18, None))
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    x_at = np.log10(max(f_at, 1e-9))
    y_at = a + b * x_at
    return float(y_at)

def _welch_power_metrics(x: np.ndarray, fs: float, drift_band=(0.5,1.0), fmin=0.5, fmax=45.0, intercept_at=20.0) -> tuple:
    nperseg = min(len(x), int(fs * 2))
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2, detrend='constant')
    m = (f >= drift_band[0]) & (f <= drift_band[1])
    P_low = np.trapz(pxx[m], f[m]) if np.any(m) else np.nan
    P_low_db = 10.0 * np.log10(P_low + 1e-12)
    inter20 = _aperiodic_intercept_at(pxx, f, fmin, fmax, intercept_at)
    return P_low_db, inter20

def _compute_artifact_metrics_per_epoch(data_ep: np.ndarray, fs: float, cfg: dict) -> dict:
    n_ep = data_ep.shape[0]
    amp_frac  = np.empty(n_ep, float)
    dor_rate  = np.empty(n_ep, float)
    plow_db   = np.empty(n_ep, float)
    inter20   = np.empty(n_ep, float)
    veto_amp  = np.zeros(n_ep, dtype=bool)
    veto_dor  = np.zeros(n_ep, dtype=bool)
    for i in range(n_ep):
        xi = data_ep[i].astype(float)
        z_amp = _amp_abs_z(xi)
        amp_frac[i] = float(np.mean(z_amp >= float(cfg['art_amp_soft_z'])))
        z_dor = _derivative_abs_z(xi, fs)
        hits = np.where(z_dor >= float(cfg['art_dor_soft_z']))[0]
        if hits.size == 0:
            dor_rate[i] = 0.0
        else:
            min_d = int(float(cfg['art_dor_refractory_s']) * fs)
            keep, last = [], -min_d
            for h in hits:
                if h - last >= min_d:
                    keep.append(h)
                    last = h
            dor_rate[i] = len(keep) / (len(xi) / fs)
        d1, d2 = _welch_power_metrics(xi, fs,
                                      drift_band=(cfg['art_plow_lo'], cfg['art_plow_hi']),
                                      fmin=cfg['art_fmin'], fmax=cfg['art_fmax'],
                                      intercept_at=cfg['art_intercept_hz'])
        plow_db[i] = d1
        inter20[i] = d2
        veto_amp[i] = _any_cluster_over(z_amp >= float(cfg['art_amp_veto_z']), fs, float(cfg['art_amp_veto_min_ms']))
        veto_dor[i] = _any_cluster_over(z_dor >= float(cfg['art_dor_veto_z']), fs, float(cfg['art_dor_veto_min_ms']))
    return dict(amp_frac=amp_frac, dor_rate=dor_rate, plow_db=plow_db, intercept20=inter20, veto_amp=veto_amp, veto_dor=veto_dor)

def _artifact_decide_keep(m: dict, strategy: str, preset: str, cfg: dict) -> tuple:
    amp_frac = m['amp_frac']; dor_rate = m['dor_rate']; plow_db = m['plow_db']; inter20 = m['intercept20']
    veto = m['veto_amp'] | m['veto_dor']
    if strategy == 'quantile':
        qa = cfg['art_q_amp']; qd = cfg['art_q_dor']; qp = cfg['art_q_plow']; qi = cfg['art_q_inter']
        thr_a = np.nanquantile(amp_frac, qa)
        thr_d = np.nanquantile(dor_rate, qd)
        thr_p = np.nanquantile(plow_db, qp)
        thr_i = np.nanquantile(inter20, qi)
        c_a = amp_frac >= thr_a
        c_d = dor_rate >= thr_d
        c_p = plow_db >= thr_p
        c_i = inter20 >= thr_i
        reject = veto | c_a | c_d | c_p | c_i
        keep = ~reject
        keep_frac = float(np.mean(keep))
        steps = 0
        if cfg['art_rescue_enabled']:
            AMPQ, DORQ, PLOWQ, INTERQ = qa, qd, qp, qi
            while (keep_frac < float(cfg['art_rescue_min_keep_frac'])) and (keep_frac < float(cfg['art_rescue_cap'])) and (steps < int(cfg['art_rescue_steps_max'])):
                PLOWQ  = min(PLOWQ + float(cfg['art_rescue_delta_power']), 0.99)
                INTERQ = min(INTERQ + float(cfg['art_rescue_delta_power']), 0.99)
                if cfg['art_rescue_relax_spikes']:
                    AMPQ = min(AMPQ + float(cfg['art_rescue_delta_spike']), 0.99)
                    DORQ = min(DORQ + float(cfg['art_rescue_delta_spike']), 0.99)
                thr_a = np.nanquantile(amp_frac, AMPQ)
                thr_d = np.nanquantile(dor_rate, DORQ)
                thr_p = np.nanquantile(plow_db, PLOWQ)
                thr_i = np.nanquantile(inter20, INTERQ)
                c_a = amp_frac >= thr_a
                c_d = dor_rate >= thr_d
                c_p = plow_db >= thr_p
                c_i = inter20 >= thr_i
                reject = veto | c_a | c_d | c_p | c_i
                keep = ~reject
                keep_frac = float(np.mean(keep))
                steps += 1
        keep_mask = ~reject
        details = dict(thr_a=float(thr_a), thr_d=float(thr_d), thr_p=float(thr_p), thr_i=float(thr_i))
        return keep_mask, details
    else:
        thr_a = float(cfg['art_fixed_amp_bad_frac'])
        thr_d = float(cfg['art_fixed_dor_eps'])
        thr_p = (np.nanpercentile(plow_db, 90) if np.isnan(cfg['art_fixed_plow_db_thr']) else float(cfg['art_fixed_plow_db_thr']))
        thr_i = (np.nanpercentile(inter20, 90) if np.isnan(cfg['art_fixed_inter_thr']) else float(cfg['art_fixed_inter_thr']))
        c_a = amp_frac >= thr_a
        c_d = dor_rate >= thr_d
        c_p = plow_db >= thr_p
        c_i = inter20 >= thr_i
        reject = veto | c_a | c_d | c_p | c_i
        keep_mask = ~reject
        details = dict(thr_a=float(thr_a), thr_d=float(thr_d), thr_p=float(thr_p), thr_i=float(thr_i))
        return keep_mask, details


warnings.filterwarnings(
    "ignore",
    ".*No burst detection thresholds are provided.*",
    category=UserWarning
)

# =====================================================================
# Helpers from your original pipeline (unchanged)
# =====================================================================
# ---------------------------------------------------------------------
# Cycle marker helpers
# ---------------------------------------------------------------------
def _first_crossing(x, thr, direction="up"):
    x = np.asarray(x)
    hits = np.where(x >= thr)[0] if direction == "up" else np.where(x <= thr)[0]
    if hits.size == 0:
        return None, None
    i = hits[0]
    if i == 0:
        return 0, 0.0
    x0, x1 = x[i - 1], x[i]
    frac = 0.0 if x1 == x0 else (thr - x0) / (x1 - x0)
    return i - 1, np.clip(frac, 0, 1)


def _compute_half_amp_points(sig, trough_idx, peak_idx):
    if np.isnan(trough_idx) or np.isnan(peak_idx):
        return np.nan
    trough_idx, peak_idx = int(trough_idx), int(peak_idx)
    if peak_idx <= trough_idx:
        return np.nan
    tr_v = sig[trough_idx]
    pk_v = sig[peak_idx]
    amp = pk_v - tr_v
    if amp == 0:
        return np.nan
    half_v = tr_v + 0.5 * amp
    seg = sig[trough_idx:peak_idx + 1]
    i, frac = _first_crossing(seg, half_v, direction="up")
    if i is None:
        return np.nan
    return trough_idx + i + frac


def _compute_half_amp_fall(sig, peak_idx, next_trough_idx):
    if np.isnan(peak_idx) or np.isnan(next_trough_idx):
        return np.nan
    peak_idx, next_trough_idx = int(peak_idx), int(next_trough_idx)
    if next_trough_idx <= peak_idx:
        return np.nan
    pk_v = sig[peak_idx]
    tr_v = sig[next_trough_idx]
    amp = pk_v - tr_v
    if amp == 0:
        return np.nan
    half_v = tr_v + 0.5 * amp
    seg = sig[peak_idx:next_trough_idx + 1]
    i, frac = _first_crossing(seg, half_v, direction="down")
    if i is None:
        return np.nan
    return peak_idx + i + frac


# ---------------------------------------------------------------------
# In-band vs flank prominence (unchanged)
# ---------------------------------------------------------------------
def wavelet_peak_prominence(power, freqs, lf, hf, flank_hz):
    """Return time-resolved in-band / flank power ratio (NaN if no flanks)."""
    freqs = np.asarray(freqs)
    in_band = (freqs >= lf) & (freqs <= hf)
    low_flk = (freqs >= max(lf - flank_hz, freqs[0])) & (freqs < lf)
    high_flk = (freqs > hf) & (freqs <= min(hf + flank_hz, freqs[-1]))

    if np.any(in_band):
        band_pow = np.nanmean(power[in_band], axis=0)
    else:
        band_pow = np.full(power.shape[1], np.nan)

    rows = []
    if np.any(low_flk):
        rows.append(power[low_flk])
    if np.any(high_flk):
        rows.append(power[high_flk])
    if rows:
        flank_pow = np.nanmean(np.concatenate(rows, axis=0), axis=0)
    else:
        flank_pow = np.full_like(band_pow, np.nan)

    return band_pow / (flank_pow + 1e-12)



def _robust_z_vector(v, eps=1e-9):
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = 1.4826 * max(mad, eps)
    return (v - med) / scale

def psd_aperiodic_and_peaks(x, fs, fmin=2.0, fmax=40.0, nperseg_s=2.0):
    """Welch PSD + log–log 1/f fit -> residual z, and dB curves for plotting.
    Returns: f, P_db, Pfit_db, f_fit, z_resid
    """
    nperseg = int(max(64, min(len(x), round(nperseg_s*fs))))
    noverlap = int(round(0.5*nperseg))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant')
    mask = (f >= fmin) & (f <= fmax)
    f_fit = f[mask]
    P_fit = Pxx[mask]
    if f_fit.size == 0:
        P_db = 10.0*np.log10(np.clip(Pxx, 1e-18, None))
        Pfit_db = np.full_like(P_db, np.nan)
        return f, P_db, Pfit_db, f_fit, np.array([])

    xlog = np.log10(np.clip(f_fit, 1e-9, None))
    ylog = np.log10(np.clip(P_fit, 1e-18, None))
    A = np.vstack([np.ones_like(xlog), xlog]).T
    coef, *_ = np.linalg.lstsq(A, ylog, rcond=None)
    a, b = coef
    yhat = a + b * xlog
    resid = ylog - yhat
    z_resid = _robust_z_vector(resid).ravel()

    P_db = 10.0*np.log10(np.clip(Pxx, 1e-18, None))
    Pfit_db = np.full_like(P_db, np.nan)
    yhat_lin = 10**yhat
    Pfit_db[mask] = 10.0*np.log10(np.clip(yhat_lin, 1e-18, None))
    return f, P_db, Pfit_db, f_fit, z_resid


def _autoscale_db(ax, arrays, pad_db=3.0, robust=True):
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        v = np.asarray(arr)
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return
    y = np.concatenate(vals)
    if robust and y.size > 10:
        lo, hi = np.nanpercentile(y, [2, 98])
    else:
        lo, hi = np.nanmin(y), np.nanmax(y)
    if np.isfinite(lo) and np.isfinite(hi):
        ax.set_ylim(lo - pad_db, hi + pad_db)

# =====================================================================
# Trimmed Bycycle-style cycle detector (plugged in)
# =====================================================================
# NOTE: this replaces direct calls to bycycle.compute_features for the
#       purpose of *burst* definition, while preserving your GUI & plots.

def _mono_fraction(sig: np.ndarray, t1: int, pk: int, t2: int) -> float:
    """Fraction of strictly monotonic steps across rise & decay."""
    rise = sig[t1:pk+1]
    decay = sig[pk:t2+1]
    rise_m = np.mean(np.diff(rise) > 0) if rise.size > 1 else np.nan
    decay_m = np.mean(np.diff(decay) < 0) if decay.size > 1 else np.nan
    if np.isnan(rise_m):
        return decay_m
    if np.isnan(decay_m):
        return rise_m
    return 0.5 * (rise_m + decay_m)


def cycles_from_signal(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> pd.DataFrame:
    d = np.diff(sig)
    troughs = np.where((d[:-1] < 0) & (d[1:] > 0))[0] + 1
    peaks   = np.where((d[:-1] > 0) & (d[1:] < 0))[0] + 1
    rows = []
    for i in range(len(troughs) - 1):
        t1, t2 = troughs[i], troughs[i + 1]
        pk_cand = peaks[(peaks > t1) & (peaks < t2)]
        if pk_cand.size == 0:
            continue
        pk = pk_cand[np.argmax(sig[pk_cand])]
        period = (t2 - t1) / fs
        freq = 1.0 / period if period else np.nan
        rows.append(dict(
            trough_start=t1,
            trough_end=t2,
            peak=pk,
            amp=sig[pk] - min(sig[t1], sig[t2]),
            period=period,
            frequency=freq,
            is_valid=(fmin <= freq <= fmax),
            amp_cons=np.nan,
            period_cons=np.nan,
            monotonicity=np.nan,
            burst_id=-1,
        ))
    return pd.DataFrame(rows)


def annotate_cycles(df: pd.DataFrame,
                    sig: np.ndarray,
                    fs: float,
                    amp_frac: float,
                    amp_cons_thr: float,
                    per_cons_thr: float,
                    mono_thr: float) -> pd.DataFrame:
    """
    Cycle validation with frequency-adaptive thresholds:
      - Period consistency uses ±1 Hz tolerance per cycle: min(f/(f+1), (f-1)/f).
      - Amplitude consistency uses explicit piecewise thresholds by band (see _AMP_CONS_BANDS).
    The UI-provided amp_cons_thr / per_cons_thr act as floors (adaptive thresholds will never be looser).
    """
    if df.empty:
        return df

    # monotonicity per cycle
    for i, r in df.iterrows():
        df.at[i, 'monotonicity'] = _mono_fraction(sig, int(r.trough_start), int(r.peak), int(r.trough_end))

    # neighbor consistency ratios (min vs neighbors)
    for i in range(len(df)):
        neigh = [j for j in (i - 1, i + 1) if 0 <= j < len(df)]
        a_i, p_i = df.loc[i, 'amp'], df.loc[i, 'period']
        amp_ratios = [min(a_i, df.loc[j, 'amp']) / max(a_i, df.loc[j, 'amp'])
                      for j in neigh if a_i > 0 and df.loc[j, 'amp'] > 0]
        per_ratios = [min(p_i, df.loc[j, 'period']) / max(p_i, df.loc[j, 'period'])
                      for j in neigh if p_i > 0 and df.loc[j, 'period'] > 0]
        df.at[i, 'amp_cons'] = min(amp_ratios) if amp_ratios else np.nan
        df.at[i, 'period_cons'] = min(per_ratios) if per_ratios else np.nan

    # Floors from UI
    amp_floor = float(amp_cons_thr)
    per_floor = float(per_cons_thr)

    # apply adaptive thresholds
    for i, r in df.iterrows():
        f = float(r.frequency) if np.isfinite(r.frequency) else np.nan

        # Period consistency: ±1 Hz tolerance as ratio, at least the floor
        eff_per_thr = _per_cons_thr_delta_hz(f, delta_hz=1.0) if np.isfinite(f) else per_floor
        eff_per_thr = max(per_floor, eff_per_thr)

        # Amplitude consistency: piecewise per band, at least the floor
        eff_amp_thr = _amp_cons_thr_piecewise(f) if np.isfinite(f) else amp_floor
        eff_amp_thr = max(amp_floor, eff_amp_thr)

        ok = True
        if (not np.isfinite(r.amp)) or (r.amp < float(amp_frac)): ok = False
        if (not np.isfinite(r.amp_cons)) or (r.amp_cons < eff_amp_thr): ok = False
        if (not np.isfinite(r.period_cons)) or (r.period_cons < eff_per_thr): ok = False
        if (not np.isfinite(r.monotonicity)) or (r.monotonicity < float(mono_thr)): ok = False
        df.at[i, 'is_valid'] = bool(ok)

    return df
    # monotonicity per cycle
    for i, r in df.iterrows():
        df.at[i, 'monotonicity'] = _mono_fraction(sig, int(r.trough_start), int(r.peak), int(r.trough_end))
    # neighbor consistency (min of ratios vs left/right neighbors)
    for i in range(len(df)):
        neigh = [j for j in (i - 1, i + 1) if 0 <= j < len(df)]
        a_i, p_i = df.loc[i, 'amp'], df.loc[i, 'period']
        amp_ratios = [min(a_i, df.loc[j, 'amp']) / max(a_i, df.loc[j, 'amp']) for j in neigh if a_i > 0 and df.loc[j, 'amp'] > 0]
        per_ratios = [min(p_i, df.loc[j, 'period']) / max(p_i, df.loc[j, 'period']) for j in neigh if p_i > 0 and df.loc[j, 'period'] > 0]
        df.at[i, 'amp_cons'] = min(amp_ratios) if amp_ratios else np.nan
        df.at[i, 'period_cons'] = min(per_ratios) if per_ratios else np.nan
    # apply thresholds
    for i, r in df.iterrows():
        if (r.amp < amp_frac or pd.isna(r.amp_cons) or r.amp_cons < amp_cons_thr
            or pd.isna(r.period_cons) or r.period_cons < per_cons_thr
            or pd.isna(r.monotonicity) or r.monotonicity < mono_thr):
            df.at[i, 'is_valid'] = False
    return df


def label_bursts(df: pd.DataFrame, min_cycles: int) -> pd.DataFrame:
    df['burst_id'] = -1 if 'burst_id' not in df else df['burst_id']
    valid = df.index[df.is_valid].tolist()
    if not valid:
        return df
    cur_group = [valid[0]]
    bid = 0
    for prev, cur in zip(valid[:-1], valid[1:]):
        if cur == prev + 1:
            cur_group.append(cur)
        else:
            if len(cur_group) >= min_cycles:
                df.loc[cur_group, 'burst_id'] = bid
                bid += 1
            cur_group = [cur]
    if len(cur_group) >= min_cycles:
        df.loc[cur_group, 'burst_id'] = bid
    return df


def detect_bursts_cycles(sig: np.ndarray,
                         fs: float,
                         fmin: float,
                         fmax: float,
                         amp_frac: float,
                         amp_cons_thr: float,
                         per_cons_thr: float,
                         mono_thr: float,
                         min_cycles: int) -> pd.DataFrame:
    df = cycles_from_signal(sig, fs, fmin, fmax)
    df = annotate_cycles(df, sig, fs, amp_frac, amp_cons_thr, per_cons_thr, mono_thr)
    df = label_bursts(df, min_cycles)
    return df


# =====================================================================
# RSEGViewer
# =====================================================================
class RSEGViewer:
    def __init__(self):


        # file
        self.path_text = widgets.Text(
            placeholder='/path/to/file.fif', description='FIF Path:',
            layout=widgets.Layout(width='600px'))
        self.load_btn = widgets.Button(description='Load File', button_style='info')
        self.proc_btn = widgets.Button(description='Process All Epochs', button_style='success')

        # ---- PSD 1/f gate & axes controls ----
        self.onef_gate = widgets.Checkbox(value=False, description='Use PSD 1/f gate')
        self.onef_zthr = widgets.FloatText(value=1.5, description='1/f Z thr:')
        self.onef_fit_lo = widgets.FloatText(value=2.0, description='Fit lo (Hz):')
        self.onef_fit_hi = widgets.FloatText(value=40.0, description='Fit hi (Hz):')
        self.psd_auto_x = widgets.Checkbox(value=True, description='Auto X range')
        self.psd_xmin = widgets.FloatText(value=2.0, description='X min (Hz):')
        self.psd_xmax = widgets.FloatText(value=40.0, description='X max (Hz):')
        self.psd_auto_y = widgets.Checkbox(value=True, description='Auto Y range')
        self.psd_ymin = widgets.FloatText(value=-20.0, description='Y min (dB):')
        self.psd_ymax = widgets.FloatText(value=50.0, description='Y max (dB):')


        # ---- PSD 1/f gate ----
        self.onef_gate = widgets.Checkbox(value=False, description='Use PSD 1/f peak gate')
        self.onef_zthr = widgets.FloatText(value=1.5, description='1/f Z thr:')
        self.onef_fit_lo = widgets.FloatText(value=2.0, description='1/f fit lo:')
        self.onef_fit_hi = widgets.FloatText(value=40.0, description='1/f fit hi:')


        # ---- Artifact cleaner widgets ----
        self.art_mode   = widgets.Dropdown(options=['quantile','fixed'], value='fixed', description='Mode:', layout=widgets.Layout(width='180px'))
        self.art_preset = widgets.Dropdown(options=['strict','lenient','very_lenient'], value='very_lenient', description='Preset:', layout=widgets.Layout(width='220px'))
        self.art_apply_btn = widgets.Button(description='Re-clean now', button_style='warning', layout=widgets.Layout(width='140px'))
        self.art_summary_label = widgets.Label(value='—', layout=widgets.Layout(width='400px'))
        self.art_amp_soft_z = widgets.FloatText(value=10.0, description='Amp Z thr:')
        self.art_dor_soft_z = widgets.FloatText(value=10.0, description='Deriv Z thr:')
        self.art_dor_refractory_s = widgets.FloatText(value=0.25, description='Deriv refr. (s):')
        self.art_amp_veto_z = widgets.FloatText(value=10.0, description='Amp veto Z:')
        self.art_amp_veto_min_ms = widgets.FloatText(value=10.0, description='Amp veto ms:')
        self.art_dor_veto_z = widgets.FloatText(value=10.0, description='Deriv veto Z:')
        self.art_dor_veto_min_ms = widgets.FloatText(value=10.0, description='Deriv veto ms:')
        self.art_plow_lo = widgets.FloatText(value=0.5, description='Drift lo Hz:')
        self.art_plow_hi = widgets.FloatText(value=1.0, description='Drift hi Hz:')
        self.art_fmin = widgets.FloatText(value=0.5, description='PSD fmin:')
        self.art_fmax = widgets.FloatText(value=45.0, description='PSD fmax:')
        self.art_intercept_hz = widgets.FloatText(value=20.0, description='Intercept @Hz:')
        self.art_q_amp  = widgets.FloatSlider(min=0.5,max=0.99,step=0.01,value=0.85, description='Q amp_frac:')
        self.art_q_dor  = widgets.FloatSlider(min=0.5,max=0.99,step=0.01,value=0.85, description='Q dor_rate:')
        self.art_q_plow = widgets.FloatSlider(min=0.5,max=0.99,step=0.01,value=0.85, description='Q plow_db:')
        self.art_q_inter= widgets.FloatSlider(min=0.5,max=0.99,step=0.01,value=0.85, description='Q intercept:')
        self.art_fixed_amp_bad_frac = widgets.FloatText(value=0.05, description='Amp bad frac ≥')
        self.art_fixed_dor_eps      = widgets.FloatText(value=1.5,  description='Deriv eps ≥')
        self.art_fixed_plow_db_thr  = widgets.FloatText(value=float('nan'), description='Plow dB thr:')
        self.art_fixed_inter_thr    = widgets.FloatText(value=float('nan'), description='Intercept thr:')
        self.art_rescue_enabled       = widgets.Checkbox(value=True, description='Enable rescue')
        self.art_rescue_min_keep_frac = widgets.FloatText(value=0.05, description='Min keep frac')
        self.art_rescue_cap           = widgets.FloatText(value=0.90, description='Rescue cap')
        self.art_rescue_delta_spike   = widgets.FloatText(value=0.01, description='Δ spike Q')
        self.art_rescue_delta_power   = widgets.FloatText(value=0.03, description='Δ power Q')
        self.art_rescue_relax_spikes  = widgets.Checkbox(value=False, description='Relax spikes')
        self.art_rescue_steps_max     = widgets.IntText(value=3, description='Steps max')

        # flank width widget (Hz)
        self.wave_flank_hz = widgets.FloatText(
            value=4.0, description='Flank Hz:', layout=widgets.Layout(width='120px'))

        # Wavelet gating controls & overlaps
        self.wave_pow_pct = widgets.FloatSlider(value=90, min=0, max=100, step=1, description='Wave pow %ile:')
        self.wave_prom_pct = widgets.FloatSlider(value=90, min=0, max=100, step=1, description='Prom %ile:')
        self.wave_overlap_pct = widgets.FloatSlider(value=50, min=0, max=100, step=1, description='Wave overlap %:')
        self.pacf_overlap_pct = widgets.FloatSlider(value=50, min=0, max=100, step=1, description='pACF overlap %:')
        self.env_overlap_pct = widgets.FloatSlider(value=50, min=0, max=100, step=1, description='Env overlap %:')

        # channel / epoch
        self.channel_slider = widgets.IntSlider(description='Channel', min=0, max=0, value=0)
        self.epoch_slider = widgets.IntSlider(description='Epoch', min=0, max=0, value=0)

        # pACF window
        self.time_start_slider = widgets.FloatSlider(description='Start (s)', min=0.0, step=0.01, value=0.0)
        self.time_end_slider = widgets.FloatSlider(description='End   (s)', min=0.0, step=0.01, value=1.0)

        # pACF params
        self.pacf_win_slider = widgets.FloatSlider(description='PACF Win', min=0.01, max=2.0, step=0.01, value=0.1)
        self.pacf_step_slider = widgets.FloatSlider(description='PACF Step', min=0.01, max=1.0, step=0.01, value=0.1)
        self.min_lag_slider = widgets.FloatSlider(description='Min Lag', min=0.0, max=1.0, step=0.001, value=0.05)
        self.max_lag_slider = widgets.FloatSlider(description='Max Lag', min=0.0, max=1.0, step=0.001, value=1.0)
        self.n_lag_slider = widgets.IntSlider(description='# Lags', min=1, max=200, step=1, value=150)
        self.stab_th_slider = widgets.FloatSlider(description='Stab Th', min=0.0, max=1.0, step=0.01, value=0.85)

        # bandpass & envelope
        self.low_freq = widgets.FloatText(value=10.0, description='Low Hz:')
        self.high_freq = widgets.FloatText(value=18.0, description='High Hz:')
        self.filter_order = widgets.IntText(value=3, description='Filt Order:')
        self.min_cycles = widgets.IntText(value=3, description='Min Cycles:')
        self.threshold = widgets.FloatSlider(value=85, min=0, max=100, step=1, description='Env %ile:')

        # artifact detection widgets (local z-score)
        # artifact detection widgets (local z-score)
        self.spike_z_thr = widgets.FloatText(
            value=5.0, description='Spike Z:', layout=widgets.Layout(width='120px'))
        self.spike_win_s = widgets.FloatText(
            value=0.25, description='Spike Win (s):', layout=widgets.Layout(width='150px'))
        # master artifact toggle
        self.artifact_enable = widgets.Checkbox(value=True, description='Use artifact rejection')
        # cycle validation CV thresholds (post-hoc QC of detected bursts)
        self.period_cv = widgets.FloatText(value=0.9, description='Period CV ≤')
        self.amp_cv = widgets.FloatText(value=0.9, description='Amp CV ≤')
        # legacy symmetry fields kept for UI continuity (not enforced here)
        self.sym_low = widgets.FloatText(value=0.5, description='Sym ≥')
        self.sym_high = widgets.FloatText(value=0.6, description='Sym ≤')

        # trimmed bycycle-style thresholds
        self.amp_frac_thr = widgets.FloatText(value=0.0, description='AmpFrac ≥')
        self.amp_cons_thr = widgets.FloatText(value=0.7, description='AmpCons ≥')
        self.per_cons_thr = widgets.FloatText(value=0.9, description='PerCons ≥')
        self.mono_thr = widgets.FloatText(value=0.5, description='Mono ≥')

        # auto consistency controls
        self.auto_consistency = widgets.Checkbox(value=True, description='Auto consistency (band-based)')
        self.auto_info = widgets.HTML(value='')

        # layout
        self.out = widgets.Output()
        gui = widgets.VBox([
            widgets.HBox([self.path_text, self.load_btn, self.proc_btn]),
            widgets.HBox([self.channel_slider, self.epoch_slider, self.wave_flank_hz]),
            widgets.HBox([self.time_start_slider, self.time_end_slider]),
            widgets.Label("→ pACF params auto‑computed from Low/High Hz ←"),
            widgets.HBox([self.pacf_win_slider, self.pacf_step_slider]),
            widgets.HBox([self.min_lag_slider, self.max_lag_slider, self.n_lag_slider]),
            self.stab_th_slider,
            widgets.HBox([self.low_freq, self.high_freq, self.filter_order]),
            widgets.Label('Wavelet gating & overlaps:'),
            widgets.HBox([self.wave_flank_hz, self.wave_pow_pct, self.wave_prom_pct]),
            widgets.HBox([self.wave_overlap_pct, self.pacf_overlap_pct, self.env_overlap_pct]),
            widgets.HBox([self.min_cycles, self.threshold, self.spike_z_thr, self.spike_win_s, self.artifact_enable]),
            widgets.Label("Cycle‑validation settings (post‑hoc on detected bursts):"),
            widgets.HBox([self.period_cv, self.amp_cv, self.sym_low, self.sym_high]),
            widgets.Label("Cycle detector thresholds (trimmed Bycycle style):"),
            widgets.HBox([self.amp_frac_thr, self.amp_cons_thr, self.per_cons_thr, self.mono_thr]),
            widgets.Label("PSD 1/f peak gate & axes:"),
            widgets.HBox([self.onef_gate, self.onef_zthr, self.onef_fit_lo, self.onef_fit_hi]),
            widgets.HBox([self.psd_auto_x, self.psd_xmin, self.psd_xmax, self.psd_auto_y, self.psd_ymin, self.psd_ymax]),
            widgets.Label("Artifact Cleaning (pre-filter on raw epochs):"),
            widgets.HBox([self.art_mode, self.art_preset, self.art_apply_btn, self.art_summary_label]),
            widgets.HBox([
                widgets.VBox([
                    widgets.Label("Soft Z (robust):"),
                    self.art_amp_soft_z, self.art_dor_soft_z, self.art_dor_refractory_s
                ], layout=widgets.Layout(width='260px')),
                widgets.VBox([
                    widgets.Label("Hard veto:"),
                    self.art_amp_veto_z, self.art_amp_veto_min_ms,
                    self.art_dor_veto_z, self.art_dor_veto_min_ms
                ], layout=widgets.Layout(width='260px')),
                widgets.VBox([
                    widgets.Label("Power metrics:"),
                    self.art_plow_lo, self.art_plow_hi,
                    self.art_fmin, self.art_fmax, self.art_intercept_hz
                ], layout=widgets.Layout(width='300px'))
            ]),
            widgets.HBox([
                widgets.VBox([
                    widgets.Label("Quantile mode (Qs):"),
                    self.art_q_amp, self.art_q_dor, self.art_q_plow, self.art_q_inter
                ], layout=widgets.Layout(width='260px')),
                widgets.VBox([
                    widgets.Label("Fixed mode (thr):"),
                    self.art_fixed_amp_bad_frac, self.art_fixed_dor_eps,
                    self.art_fixed_plow_db_thr, self.art_fixed_inter_thr
                ], layout=widgets.Layout(width='320px')),
                widgets.VBox([
                    widgets.Label("Rescue:"),
                    self.art_rescue_enabled, self.art_rescue_min_keep_frac, self.art_rescue_cap,
                    self.art_rescue_delta_spike, self.art_rescue_delta_power,
                    self.art_rescue_relax_spikes, self.art_rescue_steps_max
                ], layout=widgets.Layout(width='320px'))
            ]),
            widgets.Label('Consistency thresholds (auto ↔ manual):'),
            widgets.HBox([self.auto_consistency]),
            self.auto_info,
            self.out
        ])
        display(gui)

        # callbacks
        self.load_btn.on_click(self._on_load)
        self.proc_btn.on_click(self._on_process_all)

        # artifact UI reacts
        for w in [
            self.art_mode, self.art_preset,
            self.art_amp_soft_z, self.art_dor_soft_z, self.art_dor_refractory_s,
            self.art_amp_veto_z, self.art_amp_veto_min_ms, self.art_dor_veto_z, self.art_dor_veto_min_ms,
            self.art_plow_lo, self.art_plow_hi, self.art_fmin, self.art_fmax, self.art_intercept_hz,
            self.art_q_amp, self.art_q_dor, self.art_q_plow, self.art_q_inter,
            self.art_fixed_amp_bad_frac, self.art_fixed_dor_eps, self.art_fixed_plow_db_thr, self.art_fixed_inter_thr,
            self.art_rescue_enabled, self.art_rescue_min_keep_frac, self.art_rescue_cap,
            self.art_rescue_delta_spike, self.art_rescue_delta_power, self.art_rescue_relax_spikes, self.art_rescue_steps_max,
            self.channel_slider,
            self.wave_pow_pct, self.wave_prom_pct, self.wave_overlap_pct, self.pacf_overlap_pct, self.env_overlap_pct,
            self.artifact_enable
        ]:
            w.observe(lambda *_: self._redraw(), names='value')
        self.art_apply_btn.on_click(lambda *_: (self._apply_artifact_cleaning(), self._redraw()))
        for w in [self.onef_gate, self.onef_zthr, self.onef_fit_lo, self.onef_fit_hi,
                  self.psd_auto_x, self.psd_xmin, self.psd_xmax,
                  self.psd_auto_y, self.psd_ymin, self.psd_ymax]:
            w.observe(lambda *_: self._redraw(), names='value')
        for w in [self.onef_gate, self.onef_zthr, self.onef_fit_lo, self.onef_fit_hi]:
            w.observe(lambda *_: self._redraw(), names='value')


        for w in [
            self.channel_slider, self.epoch_slider,
            self.wave_flank_hz,
            self.time_start_slider, self.time_end_slider,
            self.pacf_win_slider, self.pacf_step_slider,
            self.min_lag_slider, self.max_lag_slider,
            self.n_lag_slider, self.stab_th_slider,
            self.filter_order, self.min_cycles, self.threshold,
            self.spike_z_thr, self.spike_win_s,
            self.period_cv, self.amp_cv, self.sym_low, self.sym_high,
            self.low_freq, self.high_freq,
            self.amp_frac_thr, self.amp_cons_thr, self.per_cons_thr, self.mono_thr
        ]:
            w.observe(lambda *_: (self._apply_artifact_cleaning(), self._redraw()), names='value')



    def set_artifact_rejection(self, enabled: bool):
        """Enable/disable artifact rejection from code (also updates the checkbox)."""
        try:
            self.artifact_enable.value = bool(enabled)
        except Exception:
            pass

    # ----- load -----
    def _on_load(self, _):
        with self.out:
            clear_output()
            path = self.path_text.value.strip()
            if not os.path.isfile(path) or not path.lower().endswith('.fif'):
                print("Invalid .fif path.")
                return
            try:
                self.epochs = mne.read_epochs(path, preload=True, verbose=False)
            except Exception as e:
                print(f"Error loading: {e}")
                return

            self._loaded_path = path
            self.data = self.epochs.get_data()
            self.fs = self.epochs.info['sfreq']
            n_ep, n_ch, n_t = self.data.shape
            dur = n_t / self.fs

            self.channel_slider.max = n_ch - 1
            self.epoch_slider.max = n_ep - 1
            self.time_start_slider.max = dur
            self.time_end_slider.max = dur
            self.time_end_slider.value = dur

            self._apply_artifact_cleaning()

            self._redraw()

    def _update_pacf_params(self, *_):
        lf, hf = self.low_freq.value, self.high_freq.value
        if lf <= 0 or hf <= lf or not hasattr(self, 'fs'):
            return

    def _compute_auto_consistency(self):
        center = (float(self.low_freq.value) + float(self.high_freq.value)) / 2.0
        auto_amp = _amp_cons_thr_piecewise(center)
        auto_per = _per_cons_thr_delta_hz(center, delta_hz=1.0)
        return center, auto_amp, auto_per

    def _apply_auto_consistency_if_enabled(self, *_):
        try:
            center, a_thr, p_thr = self._compute_auto_consistency()
        except Exception:
            return
        if getattr(self, 'auto_consistency', None) is not None and self.auto_consistency.value:
            # Set widget values and lock them
            try:
                self.amp_cons_thr.value = round(float(a_thr), 3)
                self.per_cons_thr.value = round(float(p_thr), 3)
            except Exception:
                pass
            try:
                self.amp_cons_thr.disabled = True
                self.per_cons_thr.disabled = True
            except Exception:
                pass
        else:
            # Manual mode
            try:
                self.amp_cons_thr.disabled = False
                self.per_cons_thr.disabled = False
            except Exception:
                pass

        # Update info text
        try:
            band = f"{float(self.low_freq.value):.1f}-{float(self.high_freq.value):.1f} Hz"
            mode = 'AUTO' if self.auto_consistency.value else 'MANUAL'
            self.auto_info.value = (
                f"<span style='font-size:12px;color:#444;'>"
                f"Mode: <b>{mode}</b> — Band {band} → "
                f"AmpCons={float(a_thr):.3f}, PerCons={float(p_thr):.3f} (±1 Hz). "
                "Toggle off to edit manually.</span>"
            )
        except Exception:
            pass

        win = 3.0 / lf
        step = win / 4.0
        mn = 1.0 / hf
        mx = 1.0 / lf
        nl = 40
        self.pacf_win_slider.value = np.clip(win, self.pacf_win_slider.min, self.pacf_win_slider.max)
        self.pacf_step_slider.value = np.clip(step, self.pacf_step_slider.min, self.pacf_step_slider.max)
        self.min_lag_slider.value = np.clip(mn, self.min_lag_slider.min, self.min_lag_slider.max)
        self.max_lag_slider.value = np.clip(mx, self.max_lag_slider.min, self.max_lag_slider.max)
        self.n_lag_slider.value = np.clip(nl, self.n_lag_slider.min, self.n_lag_slider.max)

    def _reset_to_raw_if_needed(self):
        """If artifact rejection is OFF, restore the original epochs/data so no prior exclusions persist."""
        try:
            if hasattr(self, 'artifact_enable') and (not self.artifact_enable.value):
                if getattr(self, '_epochs_raw', None) is not None:
                    self.epochs = self._epochs_raw.copy()
                    self.data = self.epochs.get_data()
                    n_ep = self.data.shape[0]
                    self._epoch_map = np.ones(n_ep, dtype=bool)
        except Exception:
            pass

    # ----- filtering -----
    def bandpass(self, sig):
        nyq = 0.5 * self.fs
        low, high = self.low_freq.value / nyq, self.high_freq.value / nyq
        b, a = butter(self.filter_order.value, [low, high], btype='band')
        return filtfilt(b, a, sig)

    # ----- envelope bursts (kept for diagnostics & agreement plots) -----
    def detect_bursts_envelope(self, envelope_amp):
        thr = np.percentile(envelope_amp, self.threshold.value)
        mask = envelope_amp > thr
        bursts = np.zeros_like(mask, bool)
        center = (self.low_freq.value + self.high_freq.value) / 2
        min_dur = self.min_cycles.value / center
        start, segs = None, []
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if (i - start) / self.fs >= min_dur:
                    bursts[start:i] = True
                    segs.append((start, i))
                start = None
        if start is not None and (len(mask) - start) / self.fs >= min_dur:
            bursts[start:] = True
            segs.append((start, len(mask)))
        return bursts, thr, segs

    # ----- cycle-based burst detection (trimmed bycycle) -----
    def validate_cycles(self, filt):
        df = detect_bursts_cycles(
            filt, self.fs,
            fmin=float(self.low_freq.value),
            fmax=float(self.high_freq.value),
            amp_frac=float(self.amp_frac_thr.value),
            amp_cons_thr=float(self.amp_cons_thr.value),
            per_cons_thr=float(self.per_cons_thr.value),
            mono_thr=float(self.mono_thr.value),
            min_cycles=int(self.min_cycles.value)
        )
        self._cycle_df = df  # stored for marker extraction

        # Build segment list from contiguous valid cycles (burst_id >= 0)
        segs = []
        for bid in sorted(df.burst_id.unique()):
            if bid < 0:
                continue
            sub = df[df.burst_id == bid]
            s = int(sub.trough_start.min())
            e = int(sub.trough_end.max())
            # post-hoc QC using CV thresholds
            per_cv = sub['period'].std() / sub['period'].mean() if len(sub) > 1 else np.nan
            amp_cv = sub['amp'].std() / sub['amp'].mean() if len(sub) > 1 else np.nan
            sym_min = np.nan; sym_max = np.nan  # kept for compatibility
            sym_ok = True  # symmetry not enforced in trimmed method
            valid = ( (np.isnan(per_cv) or per_cv <= float(self.period_cv.value))
                      and (np.isnan(amp_cv) or amp_cv <= float(self.amp_cv.value))
                      and sym_ok )
            segs.append({'start': s, 'end': e, 'valid': valid,
                         'per_cv': per_cv, 'amp_cv': amp_cv,
                         'sym_min': sym_min, 'sym_max': sym_max})
        return segs

    # ----- cycle markers from trimmed df -----
    def _extract_cycle_markers(self, filt):
        if not hasattr(self, '_cycle_df') or self._cycle_df is None or self._cycle_df.empty:
            return None, None
        df = self._cycle_df
        n = len(df)
        rise50 = np.full(n, np.nan)
        fall50 = np.full(n, np.nan)
        trough_samps = df['trough_start'].to_numpy(dtype=float)
        peak_samps = df['peak'].to_numpy(dtype=float)
        next_trough_samps = df['trough_end'].to_numpy(dtype=float)
        for i in range(n):
            rise50[i] = _compute_half_amp_points(filt, trough_samps[i], peak_samps[i])
            fall50[i] = _compute_half_amp_fall(filt, peak_samps[i], next_trough_samps[i])
        valid_cycle_mask = df['is_valid'].astype(bool).to_numpy()
        markers = dict(
            trough=trough_samps.astype(float),
            peak=peak_samps.astype(float),
            trough_next=next_trough_samps.astype(float),
            rise50=rise50,
            fall50=fall50
        )
        return markers, valid_cycle_mask

    # ----- local z-score artifact detection -----
    def _artifact_mask_local_z(self, sig):
        # Toggle: when off, skip artifacting and return all-false mask
        if hasattr(self, 'artifact_enable') and (not bool(self.artifact_enable.value)):
            mask = np.zeros_like(sig, dtype=bool)
            z = np.zeros_like(sig, dtype=float)
            sd = np.ones_like(sig, dtype=float)
            return mask, z, sd

        fs = self.fs
        win_s = float(self.spike_win_s.value)
        if not np.isfinite(win_s) or win_s <= 0:
            win_s = 0.25
        win = max(1, int(round(fs * win_s)))
        if win > sig.size:
            win = sig.size

        kern = np.ones(win, dtype=float)
        mu = np.convolve(sig, kern, mode='same') / win
        sig2 = sig.astype(float) ** 2
        mu2 = np.convolve(sig2, kern, mode='same') / win
        var = mu2 - mu ** 2
        var[var < 0] = 0.0
        sd = np.sqrt(var)
        sd[sd < 1e-12] = 1e-12

        z = (sig - mu) / sd
        thr = float(self.spike_z_thr.value)
        if not np.isfinite(thr) or thr <= 0:
            thr = 5.0
        mask_hits = np.abs(z) > thr

        # pad ±pacf window
        pad_samps = int(round(self.pacf_win_slider.value * fs))
        if pad_samps > 0 and mask_hits.any():
            mask = mask_hits.copy()
            hit_idxs = np.where(mask_hits)[0]
            for ix in hit_idxs:
                lo = max(0, ix - pad_samps)
                hi = min(sig.size, ix + pad_samps)
                mask[lo:hi] = True
        else:
            mask = mask_hits

        return mask, z, sd

    # ----- artifact pre-cleaning (raw epochs) -----
    def _apply_artifact_cleaning(self):
        if not hasattr(self, 'epochs'):
            return
        ch = int(self.channel_slider.value)
        X = self.epochs.get_data(copy=True)[:, ch, :]
        fs = float(self.epochs.info['sfreq'])
        cfg = dict(
            art_amp_soft_z=float(self.art_amp_soft_z.value),
            art_dor_soft_z=float(self.art_dor_soft_z.value),
            art_dor_refractory_s=float(self.art_dor_refractory_s.value),
            art_plow_lo=float(self.art_plow_lo.value),
            art_plow_hi=float(self.art_plow_hi.value),
            art_fmin=float(self.art_fmin.value),
            art_fmax=float(self.art_fmax.value),
            art_intercept_hz=float(self.art_intercept_hz.value),
            art_amp_veto_z=float(self.art_amp_veto_z.value),
            art_amp_veto_min_ms=float(self.art_amp_veto_min_ms.value),
            art_dor_veto_z=float(self.art_dor_veto_z.value),
            art_dor_veto_min_ms=float(self.art_dor_veto_min_ms.value),
            art_q_amp=float(self.art_q_amp.value),
            art_q_dor=float(self.art_q_dor.value),
            art_q_plow=float(self.art_q_plow.value),
            art_q_inter=float(self.art_q_inter.value),
            art_rescue_enabled=bool(self.art_rescue_enabled.value),
            art_rescue_min_keep_frac=float(self.art_rescue_min_keep_frac.value),
            art_rescue_cap=float(self.art_rescue_cap.value),
            art_rescue_delta_spike=float(self.art_rescue_delta_spike.value),
            art_rescue_delta_power=float(self.art_rescue_delta_power.value),
            art_rescue_relax_spikes=bool(self.art_rescue_relax_spikes.value),
            art_rescue_steps_max=int(self.art_rescue_steps_max.value),
            art_fixed_amp_bad_frac=float(self.art_fixed_amp_bad_frac.value),
            art_fixed_dor_eps=float(self.art_fixed_dor_eps.value),
            art_fixed_plow_db_thr=float(self.art_fixed_plow_db_thr.value) if not np.isnan(self.art_fixed_plow_db_thr.value) else float('nan'),
            art_fixed_inter_thr=float(self.art_fixed_inter_thr.value) if not np.isnan(self.art_fixed_inter_thr.value) else float('nan'),
        )
        metrics = _compute_artifact_metrics_per_epoch(X, fs, cfg)
        strategy = self.art_mode.value
        preset   = self.art_preset.value
        keep_mask, details = _artifact_decide_keep(metrics, strategy, preset, cfg)
        self._artifact_keep_mask = keep_mask
        self._artifact_details = details
        kept = int(keep_mask.sum()); total = len(keep_mask)
        pct = (kept / total * 100.0) if total else 0.0
        self.art_summary_label.value = f"Kept {kept}/{total} epochs ({pct:.1f}%) | thr_a={details.get('thr_a',np.nan):.3g}, thr_d={details.get('thr_d',np.nan):.3g}, thr_p={details.get('thr_p',np.nan):.3g}, thr_i={details.get('thr_i',np.nan):.3g}"
        self._epoch_map = np.where(keep_mask)[0]
        if self._epoch_map.size == 0:
            self.epoch_slider.max = 0
            self.epoch_slider.value = 0
        else:
            self.epoch_slider.max = self._epoch_map.size - 1
            self.epoch_slider.value = min(self.epoch_slider.value, self.epoch_slider.max)

    def _visible_to_original_epoch(self, ep_vis: int) -> int:
        if hasattr(self, '_epoch_map') and self._epoch_map is not None and self._epoch_map.size:
            return int(self._epoch_map[ep_vis])
        return int(ep_vis)

        fs = self.fs
        win_s = float(self.spike_win_s.value)
        if not np.isfinite(win_s) or win_s <= 0:
            win_s = 0.25
        win = max(1, int(round(fs * win_s)))
        if win > sig.size:
            win = sig.size

        kern = np.ones(win, dtype=float)
        mu = np.convolve(sig, kern, mode='same') / win
        sig2 = sig.astype(float) ** 2
        mu2 = np.convolve(sig2, kern, mode='same') / win
        var = mu2 - mu ** 2
        var[var < 0] = 0.0
        sd = np.sqrt(var)
        sd[sd < 1e-12] = 1e-12

        z = (sig - mu) / sd
        thr = float(self.spike_z_thr.value)
        if not np.isfinite(thr) or thr <= 0:
            thr = 5.0
        mask_hits = np.abs(z) > thr

        # pad ±pacf window
        pad_samps = int(round(self.pacf_win_slider.value * fs))
        if pad_samps > 0 and mask_hits.any():
            mask = mask_hits.copy()
            hit_idxs = np.where(mask_hits)[0]
            for ix in hit_idxs:
                lo = max(0, ix - pad_samps)
                hi = min(sig.size, ix + pad_samps)
                mask[lo:hi] = True
        else:
            mask = mask_hits

        return mask, z, sd

    # -----------------------------------------------------------------
    # Per-epoch analysis (no plotting) -> list of rows
    # -----------------------------------------------------------------
    def _analyze_epoch(self, ep, ch):
        """
        Run detection & compute diagnostics.
        Returns:
            rows : list of dicts (segment-level)
            agree_metrics : dict (J, P, R, F1, SegMed) for Env vs pACF
        """
        sig = self.data[ep, ch, :]
        fs = self.fs
        times = np.arange(sig.size) / fs

        # bandpass & envelope bursts (envelope kept for diagnostics only)
        filt = self.bandpass(sig)
        envelope_amp = np.abs(hilbert(filt))
        env_mask, thr_env, env_segs = self.detect_bursts_envelope(envelope_amp)
        env_above = envelope_amp > thr_env
        env_above = envelope_amp > thr_env

        # pACF over GUI window
        dur = sig.size / fs
        t0 = max(0.0, min(self.time_start_slider.value, dur))
        t1 = max(t0 + 1/fs, min(self.time_end_slider.value, dur))
        seg_mask = (times >= t0) & (times <= t1)
        seg_f = filt[seg_mask][np.newaxis, :] / 1e3
        pt_rel, stab_all, _ = compute_pacf_stability_over_time(
            seg_f, fs,
            self.pacf_win_slider.value, self.pacf_step_slider.value,
            self.min_lag_slider.value, self.max_lag_slider.value,
            self.n_lag_slider.value
        )
        halfw = self.pacf_win_slider.value / 2
        vidx = np.where((pt_rel >= halfw) & (pt_rel <= (t1 - t0) - halfw))[0]
        pt_rel = pt_rel[vidx]
        stab = stab_all[vidx, 0]
        pacf_t = pt_rel + t0
        pacf_mask = stab > self.stab_th_slider.value

        # map pACF mask to full-length
        full_pacf_mask = np.zeros_like(sig, bool)
        edges = np.diff(pacf_mask.astype(int))
        starts = pacf_t[np.where(edges == 1)[0] + 1]
        ends = pacf_t[np.where(edges == -1)[0] + 1]
        if pacf_mask.size and pacf_mask[0]:
            starts = np.insert(starts, 0, pacf_t[0])
        if pacf_mask.size and pacf_mask[-1]:
            ends = np.append(ends, pacf_t[-1])
        for s, e in zip(starts, ends):
            i0 = max(0, int(np.floor(s * fs)))
            i1 = min(sig.size, int(np.ceil(e * fs)))
            full_pacf_mask[i0:i1] = True

        # agreement metrics (Env vs pACF) on interior of window
        comp = (times >= t0 + halfw) & (times <= t1 - halfw)
        E = env_mask[comp]
        P = full_pacf_mask[comp]
        nE = E.sum(); nP = P.sum()
        nI = np.logical_and(E, P).sum(); nU = np.logical_or(E, P).sum()
        jacc = nI / nU if nU else np.nan
        prec = nI / nE if nE else np.nan
        rec = nI / nP if nP else np.nan
        f1 = (2 * prec * rec) / (prec + rec) if (prec and rec) else np.nan
        seg_support = []
        for s, e in env_segs:
            if e > s:
                seg_support.append(full_pacf_mask[s:e].sum() / (e - s))
        seg_med = np.median(seg_support) if seg_support else np.nan
        agree = dict(jacc=jacc, prec=prec, rec=rec, f1=f1, seg_med=seg_med)

        # cycle-based bursts (trimmed Bycycle approach)
        metrics = self.validate_cycles(filt)

        # wavelet extended (raw)
        lf = self.low_freq.value
        hf = self.high_freq.value
        flank = max(0.0, float(self.wave_flank_hz.value))
        nyq = fs / 2.0
        fmin_ext = max(0.0, lf - flank)
        fmax_ext = min(nyq * 0.999, hf + flank)
        span = (hf - lf) + 2 * flank if flank > 0 else (hf - lf)
        span = max(span, 1e-6)
        n_freq_ext = max(30, int(30 * span / max(hf - lf, 1e-6)))
        freqs_ext = np.linspace(fmin_ext, fmax_ext, n_freq_ext)
        power_ext_raw = mne.time_frequency.tfr_array_morlet(
            sig[np.newaxis, np.newaxis, :], sfreq=fs,
            freqs=freqs_ext, n_cycles=freqs_ext / 2, output='power'
        )[0, 0]

        # wave gating
        band_mask = (freqs_ext >= lf) & (freqs_ext <= hf)
        if np.any(band_mask):
            inband_pow_tc = power_ext_raw[band_mask].mean(axis=0)
            wave_th = np.percentile(inband_pow_tc, float(self.wave_pow_pct.value))
            wave_mask_abs = inband_pow_tc > wave_th
        else:
            pow_pct = float(getattr(self, 'wave_pow_pct', type('o',(object,),{'value':90.0})()).value)
            wave_th = np.percentile(power_ext_raw, pow_pct)
            wave_mask_abs = np.any(power_ext_raw > wave_th, axis=0)

        prom_ratio = wavelet_peak_prominence(
            power_ext_raw, freqs_ext, lf, hf,
            flank_hz=flank if flank > 0 else (hf - lf)
        )
        if np.all(np.isnan(prom_ratio)):
            prom_mask = np.ones_like(wave_mask_abs, bool)
        else:
            prom_th = np.nanpercentile(prom_ratio, float(self.wave_prom_pct.value))
            prom_mask = prom_ratio > prom_th
        wave_mask = wave_mask_abs & prom_mask

        # artifact (local z)
        artifact_mask, z_scores, local_sd = self._artifact_mask_local_z(sig)

        # segment rows (built from cycle-based metrics)
        # --- PSD 1/f gate (epoch-level) ---
        f_psd, P_db, Pfit_db, f_fit, z_resid = psd_aperiodic_and_peaks(sig, fs, fmin=float(self.onef_fit_lo.value), fmax=float(self.onef_fit_hi.value), nperseg_s=2.0)
        lf = float(self.low_freq.value); hf = float(self.high_freq.value)
        mband = (f_fit >= lf) & (f_fit <= hf)
        onef_z_inband = float(np.nanmax(z_resid[mband])) if np.any(mband) else np.nan
        onef_pass = bool(onef_z_inband >= float(self.onef_zthr.value)) if np.isfinite(onef_z_inband) else False

        rows = []
        for seg_idx, m in enumerate(metrics):
            s0, s1 = m['start'], m['end']
            dur_s = (s1 - s0) / fs
            valid_cycles = bool(m['valid'])
            seg_len = s1 - s0
            coverage = (full_pacf_mask[s0:s1].sum() / seg_len) if seg_len > 0 else 0.0
            pacf_ok = coverage >= (float(self.pacf_overlap_pct.value)/100.0)
            wave_frac = float(wave_mask[s0:s1].mean()) if seg_len > 0 else 0.0
            power_ok = (wave_frac >= (float(self.wave_overlap_pct.value)/100.0))
            artifact = bool(artifact_mask[s0:s1].any())
            # Envelope ≥ threshold for ≥50% of the segment
            seg_env_frac = float(env_above[s0:s1].mean()) if seg_len > 0 else 0.0
            env_ok = seg_env_frac >= (float(self.env_overlap_pct.value)/100.0)
            # 1/f gate applies epoch-level band peak requirement if enabled
            onef_ok = (onef_pass or (not self.onef_gate.value))
            status = 'burst' if (valid_cycles and pacf_ok and power_ok and env_ok and onef_ok and not artifact) else 'fail'
            rows.append(dict(
                Channel=ch,
                Epoch=ep,
                Seg=seg_idx,
                Start_s=s0 / fs,
                End_s=s1 / fs,
                Duration_s=dur_s,
                Period_CV=m['per_cv'],
                Amp_CV=m['amp_cv'],
                Valid_Cycles=valid_cycles,
                PACF_OK=pacf_ok,
                Power_OK=power_ok,
                Artifact=artifact,
                Env_Frac=seg_env_frac,
                PACF_Frac=coverage,
                Wave_Frac=wave_frac,
                Env_OK=env_ok,
                OneF_Z=onef_z_inband,
                OneF_OK=onef_ok,
                Status=status
            ))
        return rows, agree

    # -----------------------------------------------------------------
    # Batch process all epochs -> Excel
    # -----------------------------------------------------------------
    def _on_process_all(self, _):
        import os
        import numpy as np
        import pandas as pd
        from IPython.display import clear_output
        
        def _safe_val(obj, cast=float, default=np.nan):
            try:
                val = getattr(obj, 'value', obj)
                return cast(val)
            except Exception:
                return default
        
        if not hasattr(self, 'data'):
            with self.out:
                clear_output()
                print("No data loaded.")
            return
        
        ch = int(self.channel_slider.value)
        
        # Respect artifact-kept epochs if present
        n_ep_total = int(self.data.shape[0])
        ep_indices = getattr(self, '_epoch_map', np.arange(n_ep_total, dtype=int))
        n_ep = int(len(ep_indices))
        
        all_rows = []
        agrees = []
        with self.out:
            clear_output()
            print(f"Processing channel {ch} across {n_ep_eff} epochs...")
        
        for ep in ep_indices:
            ep = int(ep)
            r, ag = self._analyze_epoch(ep, ch)
            all_rows.extend(r)
            agrees.append((ep, ag))
        
        if not all_rows:
            with self.out:
                clear_output()
                print("No segments found across epochs.")
            return
        
        df = pd.DataFrame(all_rows)
        
        # Summary counts
        n_seg = len(df)
        n_burst = int((df['Status'] == 'burst').sum()) if 'Status' in df.columns else 0
        n_fail  = int(n_seg - n_burst)
        pct_burst = (n_burst / n_seg * 100.0) if n_seg else 0.0
        pct_fail  = (n_fail  / n_seg * 100.0) if n_seg else 0.0
        
        # Artifact info saved elsewhere in the viewer (if available)
        art_details = getattr(self, '_artifact_details', {})
        kept_mask   = getattr(self, '_artifact_keep_mask', None)
        
        # Build settings summary safely (no undefined cfg)
        summ_dict = dict(
            File=os.path.basename(getattr(self, '_loaded_path', 'UNKNOWN')),
            Channel=ch,
            Fs=_safe_val(getattr(self, 'fs', np.nan)),
            Low_Hz=_safe_val(self.low_freq, float('nan')),
            High_Hz=_safe_val(self.high_freq, float('nan')),
            Filt_Order=int(getattr(self.filter_order, 'value', 3)) if hasattr(self, 'filter_order') else np.nan,
            Env_Percentile=_safe_val(self.threshold, float('nan')),
            Min_Cycles=int(getattr(self.min_cycles, 'value', 3)) if hasattr(self, 'min_cycles') else np.nan,
            Period_CV_Thresh=_safe_val(self.period_cv, float('nan')),
            Amp_CV_Thresh=_safe_val(self.amp_cv, float('nan')),
            AmpFrac=_safe_val(self.amp_frac_thr, float('nan')),
            AmpCons=_safe_val(self.amp_cons_thr, float('nan')),
            PerCons=_safe_val(self.per_cons_thr, float('nan')),
            Mono=_safe_val(self.mono_thr, float('nan')),
            PACF_Win_s=_safe_val(self.pacf_win_slider, float('nan')),
            PACF_Step_s=_safe_val(self.pacf_step_slider, float('nan')),
            PACF_Min_Lag_s=_safe_val(self.min_lag_slider, float('nan')),
            PACF_Max_Lag_s=_safe_val(self.max_lag_slider, float('nan')),
            PACF_n_Lags=int(getattr(self.n_lag_slider, 'value', 0)) if hasattr(self, 'n_lag_slider') else np.nan,
            PACF_Stab_Th=_safe_val(self.stab_th_slider, float('nan')),
            Wave_Flank_Hz=_safe_val(self.wave_flank_hz, float('nan')),
            Spike_Z=_safe_val(self.spike_z_thr, float('nan')),
            Spike_Win_s=_safe_val(self.spike_win_s, float('nan')),
            Total_Epochs=n_ep,
            Total_Segments=n_seg,
            Bursts=n_burst,
            Fails=n_fail,
            Burst_Pct=pct_burst,
            Fail_Pct=pct_fail
        )
        
        # Optional fields (if your module defines these widgets/attrs)
        # Overlap+wavelet
        for opt_key, attr in [
            ('Env_Overlap_Pct', 'env_overlap_pct'),
            ('PACF_Overlap_Pct', 'pacf_overlap_pct'),
            ('Wave_Overlap_Pct', 'wave_overlap_pct'),
            ('Wave_Pow_Pct', 'wave_pow_pct'),
            ('Wave_Prom_Pct', 'wave_prom_pct'),
        ]:
            if hasattr(self, attr):
                summ_dict[opt_key] = _safe_val(getattr(self, attr), float('nan'))
        
        # 1/f gate settings
        for opt_key, attr, caster in [
            ('OneF_Gate', 'onef_gate', bool),
            ('OneF_Zthr', 'onef_zthr', float),
            ('OneF_FitLo', 'onef_fit_lo', float),
            ('OneF_FitHi', 'onef_fit_hi', float),
        ]:
            if hasattr(self, attr):
                try:
                    val = getattr(getattr(self, attr), 'value', getattr(self, attr))
                    summ_dict[opt_key] = caster(val)
                except Exception:
                    pass
        
        # Artifact cleaner controls and thresholds actually used
        for opt_key, attr in [
            ('Art_Mode', 'art_mode'),
            ('Art_Preset', 'art_preset'),
            ('Art_Q_amp', 'art_q_amp'),
            ('Art_Q_dor', 'art_q_dor'),
            ('Art_Q_plow', 'art_q_plow'),
            ('Art_Q_inter', 'art_q_inter'),
        ]:
            if hasattr(self, attr):
                summ_dict[opt_key] = _safe_val(getattr(self, attr), float, np.nan)
        
        summ_dict.update(dict(
            Art_thr_a=float(art_details.get('thr_a', np.nan)),
            Art_thr_d=float(art_details.get('thr_d', np.nan)),
            Art_thr_p=float(art_details.get('thr_p', np.nan)),
            Art_thr_i=float(art_details.get('thr_i', np.nan)),
            Art_Kept_Epochs=int(kept_mask.sum()) if getattr(kept_mask, 'sum', None) else n_ep,
            Art_Total_Epochs=int(len(kept_mask)) if hasattr(kept_mask, '__len__') else n_ep
        ))
        
        df_summary = pd.DataFrame([summ_dict])
        
        # Agreement metrics per epoch
        agree_rows = []
        for ep, ag in agrees:
            agree_rows.append(dict(
                Epoch=ep,
                Jaccard=ag.get('jacc', np.nan),
                Precision=ag.get('prec', np.nan),
                Recall=ag.get('rec', np.nan),
                F1=ag.get('f1', np.nan),
                SegMed=ag.get('seg_med', np.nan)
            ))
        df_agree = pd.DataFrame(agree_rows)
        
        # Build Excel path next to the source .fif
        base = os.path.splitext(os.path.basename(getattr(self, '_loaded_path', 'output')))[0]
        out_dir = os.path.dirname(getattr(self, '_loaded_path', '.')) or '.'
        out_name = f"{base}__ch{ch}_segments.xlsx"
        out_path = os.path.join(out_dir, out_name)
        
        # Write workbook with clear error reporting
        try:
            with pd.ExcelWriter(out_path, engine='openpyxl') as xlw:
                df.to_excel(xlw, sheet_name='data', index=False)
                df_summary.to_excel(xlw, sheet_name='summary', index=False)
                df_agree.to_excel(xlw, sheet_name='agreement', index=False)
        except Exception as e:
            with self.out:
                clear_output()
                print(f"Excel write failed: {e}")  # exposes missing-openpyxl or permission issues
            return
        
        with self.out:
            clear_output()
            print(f"Done. Wrote {len(df)} segment rows from {n_ep} epochs to:")
            print(out_path)
            print(f"Burst segments: {n_burst} ({pct_burst:.1f}%)  |  Fail segments: {n_fail} ({pct_fail:.1f}%)")

    # ----- redraw (interactive plot for current epoch) -----
    def _redraw(self):
        with self.out:
            clear_output()
            if not hasattr(self, 'data'):
                return

            ch, ep_vis = self.channel_slider.value, self.epoch_slider.value
            ep = self._visible_to_original_epoch(ep_vis)
            rows, agree = self._analyze_epoch(ep, ch)

            # stash for table generation below
            self._agree = agree

            sig = self.data[ep, ch, :]
            fs = self.fs
            times = np.arange(sig.size) / fs
            filt = self.bandpass(sig)
            envelope_amp = np.abs(hilbert(filt))
            env_mask, thr_env, env_segs = self.detect_bursts_envelope(envelope_amp)
            env_above = envelope_amp > thr_env

            # pACF bits re‑do for plotting
            dur = sig.size / fs
            t0 = max(0.0, min(self.time_start_slider.value, dur))
            t1 = max(t0 + 1/fs, min(self.time_end_slider.value, dur))
            seg_mask = (times >= t0) & (times <= t1)
            seg_f = filt[seg_mask][np.newaxis, :] / 1e3
            pt_rel, stab_all, _ = compute_pacf_stability_over_time(
                seg_f, fs,
                self.pacf_win_slider.value, self.pacf_step_slider.value,
                self.min_lag_slider.value, self.max_lag_slider.value,
                self.n_lag_slider.value
            )
            halfw = self.pacf_win_slider.value / 2
            vidx = np.where((pt_rel >= halfw) & (pt_rel <= (t1 - t0) - halfw))[0]
            pt_rel = pt_rel[vidx]
            stab = stab_all[vidx, 0]
            pacf_t = pt_rel + t0
            pacf_mask = stab > self.stab_th_slider.value
            full_pacf_mask = np.zeros_like(sig, bool)
            edges = np.diff(pacf_mask.astype(int))
            starts = pacf_t[np.where(edges == 1)[0] + 1]
            ends = pacf_t[np.where(edges == -1)[0] + 1]
            if pacf_mask.size and pacf_mask[0]:
                starts = np.insert(starts, 0, pacf_t[0])
            if pacf_mask.size and pacf_mask[-1]:
                ends = np.append(ends, pacf_t[-1])
            pacf_segs = list(zip(starts, ends))
            for s, e in pacf_segs:
                i0 = max(0, int(np.floor(s * fs)))
                i1 = min(sig.size, int(np.ceil(e * fs)))
                full_pacf_mask[i0:i1] = True

            # Intersection (for overlay panel)
            both_mask = env_mask & full_pacf_mask
            both_segs = []
            if both_mask.any():
                dm = np.diff(both_mask.astype(int))
                seg_st = np.where(dm == 1)[0] + 1
                seg_en = np.where(dm == -1)[0] + 1
                if both_mask[0]:
                    seg_st = np.insert(seg_st, 0, 0)
                if both_mask[-1]:
                    seg_en = np.append(seg_en, both_mask.size)
                both_segs = list(zip(seg_st, seg_en))
            both_trace_scaled = both_mask.astype(float) * 0.1

            # cycle metrics & markers
            metrics = self.validate_cycles(filt)
            markers, valid_cycle_mask = self._extract_cycle_markers(filt)

            # wavelet extended
            lf = self.low_freq.value
            hf = self.high_freq.value
            flank = max(0.0, float(self.wave_flank_hz.value))
            nyq = fs / 2.0
            fmin_ext = max(0.0, lf - flank)
            fmax_ext = min(nyq * 0.999, hf + flank)
            span = (hf - lf) + 2 * flank if flank > 0 else (hf - lf)
            span = max(span, 1e-6)
            n_freq_ext = max(30, int(30 * span / max(hf - lf, 1e-6)))
            freqs_ext = np.linspace(fmin_ext, fmax_ext, n_freq_ext)
            power_ext_raw = mne.time_frequency.tfr_array_morlet(
                sig[np.newaxis, np.newaxis, :], sfreq=fs,
                freqs=freqs_ext, n_cycles=freqs_ext / 2, output='power'
            )[0, 0]

            band_mask = (freqs_ext >= lf) & (freqs_ext <= hf)
            freqs_band = freqs_ext[band_mask]
            power_band_raw = power_ext_raw[band_mask, :] if np.any(band_mask) else np.empty((0, power_ext_raw.shape[1]))

            if np.any(band_mask):
                inband_pow_tc = power_ext_raw[band_mask].mean(axis=0)
                wave_th = np.percentile(inband_pow_tc, float(self.wave_pow_pct.value))
                wave_mask_abs = inband_pow_tc > wave_th
            else:
                wave_th = np.percentile(power_ext_raw, float(getattr(self, 'wave_pow_pct', type('o',(object,),{'value':90.0})()).value))
                wave_mask_abs = np.any(power_ext_raw > wave_th, axis=0)
                wave_mask_abs = np.any(power_ext_raw > wave_th, axis=0)

            prom_ratio = wavelet_peak_prominence(
                power_ext_raw, freqs_ext, lf, hf,
                flank_hz=flank if flank > 0 else (hf - lf)
            )
            if np.all(np.isnan(prom_ratio)):
                prom_th = np.nan
                prom_mask = np.ones_like(wave_mask_abs, bool)
            else:
                prom_th = np.nanpercentile(prom_ratio, float(self.wave_prom_pct.value))
                prom_mask = prom_ratio > prom_th
            wave_mask = wave_mask_abs & prom_mask

            # artifact
            artifact_mask, z_scores, local_sd = self._artifact_mask_local_z(sig)

            # highlight windows (high confidence)
            highlight_windows = []
            for m in metrics:
                if not m['valid']:
                    continue
                s0, s1 = m['start'], m['end']
                seg_len = s1 - s0
                pacf_coverage = (full_pacf_mask[s0:s1].sum() / seg_len) if seg_len > 0 else 0.0
                if (pacf_coverage >= (float(self.pacf_overlap_pct.value)/100.0)
                        and (wave_mask[s0:s1].sum()/(seg_len if seg_len>0 else 1)) >= (float(self.wave_overlap_pct.value)/100.0)
                        and (env_above[s0:s1].sum()/(seg_len if seg_len>0 else 1)) >= (float(self.env_overlap_pct.value)/100.0)
                        and (not artifact_mask[s0:s1].any() or (hasattr(self,'artifact_enable') and not self.artifact_enable.value))):
                    highlight_windows.append((s0 / fs, s1 / fs))

            # PSD (for plotting current epoch)
            fmin_psd = fmin_ext
            fmax_psd = fmax_ext
            n_samps = sig.size
            seg_len = int(min(n_samps, max(256, round(fs * 2.0))))
            n_fft = 1 << (seg_len - 1).bit_length()
            if n_fft < seg_len:
                n_fft = seg_len
            if n_fft > n_samps:
                n_fft = n_samps
            n_per_seg = seg_len
            n_overlap = seg_len // 2

            psd, psd_freqs = mne.time_frequency.psd_array_welch(
                sig[np.newaxis, :], sfreq=fs,
                fmin=fmin_psd, fmax=fmax_psd,
                n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap,
                verbose=False
            )
            psd = psd[0]
            psd_db = 10 * np.log10(psd + np.finfo(float).tiny)

            fit_mask = psd_freqs > 0
            f_fit = psd_freqs[fit_mask]
            p_fit = psd[fit_mask]
            logf = np.log10(f_fit)
            logp = np.log10(p_fit)
            beta, logA = np.polyfit(logf, logp, 1)
            logp_hat_all = np.full_like(psd_freqs, np.nan, dtype=float)
            valid_all = psd_freqs > 0
            logp_hat_all[valid_all] = beta * np.log10(psd_freqs[valid_all]) + logA
            aperiodic_db = np.full_like(psd_db, np.nan, dtype=float)
            aperiodic_db[valid_all] = 10 * logp_hat_all[valid_all]
            psd_resid_db = psd_db - aperiodic_db

            # build segment table rows (current epoch)
            # Epoch-level 1/f z & pass
            f_psd2, P_db2, Pfit_db2, f_fit2, z_resid2 = psd_aperiodic_and_peaks(sig, fs, fmin=float(self.onef_fit_lo.value), fmax=float(self.onef_fit_hi.value), nperseg_s=2.0)
            mband2 = (f_fit2 >= float(self.low_freq.value)) & (f_fit2 <= float(self.high_freq.value))
            onef_z = float(np.nanmax(z_resid2[mband2])) if np.any(mband2) else np.nan
            onef_ok = bool(onef_z >= float(self.onef_zthr.value)) if np.isfinite(onef_z) else False

            col_labels = [
                "Epoch#", "Start (s)", "End (s)", "Duration (s)",
                "Period CV", "Amp CV", "Valid Cycles",
                "PACF OK", "Power OK", "Artifact?", "Env≥50%?", "1/f OK?", "1/f Z", "Status"
            ]
            fmt_rows = []
            for m in metrics:
                s0, s1 = m['start'], m['end']
                dur_s = (s1 - s0) / fs
                valid_cycles = bool(m['valid'])
                seg_len = s1 - s0
                coverage = (full_pacf_mask[s0:s1].sum() / seg_len) if seg_len > 0 else 0.0
                pacf_ok = coverage >= (float(self.pacf_overlap_pct.value)/100.0)
                wave_frac = float(wave_mask[s0:s1].mean()) if seg_len > 0 else 0.0
                power_ok = (wave_frac >= (float(self.wave_overlap_pct.value)/100.0))
                artifact = bool(artifact_mask[s0:s1].any())
                seg_env_frac = float(env_above[s0:s1].mean()) if seg_len > 0 else 0.0
                env_ok = seg_env_frac >= (float(self.env_overlap_pct.value)/100.0)
                onef_gate_ok = (onef_ok or (not self.onef_gate.value))
                status = 'burst' if (valid_cycles and pacf_ok and power_ok and env_ok and onef_gate_ok and not artifact) else 'fail'
                fmt_rows.append([
                    int(ep),
                    f"{s0 / fs:.3f}",
                    f"{s1 / fs:.3f}",
                    f"{dur_s:.3f}",
                    f"{m['per_cv']:.3f}" if np.isfinite(m['per_cv']) else "nan",
                    f"{m['amp_cv']:.3f}" if np.isfinite(m['amp_cv']) else "nan",
                    str(valid_cycles),
                    str(pacf_ok),
                    str(power_ok),
                    str(artifact),
                    "True" if env_ok else "False",
                    ("True" if (onef_ok or (not self.onef_gate.value)) else "False"),
                    f"{onef_z:.2f}",
                    status
                ])
# -------------------------------------------------------------
            # Plot panels (kept from your original)
            # -------------------------------------------------------------
            fig = plt.figure(figsize=(10, 19))
            gs = fig.add_gridspec(
                12, 1,
                height_ratios=[1, 0.6, 1, 0.8, 0.3, 1, 1, 0.1, 0.6, 0.1, 0.8, 0.6],
                hspace=0.4
            )

            # pACF
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.plot(pacf_t, stab, label='pACF Stability')
            ax0.axhline(self.stab_th_slider.value, linestyle='--', label='Th')
            for s, e in pacf_segs:
                ax0.axvspan(s, e, alpha=0.3)
            ax0.set_title(f'pACF (Ch {ch}, Ep {ep})')
            ax0.set_ylim(0, 1)
            ax0.legend(loc='upper right')

            # Env vs pACF
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
            for s, e in both_segs:
                ax1.axvspan(times[s], times[e - 1], color='green', alpha=0.15, zorder=0)
            ax1.plot(times, envelope_amp / thr_env, label='Env / pctile', color='tab:blue')
            ax1.plot(pacf_t, stab, label='pACF Stability', color='tab:orange')
            ax1.axhline(1, linestyle='--', color='gray', label='Env threshold')
            ax1.plot(times, both_trace_scaled, color='green', linewidth=1, label='Env ∧ pACF')
            ax1.set_ylim(0, 1.05)
            ax1.set_title(
                'Envelope vs. pACF Stability Overlay\n'
                f'J={agree["jacc"]:.2f}  P={agree["prec"]:.2f}  R={agree["rec"]:.2f}  '
                f'F1={agree["f1"]:.2f}  SegMed={agree["seg_med"]:.2f}'
            )
            ax1.legend(loc='upper right')

            # Raw & Filtered
            ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
            ax2.plot(times, sig, label='Raw')
            ax2.plot(times, filt, label='Filtered')
            for m in metrics:
                c = 'green' if m['valid'] else 'orange'
                ax2.axvspan(m['start'] / fs, m['end'] / fs, color=c, alpha=0.3)
            if artifact_mask.any():
                ax2.fill_between(times, np.min(sig), np.max(sig),
                                 where=artifact_mask, color='red', alpha=0.1, label='Artifact')
            ax2.set_title('Raw & Filtered')
            ax2.legend(loc='upper right')

            # Cycle panel
            axC = fig.add_subplot(gs[3, 0], sharex=ax2)
            axC.plot(times, filt, color='0.6', linewidth=0.8, label='Filtered (cycle ref)')
            if markers is not None:
                samps = np.arange(filt.size, dtype=float)
                tr_t = markers['trough'] / fs
                tr_y = np.interp(markers['trough'], samps, filt, left=np.nan, right=np.nan)
                pk_t = markers['peak'] / fs
                pk_y = np.interp(markers['peak'], samps, filt, left=np.nan, right=np.nan)
                r50_t = markers['rise50'] / fs
                r50_y = np.interp(markers['rise50'], samps, filt, left=np.nan, right=np.nan)
                f50_t = markers['fall50'] / fs
                f50_y = np.interp(markers['fall50'], samps, filt, left=np.nan, right=np.nan)
                axC.scatter(tr_t, tr_y, c='C2', s=18, marker='v', label='Trough')
                axC.scatter(r50_t, r50_y, c='C3', s=18, marker='o', label='50% Rise')
                axC.scatter(pk_t, pk_y, c='C1', s=20, marker='^', label='Peak')
                axC.scatter(f50_t, f50_y, c='C4', s=18, marker='s', label='50% Fall')
            axC.set_ylabel('Filt Amp')
            axC.set_title('Cycle Markers (filtered waveform)')
            axC.legend(loc='upper right', ncol=2, fontsize=8)

            # Quality
            axQ = fig.add_subplot(gs[4, 0], sharex=ax2)
            status_arr = np.zeros_like(times)
            for m in metrics:
                s, e = int(m['start']), int(m['end'])
                seg_len = max(1, e - s)
                coverage = (full_pacf_mask[s:e].sum() / seg_len)
                pacf_ok = coverage >= (float(self.pacf_overlap_pct.value)/100.0)
                power_ok = bool(wave_mask[s:e].any())
                artifact = bool(artifact_mask[s:e].any())
                seg_env_frac = float(env_mask[s:e].mean())
                env_ok = seg_env_frac >= 0.50
                is_burst = (bool(m['valid']) and pacf_ok and power_ok and env_ok and not artifact)
                status_arr[s:e] = 1 if is_burst else -1
                mid = (s + e) / (2 * fs)
                axQ.text(
                    mid,
                    0.8 if is_burst else -0.8,
                    'burst' if is_burst else 'fail',
                    ha='center', va='center', fontsize=8
                )
            axQ.fill_between(times, status_arr, 0, where=status_arr > 0, color='green', alpha=0.5)
            axQ.fill_between(times, status_arr, 0, where=status_arr < 0, color='orange', alpha=0.5)
            axQ.set_ylim(-1.1, 1.1)
            axQ.set_yticks([])
            axQ.set_ylabel('Q')

            # Hilbert Envelope
            ax3 = fig.add_subplot(gs[5, 0], sharex=ax2)
            ax3.plot(times, envelope_amp, label='Envelope')
            ax3.axhline(
                thr_env, linestyle='--', color='red',
                label=f'{self.threshold.value}th percentile'
            )
            ax3.set_title('Hilbert Envelope')
            ax3.legend(loc='upper right')

            # Wavelet extended
            ax4 = fig.add_subplot(gs[6, 0], sharex=ax2)
            im_ext = ax4.imshow(
                power_ext_raw, aspect='auto', origin='lower',
                extent=[times[0], times[-1], freqs_ext[0], freqs_ext[-1]]
            )
            ax4.axhline(lf, color='w', linestyle=':', linewidth=0.8)
            ax4.axhline(hf, color='w', linestyle=':', linewidth=0.8)
            ax4.set_title('Wavelet Power (raw, extended)')
            ax4.set_ylabel('Freq (Hz)')
            ax4.set_xlabel('Time (s)')
            # ensure prom_pct exists for the label
            if 'prom_pct' not in locals():
                prom_pct = float(getattr(self, 'wave_prom_pct', type('o', (object,), {'value': 90.0})()).value)
            txt = ('Prom: flanks NA (abs only)' if np.isnan(prom_ratio).all() else f"Prom>{prom_pct:.0f}% (thr={prom_th:.3g})")
            ax4.text(
                0.01, 0.98, txt, ha='left', va='top', transform=ax4.transAxes,
                fontsize=8, color='w',
                bbox=dict(boxstyle='round', facecolor='k', alpha=0.3, lw=0)
            )

            # Wavelet colorbar extended
            cax_ext = fig.add_subplot(gs[7, 0])
            plt.colorbar(im_ext, cax=cax_ext, orientation='horizontal', label='Power')

            # Wavelet band only
            ax4b = fig.add_subplot(gs[8, 0], sharex=ax2)
            if power_band_raw.size:
                im_band = ax4b.imshow(
                    power_band_raw, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs_band[0], freqs_band[-1]]
                )
            else:
                im_band = None
                ax4b.text(0.5, 0.5, 'No data in band', transform=ax4b.transAxes,
                          ha='center', va='center', color='red')
            ax4b.set_ylabel('Freq (Hz)')
            ax4b.set_xlabel('Time (s)')
            ax4b.set_title('Wavelet Power (band only)')

            # Wavelet colorbar band
            cax_band = fig.add_subplot(gs[9, 0])
            if im_band is not None:
                plt.colorbar(im_band, cax=cax_band, orientation='horizontal', label='Power')
            else:
                cax_band.axis('off')

            # PSD bottom
            axP = fig.add_subplot(gs[10, 0])
            # Compute PSD + 1/f residuals
            f_psd, P_db, Pfit_db, f_fit, z_resid = psd_aperiodic_and_peaks(
                sig, fs, fmin=float(self.onef_fit_lo.value), fmax=float(self.onef_fit_hi.value), nperseg_s=2.0)
            lf = float(self.low_freq.value); hf = float(self.high_freq.value)
            mband = (f_fit >= lf) & (f_fit <= hf)
            onef_z = float(np.nanmax(z_resid[mband])) if np.any(mband) else np.nan
            onef_pass = bool(onef_z >= float(self.onef_zthr.value)) if np.isfinite(onef_z) else False
            # Plot PSD and 1/f fit
            axP.plot(f_psd, P_db, linewidth=1.0, label='PSD (dB)')
            if np.isfinite(Pfit_db).any():
                axP.plot(f_psd[np.isfinite(Pfit_db)], Pfit_db[np.isfinite(Pfit_db)], linewidth=1.5,
                        label=f"1/f fit ({self.onef_fit_lo.value:.0f}–{self.onef_fit_hi.value:.0f} Hz)")
            # Shade analysis band with pass/fail color
            axP.axvspan(lf, hf, color=('tab:green' if onef_pass else 'tab:red'), alpha=0.12)
            # Axis ranges
            if self.psd_auto_x.value:
                axP.set_xlim(float(self.onef_fit_lo.value), float(self.onef_fit_hi.value))
            else:
                axP.set_xlim(float(self.psd_xmin.value), float(self.psd_xmax.value))
            if self.psd_auto_y.value:
                _autoscale_db(axP, [P_db, Pfit_db], pad_db=3.0, robust=True)
            else:
                axP.set_ylim(float(self.psd_ymin.value), float(self.psd_ymax.value))
            # Annotate z value near top AFTER y-lims are set
            y0, y1 = axP.get_ylim()
            y_txt = y1 - 0.05 * (y1 - y0)
            if np.isfinite(onef_z):
                axP.text((lf+hf)/2.0, y_txt, f"Z={onef_z:.2f}", ha='center', va='top', fontsize=9)
            axP.set_xlabel('Frequency (Hz)')
            axP.set_ylabel('Power (dB)')
            axP.set_title('PSD + 1/f fit (epoch-level)')
            axP.legend(loc='best', fontsize=8)

            # results table (current epoch)
            axT = fig.add_subplot(gs[11, 0])
            axT.axis('off')
            if fmt_rows:
                table = axT.table(
                    cellText=fmt_rows,
                    colLabels=col_labels,
                    loc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.2)
            else:
                axT.text(0.5, 0.5, 'No segments', ha='center', va='center', fontsize=10)

            # highlight rectangles across time axes (not PSD/table)
            highlight_windows = []
            for m in metrics:
                if not m['valid']:
                    continue
                s0, s1 = m['start'], m['end']
                if ((full_pacf_mask[s0:s1].sum()/(seg_len if seg_len>0 else 1)) >= (float(self.pacf_overlap_pct.value)/100.0)
                        and (wave_mask[s0:s1].sum()/(seg_len if seg_len>0 else 1)) >= (float(self.wave_overlap_pct.value)/100.0)
                        and (env_above[s0:s1].sum()/(seg_len if seg_len>0 else 1)) >= (float(self.env_overlap_pct.value)/100.0)
                        and (not artifact_mask[s0:s1].any() or (hasattr(self,'artifact_enable') and not self.artifact_enable.value))):
                    highlight_windows.append((s0 / fs, s1 / fs))

            for ax in (ax0, ax1, ax2, axC, axQ, ax3, ax4, ax4b):
                ymin, ymax = ax.get_ylim()
                for t_start, t_end in highlight_windows:
                    rect = Rectangle(
                        (t_start, ymin),
                        width=(t_end - t_start), height=(ymax - ymin),
                        fill=False, edgecolor='black', linestyle='--', linewidth=1
                    )
                    ax.add_patch(rect)

            plt.show()


            # Raw & Filtered
            ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
            ax2.plot(times, sig, label='Raw')
            ax2.plot(times, filt, label='Filtered')
            for m in metrics:
                c = 'green' if m['valid'] else 'orange'
                ax2.axvspan(m['start'] / fs, m['end'] / fs, color=c, alpha=0.3)
            if artifact_mask.any():
                ax2.fill_between(times, np.min(sig), np.max(sig),
                                 where=artifact_mask, color='red', alpha=0.1, label='Artifact')
            ax2.set_title('Raw & Filtered')
            ax2.legend(loc='upper right')

            # Cycle panel
            axC = fig.add_subplot(gs[3, 0], sharex=ax2)
            axC.plot(times, filt, color='0.6', linewidth=0.8, label='Filtered (cycle ref)')
            if markers is not None:
                samps = np.arange(filt.size, dtype=float)
                tr_t = markers['trough'] / fs
                tr_y = np.interp(markers['trough'], samps, filt, left=np.nan, right=np.nan)
                pk_t = markers['peak'] / fs
                pk_y = np.interp(markers['peak'], samps, filt, left=np.nan, right=np.nan)
                r50_t = markers['rise50'] / fs
                r50_y = np.interp(markers['rise50'], samps, filt, left=np.nan, right=np.nan)
                f50_t = markers['fall50'] / fs
                f50_y = np.interp(markers['fall50'], samps, filt, left=np.nan, right=np.nan)
                axC.scatter(tr_t, tr_y, c='C2', s=18, marker='v', label='Trough')
                axC.scatter(r50_t, r50_y, c='C3', s=18, marker='o', label='50% Rise')
                axC.scatter(pk_t, pk_y, c='C1', s=20, marker='^', label='Peak')
                axC.scatter(f50_t, f50_y, c='C4', s=18, marker='s', label='50% Fall')
            axC.set_ylabel('Filt Amp')
            axC.set_title('Cycle Markers (filtered waveform)')
            axC.legend(loc='upper right', ncol=2, fontsize=8)

            # Quality
            axQ = fig.add_subplot(gs[4, 0], sharex=ax2)
            status_arr = np.zeros_like(times)
            for m in metrics:
                s, e = int(m['start']), int(m['end'])
                seg_len = max(1, e - s)
                coverage = (full_pacf_mask[s:e].sum() / seg_len)
                pacf_ok = coverage >= (float(self.pacf_overlap_pct.value)/100.0)
                power_ok = bool(wave_mask[s:e].any())
                artifact = bool(artifact_mask[s:e].any())
                seg_env_frac = float(env_mask[s:e].mean())
                env_ok = seg_env_frac >= 0.50
                is_burst = (bool(m['valid']) and pacf_ok and power_ok and env_ok and not artifact)
                status_arr[s:e] = 1 if is_burst else -1
                mid = (s + e) / (2 * fs)
                axQ.text(
                    mid,
                    0.8 if is_burst else -0.8,
                    'burst' if is_burst else 'fail',
                    ha='center', va='center', fontsize=8
                )
            axQ.fill_between(times, status_arr, 0, where=status_arr > 0, color='green', alpha=0.5)
            axQ.fill_between(times, status_arr, 0, where=status_arr < 0, color='orange', alpha=0.5)
            axQ.set_ylim(-1.1, 1.1)
            axQ.set_yticks([])
            axQ.set_ylabel('Q')

            # Hilbert Envelope
            ax3 = fig.add_subplot(gs[5, 0], sharex=ax2)
            ax3.plot(times, envelope_amp, label='Envelope')
            ax3.axhline(
                thr_env, linestyle='--', color='red',
                label=f'{self.threshold.value}th percentile'
            )
            ax3.set_title('Hilbert Envelope')
            ax3.legend(loc='upper right')

            # Wavelet extended
            ax4 = fig.add_subplot(gs[6, 0], sharex=ax2)
            im_ext = ax4.imshow(
                power_ext_raw, aspect='auto', origin='lower',
                extent=[times[0], times[-1], freqs_ext[0], freqs_ext[-1]]
            )
            ax4.axhline(lf, color='w', linestyle=':', linewidth=0.8)
            ax4.axhline(hf, color='w', linestyle=':', linewidth=0.8)
            ax4.set_title('Wavelet Power (raw, extended)')
            ax4.set_ylabel('Freq (Hz)')
            ax4.set_xlabel('Time (s)')
            txt = ('Prom: flanks NA (abs only)' if np.isnan(prom_ratio).all() else f"Prom>{prom_pct:.0f}% (thr={prom_th:.3g})")  # short label
            ax4.text(
                0.01, 0.98, txt, ha='left', va='top', transform=ax4.transAxes,
                fontsize=8, color='w',
                bbox=dict(boxstyle='round', facecolor='k', alpha=0.3, lw=0)
            )

            # Wavelet colorbar extended
            cax_ext = fig.add_subplot(gs[7, 0])
            plt.colorbar(im_ext, cax=cax_ext, orientation='horizontal', label='Power')

            # Wavelet band only
            ax4b = fig.add_subplot(gs[8, 0], sharex=ax2)
            if power_band_raw.size:
                im_band = ax4b.imshow(
                    power_band_raw, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs_band[0], freqs_band[-1]]
                )
            else:
                im_band = None
                ax4b.text(0.5, 0.5, 'No data in band', transform=ax4b.transAxes,
                          ha='center', va='center', color='red')
            ax4b.set_ylabel('Freq (Hz)')
            ax4b.set_xlabel('Time (s)')
            ax4b.set_title('Wavelet Power (band only)')

            # Wavelet colorbar band
            cax_band = fig.add_subplot(gs[9, 0])
            if im_band is not None:
                plt.colorbar(im_band, cax=cax_band, orientation='horizontal', label='Power')
            else:
                cax_band.axis('off')

            # PSD bottom
            axP = fig.add_subplot(gs[10, 0])
            axP.plot(psd_freqs, psd_db, color='0.6', linewidth=1, label='Raw PSD')
            axP.axvspan(lf, hf, color='green', alpha=0.15)
            axP.axvline(lf, color='green', linestyle='--', linewidth=0.8)
            axP.axvline(hf, color='green', linestyle='--', linewidth=0.8)
            axP.set_xlim(psd_freqs[0], psd_freqs[-1])
            axP.set_xlabel('Frequency (Hz)')
            axP.set_ylabel('PSD (dB)')
            axP.set_title('Epoch PSD (raw) & 1/f-corrected residual')

            axP2 = axP.twinx()
            axP2.plot(psd_freqs, psd_resid_db, color='tab:blue', linewidth=1.25, label='1/f-corrected')
            axP2.axhline(0, color='tab:blue', linestyle=':', linewidth=0.8)
            axP2.set_ylabel('Residual (dB vs 1/f)')
            axP.text(
                0.98, 0.95, f'β={beta:.2f}',
                transform=axP.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='w', alpha=0.6, lw=0.5)
            )
            lines1, labels1 = axP.get_legend_handles_labels()
            lines2, labels2 = axP2.get_legend_handles_labels()
            axP2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

            # results table (current epoch)
            axT = fig.add_subplot(gs[11, 0])
            axT.axis('off')
            if fmt_rows:
                table = axT.table(
                    cellText=fmt_rows,
                    colLabels=col_labels,
                    loc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.2)
            else:
                axT.text(0.5, 0.5, 'No segments', ha='center', va='center', fontsize=10)

            # highlight rectangles across time axes (not PSD/table)
            highlight_windows = []
            for m in metrics:
                if not m['valid']:
                    continue
                s0, s1 = m['start'], m['end']
                if ((full_pacf_mask[s0:s1].sum()/(seg_len if seg_len>0 else 1)) >= (float(self.pacf_overlap_pct.value)/100.0)
                        and (wave_mask[s0:s1].sum()/(seg_len if seg_len>0 else 1)) >= (float(self.wave_overlap_pct.value)/100.0)
                        and (env_above[s0:s1].sum()/(seg_len if seg_len>0 else 1)) >= (float(self.env_overlap_pct.value)/100.0)
                        and not artifact_mask[s0:s1].any()):
                    highlight_windows.append((s0 / fs, s1 / fs))

            for ax in (ax0, ax1, ax2, axC, axQ, ax3, ax4, ax4b):
                ymin, ymax = ax.get_ylim()
                for t_start, t_end in highlight_windows:
                    rect = Rectangle(
                        (t_start, ymin),
                        width=(t_end - t_start), height=(ymax - ymin),
                        fill=False, edgecolor='black', linestyle='--', linewidth=1
                    )
                    ax.add_patch(rect)

            plt.show()

def _robust_z_vector(v, eps=1e-9):
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = 1.4826 * max(mad, eps)
    return (v - med) / scale

def psd_aperiodic_and_peaks(x, fs, fmin=2.0, fmax=40.0, nperseg_s=2.0):
    """Welch PSD + log–log 1/f fit -> residual z, and dB curves for plotting.
    Returns: f, P_db, Pfit_db, f_fit, z_resid
    """
    nperseg = int(max(64, min(len(x), round(nperseg_s*fs))))
    noverlap = int(round(0.5*nperseg))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant')
    mask = (f >= fmin) & (f <= fmax)
    f_fit, P_fit = f[mask], Pxx[mask]
    xlog = np.log10(np.clip(f_fit, 1e-9, None))
    ylog = np.log10(np.clip(P_fit, 1e-18, None))
    A = np.vstack([np.ones_like(xlog), xlog]).T
    coef, *_ = np.linalg.lstsq(A, ylog, rcond=None)
    a, b = coef
    yhat = a + b * xlog
    resid = ylog - yhat
    z_resid = _robust_z_vector(resid)
    # dB curves for plotting
    P_db = 10.0 * np.log10(np.clip(Pxx, 1e-18, None))
    Pfit_db = np.full_like(P_db, np.nan, dtype=float)
    Pfit_db[mask] = 10.0 * np.log10(np.clip(10**yhat, 1e-18, None))
    return f, P_db, Pfit_db, f_fit, z_resid