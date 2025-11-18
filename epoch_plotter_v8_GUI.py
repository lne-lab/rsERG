import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import mne
import ipywidgets as widgets
from IPython.display import display, clear_output

# --------------------------------------
# Computation functions
# --------------------------------------
def compute_stability_index(pacf, lag_cycles):
    valid = ~np.isnan(pacf)
    if not valid.any():
        return np.nan
    area = np.trapz(pacf[valid], lag_cycles[valid])
    return area / (lag_cycles[-1] - lag_cycles[0])


def compute_pacf_stability_over_time(data, sr, pacf_win, pacf_step,
                                     min_lag=0.5, max_lag=5.0, n_lag=50):
    n_ch, n_samp = data.shape
    w = int(pacf_win * sr)
    s = int(pacf_step * sr)
    lag_cycles = np.linspace(min_lag, max_lag, n_lag)
    times, stability, avg_freqs = [], [], []
    for start in range(0, n_samp - w + 1, s):
        seg = data[:, start:start + w]
        st_row, mf = [], []
        for ch in range(n_ch):
            phase = np.unwrap(np.angle(hilbert(seg[ch])))
            inst_freq = np.diff(phase) * sr / (2 * np.pi)
            mean_f = np.nanmean(inst_freq)
            mf.append(mean_f)
            pacf = []
            N = len(phase)
            for lag in lag_cycles:
                L = int(np.rint(lag * sr / mean_f))
                if L <= 0 or L >= N:
                    pacf.append(np.nan)
                else:
                    pdiff = phase[:N - L] - phase[L:]
                    pacf.append(np.abs(np.mean(np.exp(1j * pdiff))))
            st_row.append(compute_stability_index(np.array(pacf), lag_cycles))
        times.append((start + w / 2) / sr)
        stability.append(st_row)
        avg_freqs.append(np.nanmean(mf))
    return np.array(times), np.array(stability), np.array(avg_freqs)


def compute_time_resolved_plv_per_channel(data, sr, plv_win, plv_step):
    n_ch, n_samp = data.shape
    w = int(plv_win * sr)
    s = int(plv_step * sr)
    times, plvs = [], []
    for start in range(0, n_samp - w + 1, s):
        seg = data[:, start:start + w]
        phases = np.unwrap(np.angle(hilbert(seg, axis=1)), axis=1)
        ch_plv = []
        for i in range(n_ch):
            vals = [np.abs(np.mean(np.exp(1j * (phases[i] - phases[j]))))
                    for j in range(n_ch) if j != i]
            ch_plv.append(np.mean(vals))
        times.append((start + w / 2) / sr)
        plvs.append(ch_plv)
    return np.array(times), np.array(plvs)

# --------------------------------------
# Plotting
# --------------------------------------
def plot_epoch(epoch_data, sr, burst_windows, channels,
               pacf_win=0.5, pacf_step=0.1,
               plv_win=0.3, plv_step=0.08,
               min_lag=1.0, max_lag=8.0, n_lag=60,
               plv_thresh=0.9, stab_thresh=0.5,
               alpha=0.05):
    data = epoch_data[channels]
    t_full = np.arange(data.shape[1]) / sr

    t_stab, stab, _ = compute_pacf_stability_over_time(
        data, sr, pacf_win, pacf_step,
        min_lag=min_lag, max_lag=max_lag, n_lag=n_lag
    )
    t_plv, plv = compute_time_resolved_plv_per_channel(
        data, sr, plv_win, plv_step
    )

    avg_stab = np.nanmean(stab, axis=1)
    avg_plv = np.nanmean(plv, axis=1)
    interp_plv = np.interp(t_stab, t_plv, avg_plv, left=np.nan, right=np.nan)
    mask = (interp_plv > plv_thresh) & (avg_stab > stab_thresh)

    # determine highlight spans
    segments, start_idx = [], None
    for i, m in enumerate(mask):
        if m and start_idx is None:
            start_idx = i
        elif not m and start_idx is not None:
            segments.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(mask) - 1))
    half = pacf_win / 2
    highlight_spans = [(t_stab[s] - half, t_stab[e] + half) for s, e in segments]

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(24, 9),
        gridspec_kw={'height_ratios': [1, 1, 3]}
    )

    # Stability plot
    for idx, ch in enumerate(channels):
        axes[0].plot(t_stab, stab[:, idx], label=f"Ch {ch + 1}")
    axes[0].plot(t_stab, avg_stab, 'k-', lw=2, label='Mean')
    axes[0].set_ylabel('Stability')
    axes[0].set_title(f'pACF Stability')

    # PLV plot
    for idx, ch in enumerate(channels):
        axes[1].plot(t_plv, plv[:, idx], label=f"Ch {ch + 1}")
    axes[1].plot(t_plv, avg_plv, 'k-', lw=2, label='Mean')
    axes[1].set_ylabel('PLV')
    axes[1].set_title(f'PLV')

    # Raw data
    offset = np.max(np.abs(data)) * 1.2
    for idx in range(len(channels)):
        axes[2].plot(t_full, data[idx] + idx * offset)
    axes[2].set_yticks([i * offset for i in range(len(channels))])
    axes[2].set_yticklabels([f"Ch {c + 1}" for c in channels])
    axes[2].set_ylabel('Amplitude + Offset')

    # Highlights
    for ax in axes:
        for s, e in burst_windows:
            ax.axvspan(s, e, color='magenta', alpha=0.3)
        for s, e in highlight_spans:
            ax.add_patch(
                plt.Rectangle((s, ax.get_ylim()[0]), e - s,
                              ax.get_ylim()[1] - ax.get_ylim()[0],
                              facecolor='none', edgecolor='black', linestyle='--', alpha=0.7)
            )
        ax.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()
    import matplotlib.patches as mpatches
    handles += [
        mpatches.Patch(facecolor='magenta', alpha=0.3, label='Burst Window'),
        mpatches.Patch(facecolor='none', edgecolor='black', linestyle='--', label='pACF+PLV>Th')
    ]
    axes[0].legend(handles=handles, ncol=4, fontsize='small')

    plt.tight_layout()
    plt.show()

# --------------------------------------
# Suprathreshold detection with progress
# --------------------------------------
def detect_suprathreshold_events(epochs, sr,
                                 pacf_win, pacf_step,
                                 plv_win, plv_step,
                                 min_lag, max_lag, n_lag,
                                 plv_thresh, stab_thresh):
    events = []
    data_all = epochs.get_data(copy=True)
    total_epochs = data_all.shape[0]
    progress = widgets.IntProgress(value=0, min=0, max=total_epochs, description='Detecting epochs:')
    display(progress)
    for ep, ep_data in enumerate(data_all):
        for eye, chans in [('Eye1', list(range(8))), ('Eye2', list(range(8, 16)))]:
            data = ep_data[chans]
            t_stab, stab, avg_freqs = compute_pacf_stability_over_time(
                data, sr, pacf_win, pacf_step,
                min_lag, max_lag, n_lag
            )
            t_plv, plv = compute_time_resolved_plv_per_channel(data, sr, plv_win, plv_step)
            avg_stab = np.nanmean(stab, axis=1)
            avg_plv = np.nanmean(plv, axis=1)
            interp_plv = np.interp(t_stab, t_plv, avg_plv, left=np.nan, right=np.nan)
            mask = (interp_plv > plv_thresh) & (avg_stab > stab_thresh)
            segments, start_idx = [], None
            for i, m in enumerate(mask):
                if m and start_idx is None:
                    start_idx = i
                elif not m and start_idx is not None:
                    segments.append((start_idx, i - 1))
                    start_idx = None
            if start_idx is not None:
                segments.append((start_idx, len(mask) - 1))
            half = pacf_win / 2
            for s, e in segments:
                start_s, end_s = t_stab[s] - half, t_stab[e] + half
                mean_f_seg = np.nanmean(avg_freqs[s:e + 1])
                cycles = (end_s - start_s) * mean_f_seg
                events.append({'Epoch': ep, 'Eye': eye,
                               'Start_s': start_s, 'End_s': end_s,
                               'Cycles': cycles})
        progress.value = ep + 1
    progress.close()
    return events

# --------------------------------------
# Export to Excel with metrics and progress
# --------------------------------------
def export_results(input_excel, epochs, bursts, sr,
                   pacf_win, pacf_step,
                   plv_win, plv_step,
                   min_lag, max_lag, n_lag,
                   plv_thresh, stab_thresh):
    clear_output()
    print("Starting export...")
    all_sheets = pd.read_excel(input_excel, sheet_name=None)
    print(f"Read {len(all_sheets)} sheets")
    # detect events with progress bar
    events = detect_suprathreshold_events(
        epochs, sr,
        pacf_win, pacf_step,
        plv_win, plv_step,
        min_lag, max_lag, n_lag,
        plv_thresh, stab_thresh
    )
    print(f"Detected {len(events)} suprathreshold events")
    df_global = all_sheets.get('Global').copy()
    df_global.columns = [str(c).strip() for c in df_global.columns]
    df_events = pd.DataFrame(events)
    # annotate overlap and compute avg metrics
    overlap_flags, avg_pacf_list, avg_plv_list = [], [], []
    ts_cache = {}
    for _, row in df_global.iterrows():
        ep = int(row['Epoch']); eye = row.get('Eye', 'Eye1')
        key = (ep, eye)
        if key not in ts_cache:
            data = epochs.get_data(copy=True)[ep][list(range(8)) if eye=='Eye1' else list(range(8,16))]
            t_stab, stab, _ = compute_pacf_stability_over_time(
                data, sr, pacf_win, pacf_step, min_lag, max_lag, n_lag)
            avg_stab = np.nanmean(stab, axis=1)
            t_plv, plv = compute_time_resolved_plv_per_channel(data, sr, plv_win, plv_step)
            avg_plv = np.nanmean(plv, axis=1)
            ts_cache[key] = (t_stab, avg_stab, t_plv, avg_plv)
        t_stab, avg_stab, t_plv, avg_plv = ts_cache[key]
        gs, ge = float(row['Start_s']), float(row['End_s'])
        has = df_events[(df_events.Epoch==ep)&(df_events.Eye==eye)&
                        ~((df_events.End_s<gs)|(df_events.Start_s>ge))].any(axis=None)
        overlap_flags.append('yes' if has else 'no')
        stab_idx = (t_stab >= gs)&(t_stab <= ge)
        plv_idx  = (t_plv  >= gs)&(t_plv  <= ge)
        avg_pacf_list.append(np.nanmean(avg_stab[stab_idx]) if np.any(stab_idx) else np.nan)
        avg_plv_list.append(np.nanmean(avg_plv[plv_idx]) if np.any(plv_idx) else np.nan)
    df_global['pACF + PLV'] = overlap_flags
    df_global['avg_pACF'] = avg_pacf_list
    df_global['avg_PLV'] = avg_plv_list
    n_global_pos = sum(flag=='yes' for flag in overlap_flags)
    n_total_events = len(df_events)
    params = {
        'pacf_win': pacf_win, 'pacf_step': pacf_step,
        'plv_win': plv_win, 'plv_step': plv_step,
        'min_lag': min_lag, 'max_lag': max_lag,
        'n_lag': n_lag, 'plv_thresh': plv_thresh,
        'stab_thresh': stab_thresh,
        'n_global_positive': n_global_pos,
        'n_total_pACF_PLV_events': n_total_events
    }
    df_params = pd.DataFrame([params])
    base, ext = os.path.splitext(input_excel)
    out_file = f"{base}_pACF-PLV{ext}"
    print("Writing output Excel...")
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        for nm, df in all_sheets.items():
            if nm=='Global': df_global.to_excel(writer, nm, index=False)
            else: df.to_excel(writer, nm, index=False)
        df_events.to_excel(writer, 'PACF-PLV', index=False)
        df_params.to_excel(writer, 'pACF PLV details', index=False)
    print(f"Export complete: {out_file}")
    return out_file

# --------------------------------------
# Load data utility
# --------------------------------------
def load_data(fif_path, excel_path, sheet='Global'):
    epochs = mne.read_epochs(fif_path, preload=True)
    df = pd.read_excel(excel_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    bursts = {}
    for _, r in df.iterrows():
        key = (int(r['Epoch']), r.get('Eye','Eye1'))
        bursts.setdefault(key, []).append((float(r['Start_s']), float(r['End_s'])))
    return epochs, bursts, epochs.info['sfreq']

# --------------------------------------
# GUI Definition
# --------------------------------------
def create_gui():
    state={'epochs':None,'bursts':None,'sr':None}
    fif=widgets.Text(description='FIF Path:',layout=widgets.Layout(width='80%'))
    xls=widgets.Text(description='Excel Path:',layout=widgets.Layout(width='80%'))
    load_btn=widgets.Button(description='Load Data',button_style='info')
    export_btn=widgets.Button(description='Export Excel',button_style='success')
    epoch=widgets.IntSlider(description='Epoch',min=0,max=0)
    eye=widgets.Dropdown(options=['Eye1','Eye2'],value='Eye1',description='Eye:')
    plv_win=widgets.FloatSlider(value=0.3,min=0.05,max=1.0,step=0.05,description='PLV Win')
    plv_step=widgets.FloatSlider(value=0.08,min=0.01,max=0.2,step=0.01,description='PLV Step')
    pacf_win=widgets.FloatSlider(value=0.5,min=0.1,max=1.0,step=0.1,description='PACF Win')
    pacf_step=widgets.FloatSlider(value=0.1,min=0.01,max=0.2,step=0.01,description='PACF Step')
    min_lag=widgets.FloatSlider(value=1.0,min=0.1,max=5.0,step=0.1,description='Min Lag')
    max_lag=widgets.FloatSlider(value=8.0,min=1.0,max=15.0,step=0.5,description='Max Lag')
    n_lag=widgets.IntSlider(value=60,min=10,max=200,step=10,description='# Lags')
    stab_th=widgets.FloatSlider(value=0.5,min=0.0,max=1.0,step=0.05,description='Stab Th')
    plv_th=widgets.FloatSlider(value=0.9,min=0.0,max=1.0,step=0.05,description='PLV Th')
    out=widgets.Output()

    def refresh(*args):
        with out:
            clear_output()
            if state['epochs'] is None:
                print('Load data first')
                return
            ep_data = state['epochs'].get_data(copy=True)[epoch.value]
            chans = list(range(8)) if eye.value=='Eye1' else list(range(8,16))
            burst_windows = state['bursts'].get((epoch.value,eye.value),[])
            plot_epoch(
                ep_data,state['sr'],burst_windows,chans,
                pacf_win=pacf_win.value,pacf_step=pacf_step.value,
                plv_win=plv_win.value,plv_step=plv_step.value,
                min_lag=min_lag.value,max_lag=max_lag.value,
                n_lag=n_lag.value,plv_thresh=plv_th.value,
                stab_thresh=stab_th.value
            )

    def on_load(b):
        with out:
            clear_output()
            try:
                epc,br,fr=load_data(fif.value,xls.value)
                state.update(epochs=epc,bursts=br,sr=fr)
                epoch.max=len(epc)-1
                print(f'Loaded {len(epc)} epochs; sr={fr} Hz')
                refresh()
            except Exception as e:
                print('Error loading data:',e)

    def on_export(b):
        with out:
            clear_output()
            print('Exporting Excel—please wait...')
            try:
                out_file=export_results(
                    xls.value,state['epochs'],state['bursts'],state['sr'],
                    pacf_win.value,pacf_step.value,
                    plv_win.value,plv_step.value,
                    min_lag.value,max_lag.value,n_lag.value,
                    plv_th.value,stab_th.value
                )
                print(f'Results exported to {out_file}')
            except Exception as e:
                print('Export failed:',e)

    load_btn.on_click(on_load)
    export_btn.on_click(on_export)
    for w in [epoch,eye,plv_win,plv_step,pacf_win,pacf_step,
              min_lag,max_lag,n_lag,stab_th,plv_th]:
        w.observe(refresh,names='value')

    ui=widgets.VBox([
        fif,xls,
        widgets.HBox([load_btn,export_btn]),
        widgets.HBox([epoch,eye]),
        widgets.HBox([plv_win, plv_step, pacf_win, pacf_step]),
        widgets.HBox([min_lag, max_lag, n_lag]),
        widgets.HBox([stab_th, plv_th]),
        out
    ])
    display(ui)

# At the bottom of your script or notebook cell, call:
#create_gui()

