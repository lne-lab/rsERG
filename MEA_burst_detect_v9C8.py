import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from ipywidgets import (
    VBox, HBox, IntSlider, FloatSlider, Button, Output,
    Checkbox, Dropdown, Text, Label
)
from IPython.display import display
from collections import defaultdict

##############################################################################
# TOPOGRAPHIC MAP DICTIONARIES
##############################################################################

channel_positions_combined = {
    'Ch17': (-1,  1, 0),
    'Ch18': (-1,  0, 0),
    'Ch19': ( 0,  2, 0),
    'Ch20': ( 0,  0, 0),
    'Ch21': ( 0, -1, 0),
    'Ch22': ( 0,  1, 0),
    'Ch23': ( 1,  0, 0),
    'Ch24': ( 1,  1, 0),
    'Ch57': (-1, 5, 0),
    'Ch58': (-1, 4, 0),
    'Ch59': ( 0,  6, 0),
    'Ch60': ( 0,  4, 0),
    'Ch61': ( 0,  3, 0),
    'Ch62': ( 0,  5, 0),
    'Ch63': ( 1,  4, 0),
    'Ch64': ( 1,  5, 0),
}

channel_positions_separated = {
    'Ch17': (-1,  1, 0),
    'Ch18': (-1,  0, 0),
    'Ch19': ( 0,  2, 0),
    'Ch20': ( 0,  0, 0),
    'Ch21': ( 0, -1, 0),
    'Ch22': ( 0,  1, 0),
    'Ch23': ( 1,  0, 0),
    'Ch24': ( 1,  1, 0),
    'Ch57': (-1,  1, 0),
    'Ch58': (-1,  0, 0),
    'Ch59': ( 0,  2, 0),
    'Ch60': ( 0,  0, 0),
    'Ch61': ( 0, -1, 0),
    'Ch62': ( 0,  1, 0),
    'Ch63': ( 1,  0, 0),
    'Ch64': ( 1,  1, 0),
}

##############################################################################
# DATA LOADING AND SYNTHETIC CHANNELS
##############################################################################

def load_epochs(fif_path):
    epochs = mne.read_epochs(fif_path, preload=True)
    fs = epochs.info['sfreq']
    epoch_data = epochs.get_data()
    print(f"Loaded {epoch_data.shape[0]} epochs, {epoch_data.shape[1]} channels, "
          f"{epoch_data.shape[2]} samples per epoch")
    print(f"Sampling frequency: {fs} Hz")
    return epoch_data, fs

def add_synthetic_channels(epoch_data):
    eye1 = epoch_data[:, 0:8, :]
    eye2 = epoch_data[:, 8:16, :]
    syn_eye1 = np.mean(eye1, axis=1, keepdims=True)
    syn_eye2 = np.mean(eye2, axis=1, keepdims=True)
    new_epoch_data = np.concatenate([syn_eye1, epoch_data, syn_eye2], axis=1)
    return new_epoch_data

def create_channel_names(original_names=None):
    if original_names is None:
        eye1_names = [f"Eye1 Ch {i+1}" for i in range(8)]
        eye2_names = [f"Eye2 Ch {i+9}" for i in range(8)]
        original_names = eye1_names + eye2_names
    new_names = []
    new_names.append("Synthetic Eye1 Average (Ch 1-8)")
    new_names.extend(original_names)
    new_names.append("Synthetic Eye2 Average (Ch 9-16)")
    return new_names

##############################################################################
# TOPOGRAPHY
##############################################################################

from matplotlib.patches import Ellipse

def draw_mea_background(ax):
    ellipse = Ellipse(xy=(0, 2.5), width=3, height=5, edgecolor='none',
                      facecolor='lightgrey', alpha=0.5)
    ax.add_patch(ellipse)

def plot_topographic(ax, topo_positions, channel_names, channels_to_show):
    draw_mea_background(ax)
    colors = ['red','blue','green','orange','purple','brown','cyan','magenta']
    for i, ch_idx in enumerate(channels_to_show):
        if ch_idx < 9:
            dict_key = f"Ch{16 + ch_idx}"
        else:
            dict_key = f"Ch{48 + ch_idx}"
        if dict_key in topo_positions:
            x, y, _ = topo_positions[dict_key]
            ax.scatter(x, y, s=500, color=colors[i % len(colors)], zorder=3)
            label = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch {ch_idx}"
            ax.text(x, y, label, ha='center', va='center', color='black', fontsize=9,
                    fontweight='bold', zorder=4)
    ax.set_title("Channel Topography", color='black')
    ax.set_xlabel("X", color='black')
    ax.set_ylabel("Y", color='black')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 7)
    ax.set_xticks([])
    ax.set_yticks([])

##############################################################################
# HELPER FUNCTION FOR SHARED BURST INTERVALS
##############################################################################

def compute_shared_intervals(burst_intervals, threshold):
    events = []
    for start, end in burst_intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda x: (x[0], -x[1]))

    intervals = []
    cum = 0
    current_start = None
    for time, delta in events:
        prev = cum
        cum += delta
        if prev < threshold and cum >= threshold:
            current_start = time
        elif prev >= threshold and cum < threshold and current_start is not None:
            intervals.append((current_start, time))
            current_start = None

    if not intervals:
        return []

    # Expand intervals to cover the union of all overlapping bursts within them
    expanded = []
    for (start_int, end_int) in intervals:
        relevant = []
        for (bstart, bend) in burst_intervals:
            if (bstart < end_int) and (bend > start_int):
                relevant.append((bstart, bend))
        if relevant:
            new_start = min(b[0] for b in relevant)
            new_end   = max(b[1] for b in relevant)
        else:
            new_start, new_end = start_int, end_int
        expanded.append((new_start, new_end))

    # Merge any intervals that might overlap
    expanded.sort(key=lambda x: x[0])
    merged = []
    current = expanded[0]
    for i in range(1, len(expanded)):
        nxt = expanded[i]
        if nxt[0] <= current[1]:
            current = (current[0], max(current[1], nxt[1]))
        else:
            merged.append(current)
            current = nxt
    merged.append(current)

    return merged

##############################################################################
# MAIN PLOTTING + TABLE BUILDING (Single Epoch)
##############################################################################

def plot_burst_cycles(
    epoch_data, fs,
    epoch=0, channels=None, tmin=0.0, tmax=0.7,
    amp_fraction=0.0,
    amp_consistency=1.0,
    period_consistency=1.0,
    monotonicity=0.4,
    min_n_cycles=3,
    freq_min=10.0,
    freq_max=18.0,
    min_phase_duration=0.0,
    amp_min=None, amp_max=None,
    page=0,
    channels_per_page=4,
    min_shared_channels=1,
    show_longest_burst=True,
    channel_names=None,
    topo_positions=None,
    show_amplitude=True,
    show_frequency=True,
    skip_plot=False,
    cycle_param='frequency'
):
    """
    This function detects bursts (ByCycle style) and optionally plots them.

    We store new columns:
      MeanAmpCons, StdAmpCons, MeanPerCons, StdPerCons, MeanMono, StdMono.
    """

    if channels is None:
        channels = list(range(epoch_data.shape[1]))
    s_start, s_end = int(tmin * fs), int(tmax * fs)
    time = np.arange(s_start, s_end) / fs

    # ----------------------------------------------------------------------
    # 1) Cycle Detection with ByCycle logic
    # ----------------------------------------------------------------------

    def run_full_bycycle_detection(signal):
        # Identify troughs / peaks
        d = np.diff(signal)
        troughs = [i+1 for i in range(len(d)-1) if d[i]<0 and d[i+1]>0]
        peaks   = [i+1 for i in range(len(d)-1) if d[i]>0 and d[i+1]<0]

        cycles = []
        for i in range(len(troughs)-1):
            t1 = troughs[i]
            t2 = troughs[i+1]
            in_peaks = [p_ for p_ in peaks if t1<p_<t2]
            if not in_peaks:
                continue
            p_ = max(in_peaks, key=lambda x: signal[x])

            rise_amp  = signal[p_] - signal[t1]
            decay_amp = signal[p_] - signal[t2]
            amp_      = signal[p_] - min(signal[t1], signal[t2])
            per_      = (t2 - t1)/fs
            freq_     = 1.0/per_ if per_>0 else np.nan
            in_band   = (freq_min <= freq_ <= freq_max)

            cycles.append({
                'trough_start': t1,
                'peak_index': p_,
                'trough_end': t2,
                'rise_amp': rise_amp,
                'decay_amp': decay_amp,
                'amp': amp_,
                'period': per_,
                'frequency': freq_,
                'is_valid': in_band,
                'monotonicity_byc': np.nan,
                'amp_cons': np.nan,
                'period_cons': np.nan,
            })

        df_ = pd.DataFrame(cycles)
        if df_.empty:
            return df_

        # 2. Compute monotonicity
        def compute_monotonicity(sig, t1, p_, t2):
            n = len(sig)
            t1 = max(0, min(t1, n-1))
            p_ = max(0, min(p_,  n-1))
            t2 = max(0, min(t2, n-1))
            if t2 <= t1:
                return np.nan

            # "rise_mon" is the fraction of strictly ascending steps in the rise
            # "decay_mon" is fraction of strictly descending steps in the decay
            if p_ <= t1:
                rise_mon = np.nan
            else:
                rise_seg = sig[t1:p_+1]
                if len(rise_seg)<2:
                    rise_mon = np.nan
                else:
                    rdif = np.diff(rise_seg)
                    rise_mon = np.mean(rdif > 0)

            if t2 <= p_:
                decay_mon = np.nan
            else:
                decay_seg = sig[p_:t2+1]
                if len(decay_seg)<2:
                    decay_mon = np.nan
                else:
                    ddif = np.diff(decay_seg)
                    decay_mon = np.mean(ddif < 0)

            if np.isnan(rise_mon) and np.isnan(decay_mon):
                return np.nan
            elif np.isnan(rise_mon):
                return decay_mon
            elif np.isnan(decay_mon):
                return rise_mon
            else:
                return 0.5*(rise_mon + decay_mon)

        for idx,row in df_.iterrows():
            if not row['is_valid']:
                continue
            t1 = int(row['trough_start'])
            p_ = int(row['peak_index'])
            t2 = int(row['trough_end'])
            mono = compute_monotonicity(signal, t1, p_, t2)
            df_.at[idx,'monotonicity_byc'] = mono

        # 3. amplitude & period consistency
        idxs = df_.index.to_list()
        for i, idx in enumerate(idxs):
            row = df_.loc[idx]
            rise_i  = row['rise_amp']
            decay_i = row['decay_amp']
            per_i   = row['period']
            prev_idx = idxs[i-1] if i>0 else None
            next_idx = idxs[i+1] if i+1<len(idxs) else None

            ratios_amp = []
            if rise_i>0 and decay_i>0:
                ratios_amp.append(min(rise_i, decay_i)/max(rise_i, decay_i))
            if prev_idx is not None:
                decay_prev = df_.loc[prev_idx, 'decay_amp']
                if decay_prev>0 and rise_i>0:
                    ratios_amp.append(min(decay_prev, rise_i)/max(decay_prev, rise_i))
            if next_idx is not None:
                rise_next = df_.loc[next_idx, 'rise_amp']
                if rise_next>0 and decay_i>0:
                    ratios_amp.append(min(rise_next, decay_i)/max(rise_next, decay_i))

            amp_cons_val = min(ratios_amp) if ratios_amp else np.nan

            ratios_per = []
            if prev_idx is not None:
                per_prev = df_.loc[prev_idx, 'period']
                if per_prev>0 and per_i>0:
                    ratios_per.append(min(per_prev, per_i)/max(per_prev, per_i))
            if next_idx is not None:
                per_next = df_.loc[next_idx, 'period']
                if per_next>0 and per_i>0:
                    ratios_per.append(min(per_next, per_i)/max(per_next, per_i))

            per_cons_val = min(ratios_per) if ratios_per else np.nan

            df_.at[idx,'amp_cons']    = amp_cons_val
            df_.at[idx,'period_cons'] = per_cons_val

        # 4. threshold checks
        for idx,row in df_.iterrows():
            if not row['is_valid']:
                continue
            if row['amp'] < amp_fraction:
                df_.at[idx, 'is_valid'] = False
                continue

        for idx,row in df_.iterrows():
            if not row['is_valid']:
                continue
            if pd.isna(row['amp_cons']) or (row['amp_cons'] < amp_consistency):
                df_.at[idx,'is_valid'] = False
                continue
            if pd.isna(row['period_cons']) or (row['period_cons'] < period_consistency):
                df_.at[idx,'is_valid'] = False
                continue
            if pd.isna(row['monotonicity_byc']) or (row['monotonicity_byc'] < monotonicity):
                df_.at[idx,'is_valid'] = False
                continue

        # 5. group valid cycles into bursts
        df_['burst_id'] = -1
        valid = df_[df_['is_valid']]
        if valid.empty:
            return df_

        burst_id = 0
        group = []
        v_idxs = valid.index.to_list()
        for i in range(len(v_idxs)):
            if i==0 or v_idxs[i]==v_idxs[i-1]+1:
                group.append(v_idxs[i])
            else:
                if len(group) >= min_n_cycles:
                    df_.loc[group,'burst_id'] = burst_id
                    burst_id+=1
                group = [v_idxs[i]]
        if len(group) >= min_n_cycles:
            df_.loc[group,'burst_id'] = burst_id

        return df_

    # run detection on selected channels
    all_dfs = {}
    for ch in channels:
        raw_sig = epoch_data[epoch,ch,s_start:s_end]
        tmpdf = run_full_bycycle_detection(raw_sig)
        tmpdf['channel'] = ch
        all_dfs[ch] = tmpdf

    n_channels = len(channels)
    if not skip_plot:
        if topo_positions is not None:
            fig = plt.figure(figsize=(16, 2.2*n_channels))
            gs = gridspec.GridSpec(nrows=n_channels, ncols=2, width_ratios=[3,1])
            axs = [fig.add_subplot(gs[i,0]) for i in range(n_channels)]
            ax_topo = fig.add_subplot(gs[:,1])
        else:
            fig, axs = plt.subplots(n_channels,1, figsize=(14, 2.2*n_channels), sharex=True)
            if n_channels==1:
                axs = [axs]

    else:
        fig, axs = None, []

    bursts_dict = defaultdict(list)
    real_burst_intervals = {}

    # gather range for param color-coding
    def get_param_range(cycles_df, param):
        if param == 'frequency':
            return freq_min, freq_max
        elif param == 'amp_cons':
            return 0.0, 1.0
        elif param == 'period_cons':
            return 0.0, 1.0
        elif param == 'monotonicity_byc':
            return 0.0, 1.0
        elif param == 'amp':
            if cycles_df.empty:
                return 0.0, 1.0
            amin_ = cycles_df['amp'].min()
            amax_ = cycles_df['amp'].max()
            if amin_ == amax_:
                amax_ = amin_ + 1e-6
            return amin_, amax_
        else:
            return 0.0, 1.0

    all_valid_cycles = pd.concat([df_ for df_ in all_dfs.values()]) if len(all_dfs)>0 else pd.DataFrame()
    if not all_valid_cycles.empty:
        all_valid_cycles = all_valid_cycles[all_valid_cycles['is_valid']]
    param_min, param_max = get_param_range(all_valid_cycles, cycle_param)
    if param_min>=param_max:
        param_min, param_max = 0.0, 1.0
    cmap_ = cm.get_cmap("viridis")
    norm_ = plt.Normalize(vmin=param_min, vmax=param_max)

    # Build bursts data per channel
    for i,ch in enumerate(channels):
        dfch = all_dfs[ch]
        if not skip_plot:
            ax_ = axs[i]
            s_  = epoch_data[epoch,ch,s_start:s_end]
            ax_.plot(time, s_, color='black', lw=1)
            if (amp_min is not None) and (amp_max is not None):
                ax_.set_ylim(amp_min, amp_max)

        ch_label = (channel_names[ch] if (channel_names is not None and ch<len(channel_names)) 
                    else f"Channel {ch}")

        valid_ = dfch[dfch['is_valid']]
        if not skip_plot:
            for idx_ in valid_.index:
                row_ = valid_.loc[idx_]
                t1_  = row_['trough_start']/fs
                t2_  = row_['trough_end']/fs
                if cycle_param in row_:
                    paramVal = row_[cycle_param]
                else:
                    paramVal = row_.get('amp', np.nan)
                if pd.isna(paramVal):
                    paramVal = param_min

                c_col = cmap_(norm_(paramVal))
                axs[i].axvspan(t1_, t2_, color=c_col, alpha=0.3)

                # decimal logic for labeling
                if cycle_param in ('amp_cons','period_cons','monotonicity_byc'):
                    label_str = f"{paramVal:.1f}"
                else:
                    label_str = f"{paramVal:.2f}"

                b_, top_ = axs[i].get_ylim()
                text_y = b_ + 0.05*(top_-b_)
                axs[i].text((t1_+t2_)/2, text_y,
                            label_str,
                            ha='center', va='bottom', fontsize=6, color='black')

                tr_ = int(row_['trough_start'])
                pk_ = int(row_['peak_index'])
                dc_ = int(row_['trough_end'])
                if 0<=tr_<len(s_) and 0<=pk_<len(s_) and 0<=dc_<len(s_):
                    axs[i].scatter(time[tr_], s_[tr_], color='red', s=25)
                    axs[i].scatter(time[pk_], s_[pk_], color='blue', s=25)
                    hrval_ = s_[tr_] + 0.5*(s_[pk_] - s_[tr_])
                    rise_seg = s_[tr_:pk_+1]
                    if len(rise_seg)>0:
                        r_off= np.argmin(np.abs(rise_seg - hrval_))
                        r_mid= tr_+r_off
                        if 0<=r_mid<len(s_):
                            axs[i].scatter(time[r_mid], s_[r_mid], color='green', s=25)
                    hdval_ = s_[pk_] + 0.5*(s_[dc_] - s_[pk_])
                    decay_seg = s_[pk_:dc_+1]
                    if len(decay_seg)>0:
                        d_off= np.argmin(np.abs(decay_seg - hdval_))
                        d_mid= pk_+ d_off
                        if 0<=d_mid<len(s_):
                            axs[i].scatter(time[d_mid], s_[d_mid], color='magenta', s=25)

        # Summarize each channel burst
        for b_id in dfch['burst_id'].unique():
            if b_id<0:
                continue
            sub_ = dfch[dfch['burst_id']==b_id]
            t1_  = sub_['trough_start'].min()
            t2_  = sub_['trough_end'].max()
            t1f  = t1_/fs
            t2f  = t2_/fs

            if not skip_plot:
                s_   = epoch_data[epoch,ch,s_start:s_end]
                idx1_= int(t1_) - s_start
                idx2_= int(t2_) - s_start
                if 0<=idx1_<len(s_) and 0<idx2_<=len(s_):
                    axs[i].plot(time[idx1_:idx2_], s_[idx1_:idx2_], color='blue', lw=2)
                    top_v = axs[i].get_ylim()[1]*0.95
                    medf_ = sub_['frequency'].median()
                    stdf_ = sub_['frequency'].std()
                    axs[i].hlines(top_v, t1f, t2f, color='blue', lw=2)
                    axs[i].text((t1f+t2f)/2, top_v*1.01,
                                f"{medf_:.1f} ± {stdf_:.1f}",
                                ha='center', color='black', fontsize=8)

            cyc_count      = len(sub_)
            avg_f          = sub_['frequency'].mean()
            std_f          = sub_['frequency'].std()
            avg_amp        = sub_['amp'].mean()
            std_amp        = sub_['amp'].std()
            mean_ampcons   = sub_['amp_cons'].mean()
            std_ampcons    = sub_['amp_cons'].std()
            mean_percons   = sub_['period_cons'].mean()
            std_percons    = sub_['period_cons'].std()
            mean_monotonic = sub_['monotonicity_byc'].mean()
            std_monotonic  = sub_['monotonicity_byc'].std()

            bursts_dict[ch].append({
                'ChannelIndex': ch,
                'ChannelLabel': ch_label,
                'BurstID': b_id,
                'Start_s': (tmin + t1f),
                'End_s':   (tmin + t2f),
                'Duration_s': (t2f - t1f),
                'NumCycles': cyc_count,
                'MeanFreq': avg_f,
                'StdFreq':  std_f,
                'MeanAmp':  avg_amp,
                'StdAmp':   std_amp,

                'MeanAmpCons': mean_ampcons,
                'StdAmpCons':  std_ampcons,
                'MeanPerCons': mean_percons,
                'StdPerCons':  std_percons,
                'MeanMono':    mean_monotonic,
                'StdMono':     std_monotonic,

                'IsGlobal': False,
                'GlobalID': -1,
                'NumChannelsInGlobal':1,
                'SynthEye1InGlobal':False,
                'SynthEye2InGlobal':False,
                'GlobalCoveragePct':0.0
            })

        # intervals for global detection
        if ch not in [0,17]:
            intervals_=[]
            for b_id in dfch['burst_id'].unique():
                if b_id<0:
                    continue
                s2_ = dfch[dfch['burst_id']==b_id]
                st_ = s2_['trough_start'].min()/fs
                en_ = s2_['trough_end'].max()/fs
                intervals_.append((st_,en_))
            real_burst_intervals[ch] = intervals_

        if not skip_plot:
            axs[i].set_ylabel(ch_label)
            axs[i].grid(True)

    # compute global bursts
    real_channels= [ch for ch in channels if ch not in [0,17]]
    combined_intervals = []
    for ch in real_channels:
        combined_intervals.extend(real_burst_intervals.get(ch,[]))

    shared_intervals= compute_shared_intervals(combined_intervals, threshold=min_shared_channels)

    if not skip_plot:
        for (st_,en_) in shared_intervals:
            for ax_ in axs:
                ax_.axvspan(st_, en_, color='magenta', alpha=0.2)

    all_rows=[]
    for c_ in bursts_dict:
        all_rows.extend(bursts_dict[c_])
    df_bursts= pd.DataFrame(all_rows)
    if not df_bursts.empty:
        df_bursts.sort_values(by=['ChannelIndex','Start_s'], inplace=True)

    # build global data
    glist=[]
    for gid,(gst,gen) in enumerate(shared_intervals):
        glist.append({
            'GlobalID': gid,
            'Start_s': tmin+gst,
            'End_s':   tmin+gen,
            'Duration_s': (gen-gst),
            'Channels': [],
            'SynthEye1': False,
            'SynthEye2': False,
            'MeanFreq': np.nan,
            'StdFreq':  np.nan,
            'MeanAmp':  np.nan,
            'StdAmp':   np.nan,

            'MeanAmpCons': np.nan,
            'StdAmpCons':  np.nan,
            'MeanPerCons': np.nan,
            'StdPerCons':  np.nan,
            'MeanMono':    np.nan,
            'StdMono':     np.nan,

            'TotalCycles': 0
        })

    synth_burst_map= {0:[], 17:[]}
    for ch_syn in [0,17]:
        if ch_syn in all_dfs:
            dfx_ = all_dfs[ch_syn]
            for b_id in dfx_['burst_id'].unique():
                if b_id<0:
                    continue
                part_ = dfx_[dfx_['burst_id']==b_id]
                st_ = part_['trough_start'].min()/fs
                en_ = part_['trough_end'].max()/fs
                synth_burst_map[ch_syn].append((st_, en_))

    for gb in glist:
        gst2= gb['Start_s']
        gen2= gb['End_s']
        involved=[]
        freqvals=[]
        ampvals=[]
        ampconsvals=[]
        perconsvals=[]
        monvals=[]
        cyc_count=0
        for idx,row in df_bursts.iterrows():
            cxx= row['ChannelIndex']
            if cxx in [0,17]:
                continue
            if (row['Start_s']< gen2) and (row['End_s']>gst2):
                if cxx not in involved:
                    involved.append(cxx)
                freqvals.append(row['MeanFreq'])
                ampvals.append(row['MeanAmp'])
                ampconsvals.append(row['MeanAmpCons'])
                perconsvals.append(row['MeanPerCons'])
                monvals.append(row['MeanMono'])
                cyc_count+= row['NumCycles']
        gb['Channels']= involved
        gb['TotalCycles']= cyc_count
        if freqvals:
            gb['MeanFreq']= np.mean(freqvals)
            gb['StdFreq'] = np.std(freqvals)
        if ampvals:
            gb['MeanAmp']= np.mean(ampvals)
            gb['StdAmp'] = np.std(ampvals)
        if ampconsvals:
            gb['MeanAmpCons']= np.mean(ampconsvals)
            gb['StdAmpCons'] = np.std(ampconsvals)
        if perconsvals:
            gb['MeanPerCons']= np.mean(perconsvals)
            gb['StdPerCons'] = np.std(perconsvals)
        if monvals:
            gb['MeanMono']= np.mean(monvals)
            gb['StdMono'] = np.std(monvals)

        for ch_syn in [0,17]:
            if ch_syn in synth_burst_map:
                local_st= gst2 - tmin
                local_en= gen2 - tmin
                for (stx, enx) in synth_burst_map[ch_syn]:
                    if (stx< local_en) and (enx> local_st):
                        if ch_syn==0:
                            gb['SynthEye1']= True
                        else:
                            gb['SynthEye2']= True
                        break

    for gb in glist:
        gid_ = gb['GlobalID']
        gst3= gb['Start_s']
        gen3= gb['End_s']
        glen= (gen3-gst3) if (gen3>gst3) else 1e-12
        chs_= gb['Channels']
        for idx,row in df_bursts.iterrows():
            if row['ChannelIndex'] not in chs_:
                continue
            if (row['Start_s']< gen3) and (row['End_s']>gst3):
                df_bursts.at[idx,'IsGlobal']=True
                df_bursts.at[idx,'GlobalID']=gid_
                df_bursts.at[idx,'NumChannelsInGlobal']= len(chs_)
                if gb['SynthEye1']:
                    df_bursts.at[idx,'SynthEye1InGlobal']=True
                if gb['SynthEye2']:
                    df_bursts.at[idx,'SynthEye2InGlobal']=True
                overlap_ = max(0, min(row['End_s'], gen3) - max(row['Start_s'],gst3))
                cov_ = (overlap_/glen)*100.0
                df_bursts.at[idx,'GlobalCoveragePct']= cov_

    global_data=[]
    for gb in glist:
        global_data.append({
            'GlobalID': gb['GlobalID'],
            'Start_s': gb['Start_s'],
            'End_s':   gb['End_s'],
            'Duration_s': gb['Duration_s'],
            'NumChannels': len(gb['Channels']),
            'Channels': ','.join(str(c) for c in gb['Channels']),
            'SynthEye1Overlap': gb['SynthEye1'],
            'SynthEye2Overlap': gb['SynthEye2'],
            'MeanFreq': gb['MeanFreq'],
            'StdFreq':  gb['StdFreq'],
            'MeanAmp':  gb['MeanAmp'],
            'StdAmp':   gb['StdAmp'],

            'MeanAmpCons': gb['MeanAmpCons'],
            'StdAmpCons':  gb['StdAmpCons'],
            'MeanPerCons': gb['MeanPerCons'],
            'StdPerCons':  gb['StdPerCons'],
            'MeanMono':    gb['MeanMono'],
            'StdMono':     gb['StdMono'],

            'TotalCycles': gb['TotalCycles'],
        })
    df_global= pd.DataFrame(global_data)
    if not df_global.empty:
        df_global.sort_values(by='GlobalID', inplace=True)

    df_bursts['Eye1DistantOverlap']=False
    df_bursts['Eye1NearOverlap']   =False
    df_bursts['Eye2DistantOverlap']=False
    df_bursts['Eye2NearOverlap']   =False

    def intervals_overlap(a1,a2,b1,b2):
        return (a1<b2) and (a2>b1)

    channel_bursts_map= defaultdict(list)
    for ix,row in df_bursts.iterrows():
        channel_bursts_map[row['ChannelIndex']].append((row['Start_s'],row['End_s'], ix))

    pairs={
        'Eye1Distant': (3,5),
        'Eye1Near':    (4,6),
        'Eye2Distant': (11,13),
        'Eye2Near':    (12,14)
    }
    for coln,(chA,chB) in pairs.items():
        if chA not in channel_bursts_map and chB not in channel_bursts_map:
            continue
        a_bursts = channel_bursts_map.get(chA,[])
        b_bursts = channel_bursts_map.get(chB,[])
        for (a_s,a_e,a_ix) in a_bursts:
            found_overlap=False
            for (b_s,b_e,b_ix) in b_bursts:
                if intervals_overlap(a_s,a_e,b_s,b_e):
                    found_overlap=True
                    break
            if found_overlap:
                df_bursts.at[a_ix, coln+'Overlap']=True
        for (b_s,b_e,b_ix) in b_bursts:
            found_overlap=False
            for (a_s,a_e,a_ix) in a_bursts:
                if intervals_overlap(b_s,b_e,a_s,a_e):
                    found_overlap=True
                    break
            if found_overlap:
                df_bursts.at[b_ix, coln+'Overlap']=True

    if not skip_plot:
        pd.set_option('display.max_rows', None)
        print("\n================= ALL BURSTS (PER CHANNEL) =================")
        if df_bursts.empty:
            print("No single-channel bursts found.")
        else:
            display(df_bursts)

        print("\n================= GLOBAL BURSTS (AGGREGATED) ===============")
        if df_global.empty:
            print("No global bursts found for current threshold.")
        else:
            display(df_global)

        if topo_positions is not None:
            if 0 in channels:
                real_for_topo=[c for c in channels if c!=0]
            elif 17 in channels:
                real_for_topo=[c for c in channels if c!=17]
            else:
                real_for_topo= channels
            plot_topographic(ax_topo, topo_positions, channel_names, real_for_topo)
            axs[-1].set_xlabel("Time (s)")
        else:
            axs[-1].set_xlabel("Time (s)")

        plt.suptitle(f"Burst Visualization – Epoch {epoch}, {tmin:.1f}–{tmax:.1f}s", fontsize=14)
        plt.tight_layout()
        plt.show()

    return df_bursts, df_global

##############################################################################
# INTERACTIVE UI
##############################################################################

class BurstAnalysisApp:
    def __init__(self):
        self.epoch_data = None
        self.fs = None
        self.channel_names = None

        self.last_df_bursts = None
        self.last_df_global = None

        self.fif_path_text = Text(
            value="D:\\path\\to\\your_file.fif",
            description="FIF Path:",
            layout={'width': '700px'}
        )
        self.load_data_button = Button(description="Load Data", button_style='primary')
        self.load_output = Output()

        self.epoch_slider = IntSlider(value=0, min=0, max=0, description='Epoch:')
        self.channel_group_dropdown = Dropdown(
            options={
                'Eye 1 (Synthetic + Ch 1-8)': [0] + list(range(1, 9)),
                'Eye 2 (Ch 9-16 + Synthetic)': list(range(9, 18))
            },
            value=[0] + list(range(1, 9)),
            description='Channels:'
        )
        self.display_mode_dropdown = Dropdown(
            options=['paged', 'all'],
            value='paged',
            description='Display:'
        )
        self.topo_mode_dropdown = Dropdown(
            options=['Combined', 'Separated'],
            value='Combined',
            description='Topo Mode:'
        )
        self.tmin_slider = FloatSlider(value=0.0, min=0.0, max=5.0, step=0.1, description='Tmin (s):')
        self.tmax_slider = FloatSlider(value=5.0, min=0.1, max=5.0, step=0.1, description='Tmax (s):')

        self.amp_min_slider = FloatSlider(value=-100, min=-1000, max=0, step=10, description='Amp Min:')
        self.amp_max_slider = FloatSlider(value=100, min=0, max=1000, step=10, description='Amp Max:')

        self.amp_fraction_slider = FloatSlider(value=0.0, min=0.0, max=3.0, step=0.05, description='Amp Frac:')
        self.amp_consistency_slider = FloatSlider(value=1.0, min=0.0, max=1.0, step=0.05, description='Amp Cons:')
        self.period_consistency_slider = FloatSlider(value=1.0, min=0.0, max=1.0, step=0.05, description='Period Cons:')
        self.monotonicity_slider = FloatSlider(value=0.4, min=0.0, max=1.0, step=0.05, description='Monotonicity:')
        self.min_cycles_slider = IntSlider(value=3, min=1, max=10, step=1, description='Min Cycles:')
        self.freq_min_slider = FloatSlider(value=10.0, min=1.0, max=30.0, step=0.5, description='Freq Min:')
        self.freq_max_slider = FloatSlider(value=18.0, min=1.0, max=30.0, step=0.5, description='Freq Max:')
        self.phase_dur_slider = FloatSlider(value=0.0, min=0.0, max=1.0, step=0.05, description='Min Phase:')
        self.min_shared_channels_slider = IntSlider(value=8, min=1, max=8, step=1, description='Shared Chs:')

        self.show_amplitude_toggle = Checkbox(value=True, description='Show Amplitude')
        self.show_frequency_toggle = Checkbox(value=True, description='Show Frequency')

        self.page_slider = IntSlider(value=0, min=0, max=0, description='Page:')
        self.plot_button = Button(description="Plot Bursts", button_style='primary')
        self.plot_output = Output()

        self.excel_export_text = Text(
            value="C:/temp/burst_export.xlsx",
            description="Excel Export Path:",
            layout={'width': '600px'}
        )
        self.export_button = Button(description="Export to Excel", button_style='success')
        self.export_output = Output()

        self.process_all_button = Button(description="Process All Epochs & Export", button_style='success')
        self.all_export_output = Output()

        self.cycle_param_dropdown = Dropdown(
            options=[
                ('Frequency (Hz)', 'frequency'),
                ('Amplitude', 'amp'),
                ('Amplitude Consistency', 'amp_cons'),
                ('Period Consistency', 'period_cons'),
                ('Monotonicity', 'monotonicity_byc')
            ],
            value='frequency',
            description='Cycle Param:'
        )

        # Bind events
        self.load_data_button.on_click(self.on_load_data_clicked)
        self.plot_button.on_click(self.update_plot)
        self.export_button.on_click(self.on_export_button_clicked)
        self.process_all_button.on_click(self.process_all_epochs_and_export)

        self.epoch_slider.observe(self.update_plot, names="value")
        self.page_slider.observe(self.update_plot, names="value")
        self.channel_group_dropdown.observe(self.update_plot, names="value")
        self.display_mode_dropdown.observe(self.update_plot, names="value")
        self.topo_mode_dropdown.observe(self.update_plot, names="value")
        self.amp_min_slider.observe(self.update_plot, names="value")
        self.amp_max_slider.observe(self.update_plot, names="value")
        self.show_amplitude_toggle.observe(self.update_plot, names="value")
        self.show_frequency_toggle.observe(self.update_plot, names="value")
        self.cycle_param_dropdown.observe(self.update_plot, names="value")

    def on_load_data_clicked(self, b):
        with self.load_output:
            self.load_output.clear_output()
            print("Loading data from:", self.fif_path_text.value)
            try:
                epoch_data, fs = load_epochs(self.fif_path_text.value)
                epoch_data = add_synthetic_channels(epoch_data)
                channel_names = create_channel_names()
                self.epoch_data = epoch_data
                self.fs = fs
                self.channel_names = channel_names
                self.epoch_slider.max = epoch_data.shape[0] - 1
                self.epoch_slider.value = 0
                print("Data loaded successfully!")
            except Exception as e:
                print("Failed to load data:", e)

    def update_plot(self, *args):
        if self.epoch_data is None:
            return
        with self.plot_output:
            self.plot_output.clear_output()

            channels = self.channel_group_dropdown.value
            mode = self.display_mode_dropdown.value
            if mode == 'paged':
                n_pages = (len(channels) - 1) // 4
                self.page_slider.max = n_pages
                start = self.page_slider.value * 4
                selected_channels = channels[start:start+4]
            else:
                self.page_slider.max=0
                self.page_slider.value=0
                selected_channels = channels

            topo_dict = (channel_positions_combined
                         if self.topo_mode_dropdown.value=='Combined'
                         else channel_positions_separated)

            cycle_param_selected = self.cycle_param_dropdown.value

            df_bursts, df_global = plot_burst_cycles(
                self.epoch_data, self.fs,
                epoch=self.epoch_slider.value,
                channels=selected_channels,
                tmin=self.tmin_slider.value,
                tmax=self.tmax_slider.value,
                amp_fraction=self.amp_fraction_slider.value,
                amp_consistency=self.amp_consistency_slider.value,
                period_consistency=self.period_consistency_slider.value,
                monotonicity=self.monotonicity_slider.value,
                min_n_cycles=self.min_cycles_slider.value,
                freq_min=self.freq_min_slider.value,
                freq_max=self.freq_max_slider.value,
                min_phase_duration=self.phase_dur_slider.value,
                amp_min=self.amp_min_slider.value,
                amp_max=self.amp_max_slider.value,
                min_shared_channels=self.min_shared_channels_slider.value,
                show_longest_burst=True,
                page=self.page_slider.value,
                channels_per_page=len(selected_channels),
                channel_names=self.channel_names,
                topo_positions=topo_dict,
                show_amplitude=self.show_amplitude_toggle.value,
                show_frequency=self.show_frequency_toggle.value,
                skip_plot=False,
                cycle_param=cycle_param_selected
            )

            self.last_df_bursts = df_bursts
            self.last_df_global = df_global

    def on_export_button_clicked(self, b):
        """
        Export single-epoch results to Excel with 4 sheets:
            1) Eye1 bursts
            2) Eye2 bursts
            3) Global bursts
            4) Details sheet, with separate global counts for Eye1 vs Eye2.
        """
        with self.export_output:
            self.export_output.clear_output()
            if self.last_df_bursts is None or self.last_df_global is None:
                print("No data available to export. Please plot first.")
                return

            path = self.excel_export_text.value
            print(f"Exporting SINGLE-EPOCH results => {path}")

            df_bursts = self.last_df_bursts
            df_global = self.last_df_global

            # Separate eye1 vs eye2 *channel* bursts
            df_eye1 = df_bursts[
                (df_bursts['ChannelIndex']>=0)&
                (df_bursts['ChannelIndex']<=8)
            ].copy()
            df_eye2 = df_bursts[
                (df_bursts['ChannelIndex']>=9)&
                (df_bursts['ChannelIndex']<=17)
            ].copy()

            # --- Separate out Eye-1 vs Eye-2 "global bursts" (sheet 4 wants them distinct).
            # We'll define an Eye‐1 global as one whose Channels are all in 1..8,
            # an Eye‐2 global as one whose Channels are all in 9..16.
            # Mixed bursts won't appear in either category.
            def is_eye1_only(ch_str):
                if not ch_str.strip():
                    return False
                chlist = [int(x.strip()) for x in ch_str.split(',')]
                return all((0 < c <= 8) for c in chlist)  # ignoring synthetic ch=0

            def is_eye2_only(ch_str):
                if not ch_str.strip():
                    return False
                chlist = [int(x.strip()) for x in ch_str.split(',')]
                return all((9 <= c <= 16) for c in chlist)

            df_global_eye1 = df_global[df_global['Channels'].apply(is_eye1_only)]
            df_global_eye2 = df_global[df_global['Channels'].apply(is_eye2_only)]

            num_global_eye1 = len(df_global_eye1)
            num_global_eye2 = len(df_global_eye2)

            # Build the "Details" sheet
            details_data = {
                'AmpFraction': [self.amp_fraction_slider.value],
                'AmpConsistency': [self.amp_consistency_slider.value],
                'PeriodConsistency': [self.period_consistency_slider.value],
                'Monotonicity': [self.monotonicity_slider.value],
                'MinCycles': [self.min_cycles_slider.value],
                'FreqMin': [self.freq_min_slider.value],
                'FreqMax': [self.freq_max_slider.value],
                'SharedChannels': [self.min_shared_channels_slider.value],
                'TotalEpochs': [self.epoch_data.shape[0]],

                'TotalEye1Bursts': [len(df_eye1)],
                'TotalEye2Bursts': [len(df_eye2)],

                # Now we have separate Eye1/Eye2 global counts:
                'GlobalBurstsEye1': [num_global_eye1],
                'GlobalBurstsEye2': [num_global_eye2],

                # (Optional) the total "df_global" as well
                'GlobalBurstsTOTAL': [len(df_global)]
            }
            df_details = pd.DataFrame(details_data)

            try:
                with pd.ExcelWriter(path) as writer:
                    df_eye1.to_excel(writer, sheet_name='Eye1', index=False)
                    df_eye2.to_excel(writer, sheet_name='Eye2', index=False)
                    df_global.to_excel(writer, sheet_name='Global', index=False)
                    df_details.to_excel(writer, sheet_name='Details', index=False)

                print("Export successful!")
            except Exception as e:
                print("Export failed:", e)

    def process_all_epochs_and_export(self, b):
        """
        Process ALL epochs for Eye1 + Eye2, then export a 4-sheet Excel file:
            1) Eye1 (all bursts from all epochs)
            2) Eye2 (all bursts from all epochs)
            3) Global (all global bursts from all epochs)
            4) Details (including separate Eye1/Eye2 global counts).
        """
        with self.all_export_output:
            self.all_export_output.clear_output()
            if self.epoch_data is None:
                print("No data loaded. Please load first.")
                return

            n_epochs= self.epoch_data.shape[0]
            eye1_channels= list(range(0,9))
            eye2_channels= list(range(9,18))

            all_bursts_eye1= []
            all_global_eye1= []
            all_bursts_eye2= []
            all_global_eye2= []

            total_steps= 2*n_epochs
            step_count= 0

            tmin_= self.tmin_slider.value
            tmax_= self.tmax_slider.value
            ampfrac_= self.amp_fraction_slider.value
            ampcons_= self.amp_consistency_slider.value
            percons_= self.period_consistency_slider.value
            monot_= self.monotonicity_slider.value
            mincycles_= self.min_cycles_slider.value
            freqmin_= self.freq_min_slider.value
            freqmax_= self.freq_max_slider.value
            phasedur_= self.phase_dur_slider.value
            ampmin_= self.amp_min_slider.value
            ampmax_= self.amp_max_slider.value
            shared_= self.min_shared_channels_slider.value
            showAmp_= self.show_amplitude_toggle.value
            showFreq_= self.show_frequency_toggle.value

            cycle_param_all= self.cycle_param_dropdown.value
            ch_names= self.channel_names

            nfound_bursts_eye1= 0
            nfound_global_eye1= 0
            nfound_bursts_eye2= 0
            nfound_global_eye2= 0

            for e_ in range(n_epochs):
                # Eye1
                step_count+=1
                dfb1, dfg1= plot_burst_cycles(
                    self.epoch_data, self.fs,
                    epoch=e_,
                    channels=eye1_channels,
                    tmin=tmin_, tmax=tmax_,
                    amp_fraction=ampfrac_,
                    amp_consistency=ampcons_,
                    period_consistency=percons_,
                    monotonicity=monot_,
                    min_n_cycles=mincycles_,
                    freq_min=freqmin_, freq_max=freqmax_,
                    min_phase_duration=phasedur_,
                    amp_min=ampmin_, amp_max=ampmax_,
                    min_shared_channels=shared_,
                    show_longest_burst=True,
                    page=0,
                    channels_per_page=len(eye1_channels),
                    channel_names=ch_names,
                    topo_positions=None,
                    show_amplitude=showAmp_,
                    show_frequency=showFreq_,
                    skip_plot=True,
                    cycle_param=cycle_param_all
                )
                if not dfb1.empty:
                    dfb1['Epoch']= e_
                    dfb1['Eye']  = 'Eye1'
                    all_bursts_eye1.append(dfb1)
                    nfound_bursts_eye1+= len(dfb1)
                if not dfg1.empty:
                    dfg1['Epoch']= e_
                    dfg1['Eye']  = 'Eye1'
                    all_global_eye1.append(dfg1)
                    nfound_global_eye1+= len(dfg1)

                pc_ = 100.0*step_count/total_steps
                print(f"Epoch {e_+1}/{n_epochs}, Eye1 => {len(dfb1)} bursts, {len(dfg1)} global. {pc_:.1f}% done",
                      end='\r')

                # Eye2
                step_count+=1
                dfb2, dfg2= plot_burst_cycles(
                    self.epoch_data, self.fs,
                    epoch=e_,
                    channels=eye2_channels,
                    tmin=tmin_, tmax=tmax_,
                    amp_fraction=ampfrac_,
                    amp_consistency=ampcons_,
                    period_consistency=percons_,
                    monotonicity=monot_,
                    min_n_cycles=mincycles_,
                    freq_min=freqmin_, freq_max=freqmax_,
                    min_phase_duration=phasedur_,
                    amp_min=ampmin_, amp_max=ampmax_,
                    min_shared_channels=shared_,
                    show_longest_burst=True,
                    page=0,
                    channels_per_page=len(eye2_channels),
                    channel_names=ch_names,
                    topo_positions=None,
                    show_amplitude=showAmp_,
                    show_frequency=showFreq_,
                    skip_plot=True,
                    cycle_param=cycle_param_all
                )
                if not dfb2.empty:
                    dfb2['Epoch']= e_
                    dfb2['Eye']  = 'Eye2'
                    all_bursts_eye2.append(dfb2)
                    nfound_bursts_eye2+= len(dfb2)
                if not dfg2.empty:
                    dfg2['Epoch']= e_
                    dfg2['Eye']  = 'Eye2'
                    all_global_eye2.append(dfg2)
                    nfound_global_eye2+= len(dfg2)

                pc_ = 100.0*step_count/total_steps
                print(f"Epoch {e_+1}/{n_epochs}, Eye2 => {len(dfb2)} bursts, {len(dfg2)} global. {pc_:.1f}% done",
                      end='\r')

            print("")  # newline after progress

            if len(all_bursts_eye1)>0:
                df_eye1_bursts = pd.concat(all_bursts_eye1, ignore_index=True)
            else:
                df_eye1_bursts = pd.DataFrame()
            if len(all_global_eye1)>0:
                df_eye1_global = pd.concat(all_global_eye1, ignore_index=True)
            else:
                df_eye1_global = pd.DataFrame()

            if len(all_bursts_eye2)>0:
                df_eye2_bursts = pd.concat(all_bursts_eye2, ignore_index=True)
            else:
                df_eye2_bursts = pd.DataFrame()
            if len(all_global_eye2)>0:
                df_eye2_global = pd.concat(all_global_eye2, ignore_index=True)
            else:
                df_eye2_global= pd.DataFrame()

            df_global_all= pd.concat([df_eye1_global, df_eye2_global], ignore_index=True)

            path_ = self.excel_export_text.value
            print(f"\nWriting ALL-EPOCH results to: {path_}")

            # Separate counts for Eye1/Eye2 global
            count_eye1_global = len(df_eye1_global)
            count_eye2_global = len(df_eye2_global)

            details_data = {
                'AmpFraction': [self.amp_fraction_slider.value],
                'AmpConsistency': [self.amp_consistency_slider.value],
                'PeriodConsistency': [self.period_consistency_slider.value],
                'Monotonicity': [self.monotonicity_slider.value],
                'MinCycles': [self.min_cycles_slider.value],
                'FreqMin': [self.freq_min_slider.value],
                'FreqMax': [self.freq_max_slider.value],
                'SharedChannels': [self.min_shared_channels_slider.value],
                'TotalEpochs': [n_epochs],
                'TotalEye1Bursts': [len(df_eye1_bursts)],
                'TotalEye2Bursts': [len(df_eye2_bursts)],
                'GlobalBurstsEye1': [count_eye1_global],
                'GlobalBurstsEye2': [count_eye2_global],
                'GlobalBurstsTOTAL': [len(df_global_all)]
            }
            df_details = pd.DataFrame(details_data)

            try:
                with pd.ExcelWriter(path_) as writer:
                    df_eye1_bursts.to_excel(writer, sheet_name='Eye1', index=False)
                    df_eye2_bursts.to_excel(writer, sheet_name='Eye2', index=False)
                    df_global_all.to_excel(writer, sheet_name='Global', index=False)
                    df_details.to_excel(writer, sheet_name='Details', index=False)

                print("ALL-EPOCH export successful!")
            except Exception as e:
                print("Export failed:", e)

            print(f"Total Eye1 bursts found: {nfound_bursts_eye1}, Eye1 global bursts: {nfound_global_eye1}")
            print(f"Total Eye2 bursts found: {nfound_bursts_eye2}, Eye2 global bursts: {nfound_global_eye2}")

    def display(self):
        load_box = VBox([
            HBox([Label("Enter FIF file path:"), self.fif_path_text]),
            self.load_data_button,
            self.load_output
        ])

        plot_controls = VBox([
            HBox([
                self.epoch_slider, self.channel_group_dropdown,
                self.display_mode_dropdown, self.topo_mode_dropdown,
                self.page_slider
            ]),
            HBox([self.tmin_slider, self.tmax_slider]),
            HBox([self.amp_min_slider, self.amp_max_slider]),
            HBox([self.cycle_param_dropdown]),
            HBox([self.amp_fraction_slider, self.amp_consistency_slider]),
            HBox([self.period_consistency_slider, self.monotonicity_slider]),
            HBox([self.min_cycles_slider, self.freq_min_slider, self.freq_max_slider]),
            HBox([self.phase_dur_slider, self.min_shared_channels_slider]),
            HBox([self.show_amplitude_toggle, self.show_frequency_toggle]),
            self.plot_button,
            self.plot_output
        ])

        export_box_single = VBox([
            HBox([Label("Excel Path:"), self.excel_export_text]),
            self.export_button,
            self.export_output
        ])

        export_box_all = VBox([
            self.process_all_button,
            self.all_export_output
        ])

        display(load_box, plot_controls, export_box_single, export_box_all)

def main():
    app = BurstAnalysisApp()
    app.display()

if __name__=="__main__":
    main()
