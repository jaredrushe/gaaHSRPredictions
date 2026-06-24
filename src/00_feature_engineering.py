# 00_feature_engineering.py
# Run first -- generates all data files needed by downstream scripts.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from config import (DATA_PATH, OUTPUTS_DIR, Q_SPLIT_SECONDS,
                    COVERAGE_THRESHOLD, MZ_THRESHOLD,
                    HSR_ACTIONS, ACTION_ZONES,
                    WINDOW_S, STEP_S, N_STEPS)
from utils import update_stat_rows, ratio_to_label

df = pd.read_csv(DATA_PATH)
print(f"Raw: {len(df):,} rows | {df['GameID'].nunique()} games | "
      f"{df['PlayerID'].nunique()} players")

# Game boundaries
h2s = df[df['Half']==2].groupby('GameID')['Start_Second'].min().rename('h2_start')
ge  = df.groupby('GameID')['End_Second'].max().rename('game_end')
gs  = df.groupby('GameID')['Start_Second'].min().rename('game_start')
gb  = pd.concat([gs, h2s, ge], axis=1).reset_index()
gb['q4_cut']       = gb['h2_start'] + Q_SPLIT_SECONDS
gb['q4_end']       = gb['game_end']
gb['game_duration'] = gb['game_end'] - gb['game_start']
df = df.merge(gb[['GameID','game_duration']], on='GameID')

# Coverage filter
print("Step 1: Coverage filter")
pc = (df.groupby(['GameID','PlayerID'])
        .agg(pd_=('Duration','sum'), gd=('game_duration','first'))
        .reset_index())
pc['cov'] = pc['pd_'] / pc['gd']
eligible  = pc[pc['cov'] >= COVERAGE_THRESHOLD][['GameID','PlayerID']]
df_elig   = df.merge(eligible, on=['GameID','PlayerID'])
print(f"  Eligible: {len(eligible)} player-game combos | "
      f"{eligible['PlayerID'].nunique()} players")

# Slice helpers
def zone_slice(rows, s, e, actions):
    
    dur = e - s
    if dur <= 0: return 0.0
    mask  = (rows['Start_Second'] < e) & (rows['End_Second'] >= s)
    chunk = rows[mask & rows['Action'].isin(actions)].copy()
    if chunk.empty: return 0.0
    os_ = chunk['Start_Second'].clip(lower=s)
    oe  = chunk['End_Second'].clip(upper=e)
    ov  = (oe - os_).clip(lower=0)
    ad  = (chunk['End_Second'] - chunk['Start_Second']).replace(0, np.nan)
    return float((chunk['Distance'] * (ov / ad)).sum() / (dur / 60))

def hsr_slice(rows, s, e):
    return zone_slice(rows, s, e, HSR_ACTIONS)

# Outlier removal
print("Step 2: Outlier removal")

def modified_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0: mad = np.mean(np.abs(x - med))
    return 0.6745 * (x - med) / mad

_pg_ratios = {}
for (gid, pid), grp in df_elig.groupby(['GameID','PlayerID']):
    b      = gb[gb['GameID']==gid].iloc[0]
    gs_v   = b['game_start']; q4c = b['q4_cut']
    q1     = hsr_slice(grp, gs_v, gs_v+Q_SPLIT_SECONDS)
    q2     = hsr_slice(grp, gs_v+Q_SPLIT_SECONDS, b['h2_start'])
    q3     = hsr_slice(grp, b['h2_start'], q4c)
    q4     = hsr_slice(grp, q4c, b['q4_end'])
    mean13 = np.mean([q1, q2, q3])
    _pg_ratios[(gid, pid)] = (q4 / mean13) if mean13 > 0 else np.nan

_ratio_list   = [(k, v) for k, v in _pg_ratios.items() if not np.isnan(v)]
_mz_scores    = modified_zscore(np.array([v for _, v in _ratio_list]))
_outlier_keys = {k for (k, v), mz in zip(_ratio_list, _mz_scores)
                 if abs(mz) > MZ_THRESHOLD}

print(f"  Flagged {len(_outlier_keys)} observations:")
for k in sorted(_outlier_keys):
    mz = next(mz for (kk,_), mz in zip(_ratio_list, _mz_scores) if kk==k)
    print(f"    GameID={k[0]}  PlayerID={k[1]}  "
          f"ratio={_pg_ratios[k]:.3f}  modified_z={mz:.2f}")

eligible  = eligible[~eligible.apply(
    lambda r: (r['GameID'], r['PlayerID']) in _outlier_keys, axis=1)]
df_elig   = df.merge(eligible, on=['GameID','PlayerID'])
print(f"  Remaining: {len(eligible)} | {eligible['PlayerID'].nunique()} players")

# Per-player-game stats
game_order = sorted(df['GameID'].unique())
game_rank  = {g: i for i, g in enumerate(game_order)}
pg_stats   = {}

for (gid, pid), grp in df_elig.groupby(['GameID','PlayerID']):
    b      = gb[gb['GameID']==gid].iloc[0]
    gs_v   = b['game_start']; q4c = b['q4_cut']; q4e = b['q4_end']
    q1     = hsr_slice(grp, gs_v, gs_v+Q_SPLIT_SECONDS)
    q2     = hsr_slice(grp, gs_v+Q_SPLIT_SECONDS, b['h2_start'])
    q3     = hsr_slice(grp, b['h2_start'], q4c)
    q4     = hsr_slice(grp, q4c, q4e)
    mean13 = np.mean([q1, q2, q3])
    ratio  = (q4 / mean13) if mean13 > 0 else np.nan
    pg_stats[(gid, pid)] = {
        'q4_hsr':    q4,
        'q4_ratio':  ratio,
        'q1_3_mean': mean13,
        'game_rank': game_rank[gid],
    }

# Rolling windows + features
print("Step 3: Rolling windows + features")
print(f"  {N_STEPS} steps x {WINDOW_S//60}-min windows, "
      f"stepping back {STEP_S//60} min from Q4 cutpoint")

flat_records = []
sequences_uv = []
sequences_mv = []
seq_meta     = []
n_skipped    = 0

for (gid, pid), grp in df_elig.groupby(['GameID','PlayerID']):
    b        = gb[gb['GameID']==gid].iloc[0]
    gs_v     = b['game_start']; q4c = b['q4_cut']
    st       = pg_stats[(gid, pid)]
    q4_ratio = st['q4_ratio']
    if pd.isna(q4_ratio):
        n_skipped += 1
        continue

    label_bin = ratio_to_label(q4_ratio)  

    uv_steps, mv_steps = [], []
    for step in range(N_STEPS):
        we = q4c - step * STEP_S
        ws = max(we - WINDOW_S, gs_v)
        uv_steps.append(hsr_slice(grp, ws, we))
        mv_steps.append([zone_slice(grp, ws, we, {z}) for z in ACTION_ZONES])
    uv_steps.reverse(); mv_steps.reverse()

    s = np.array(uv_steps)
    x = np.arange(N_STEPS, dtype=float)

    flat_records.append({
        'GameID':          gid,
        'PlayerID':        pid,
        'game_rank':       game_rank[gid],
        'series_mean':     float(np.mean(s)),
        'series_std':      float(np.std(s)),
        'series_last':     float(s[-1]),
        'series_slope':    float(np.polyfit(x, s, 1)[0]),
        'last10_slope':    float(np.polyfit(x[-10:], s[-10:], 1)[0]),
        'Q4_HSR_per_min':  st['q4_hsr'],
        'Q4_label_binary': label_bin,
    })
    sequences_uv.append(uv_steps)
    sequences_mv.append(mv_steps)
    seq_meta.append({
        'GameID':          gid,
        'PlayerID':        pid,
        'Q4_HSR_per_min':  st['q4_hsr'],
        'Q4_label_binary': label_bin,
    })

flat_df     = pd.DataFrame(flat_records)
seq_arr_uv  = np.array(sequences_uv, dtype=np.float32).reshape(-1, N_STEPS, 1)
seq_arr_mv  = np.array(sequences_mv, dtype=np.float32)
seq_meta_df = pd.DataFrame(seq_meta)

print(f"  Skipped (undefined ratio): {n_skipped}")
print(f"  Rows: {len(flat_df)} | Players: {flat_df['PlayerID'].nunique()}")
print(f"  UV shape: {seq_arr_uv.shape}  MV shape: {seq_arr_mv.shape}")
print(f"  Label distribution:")
for lbl, cnt in flat_df['Q4_label_binary'].value_counts().items():
    print(f"    {lbl}: {cnt} ({cnt/len(flat_df):.1%})")
print(f"  Games per player:")
for pid, cnt in sorted(flat_df.groupby('PlayerID')['GameID'].count().items()):
    print(f"    Player {pid:>4}: {cnt} games")

# Historical features
print("Step 4: Historical features")
hist_mpm = []
for _, row in flat_df.iterrows():
    other = flat_df[(flat_df['PlayerID'] == row['PlayerID']) &
                    (flat_df['GameID']   != row['GameID'])]
    if len(other) == 0:
        hist_mpm.append(np.nan)
    else:
        hist_mpm.append(float(other['Q4_HSR_per_min'].mean()))

flat_df['hist_mean_q4_mpm'] = hist_mpm
print(f"  With hist:    {flat_df['hist_mean_q4_mpm'].notna().sum()} rows")
print(f"  Without hist: {flat_df['hist_mean_q4_mpm'].isna().sum()} rows "
      f"(single-game players)")

# Save
flat_df.to_csv(OUTPUTS_DIR + 'gaa_features_final.csv', index=False)
np.save(OUTPUTS_DIR + 'gaa_sequences.npy',    seq_arr_uv)
np.save(OUTPUTS_DIR + 'gaa_sequences_mv.npy', seq_arr_mv)
seq_meta_df.to_csv(OUTPUTS_DIR + 'gaa_sequences_meta.csv', index=False)

print("\nSaved:")
print(f"   gaa_features_final.csv  - {len(flat_df)} rows")
print(f"   gaa_sequences.npy       - {seq_arr_uv.shape}")
print(f"   gaa_sequences_mv.npy    - {seq_arr_mv.shape}")
print(f"   gaa_sequences_meta.csv  - {len(seq_meta_df)} rows")

print("Step 5: Distribution stats")
update_stat_rows(flat_df)