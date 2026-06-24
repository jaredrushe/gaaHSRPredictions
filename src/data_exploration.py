import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DATA_PATH, OUTPUTS_DIR, Q_SPLIT_SECONDS, COVERAGE_THRESHOLD

df = pd.read_csv(DATA_PATH)
print(f"Raw rows: {len(df)}")
print(df.head())

# Coverage per player-game
def coverage_stats(grp):
    start = grp['Start_Second'].min()
    end   = grp['End_Second'].max()
    dur   = grp['Duration'].sum()
    span  = end - start
    return pd.Series({'duration': dur, 'span': span,
                      'coverage': dur / span if span > 0 else 0})

cov = df.groupby(['GameID','PlayerID']).apply(coverage_stats).reset_index()
full = cov[cov['coverage'] >= COVERAGE_THRESHOLD]
df_full = df[df.set_index(['GameID','PlayerID']).index.isin(
    full.set_index(['GameID','PlayerID']).index)]

print(f"\nAfter {COVERAGE_THRESHOLD*100:.0f}% coverage filter: "
      f"{full['GameID'].nunique()} games, "
      f"{full['PlayerID'].nunique()} players, "
      f"{len(full)} player-game observations")

# Q4 HSR distribution
hsr_actions = {'Running', 'High Intensity Running', 'Sprint'}
records = []
for (gid, pid), grp in df_full.groupby(['GameID','PlayerID']):
    h2 = grp[grp['Half'] == 2]
    if h2.empty: continue
    h2_start = h2['Start_Second'].min()
    h2_mid   = h2_start + Q_SPLIT_SECONDS
    q4 = grp[(grp['Half']==2) & (grp['Start_Second'] >= h2_mid)]
    q4_dur = q4['Duration'].sum()
    if q4_dur == 0: continue
    q4_hsr = q4[q4['Action'].isin(hsr_actions)]['Distance'].sum()
    records.append({'GameID': gid, 'PlayerID': pid,
                    'Q4_HSR_mpm': q4_hsr / (q4_dur / 60)})

hsr_df = pd.DataFrame(records)
print(f"\nQ4 HSR distribution (m/min):")
print(f"  n={len(hsr_df)}")
print(f"  mean={hsr_df['Q4_HSR_mpm'].mean():.2f}")
print(f"  std={hsr_df['Q4_HSR_mpm'].std():.2f}")
print(f"  min={hsr_df['Q4_HSR_mpm'].min():.2f}  max={hsr_df['Q4_HSR_mpm'].max():.2f}")

# Q4 vs Q1-Q3 HSR trend
records2 = []
for (gid, pid), grp in df_full.groupby(['GameID','PlayerID']):
    h2 = grp[grp['Half']==2]
    if h2.empty: continue
    h2_start = h2['Start_Second'].min()
    h2_mid   = h2_start + Q_SPLIT_SECONDS
    q123 = grp[~((grp['Half']==2) & (grp['Start_Second'] >= h2_mid))]
    q4   = grp[(grp['Half']==2) & (grp['Start_Second'] >= h2_mid)]
    q123_dur = q123['Duration'].sum()
    q4_dur   = q4['Duration'].sum()
    if q123_dur == 0 or q4_dur == 0: continue
    q123_hsr = q123[q123['Action'].isin(hsr_actions)]['Distance'].sum()
    q4_hsr   = q4[q4['Action'].isin(hsr_actions)]['Distance'].sum()
    records2.append({
        'GameID': gid, 'PlayerID': pid,
        'Q1_Q3_mpm': q123_hsr / (q123_dur / 60),
        'Q4_mpm':    q4_hsr   / (q4_dur   / 60),
    })

trend_df = pd.DataFrame(records2)
trend_df['diff'] = trend_df['Q4_mpm'] - trend_df['Q1_Q3_mpm']
decreased = (trend_df['diff'] < 0).sum()
increased = (trend_df['diff'] > 0).sum()
total     = len(trend_df)
print(f"\nQ1-Q3 vs Q4 HSR trend ({total} player-games):")
print(f"  Decreased: {decreased} ({decreased/total*100:.1f}%)")
print(f"  Increased: {increased} ({increased/total*100:.1f}%)")
print(f"  Mean change: {trend_df['diff'].mean():+.2f} m/min")

# Per-player HSR variability
player_hsr = hsr_df.groupby('PlayerID').agg(
    n_games=('Q4_HSR_mpm','count'),
    mean=('Q4_HSR_mpm','mean'),
    std=('Q4_HSR_mpm','std'),
).round(2)
print(f"\nPer-player Q4 HSR variability:")
print(player_hsr.to_string())

# Plot: Q4 HSR distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(hsr_df['Q4_HSR_mpm'], bins=20, color='#378ADD', alpha=0.8, edgecolor='white')
axes[0].set_xlabel('Q4 HSR m/min')
axes[0].set_ylabel('Count')
axes[0].set_title('Q4 HSR Distribution')
axes[0].grid(alpha=0.2)

axes[1].scatter(trend_df['Q1_Q3_mpm'], trend_df['Q4_mpm'],
                alpha=0.5, color='#378ADD', s=20)
lims = [0, max(trend_df['Q1_Q3_mpm'].max(), trend_df['Q4_mpm'].max())+2]
axes[1].plot(lims, lims, 'r--', alpha=0.4, linewidth=1)
axes[1].set_xlabel('Q1-Q3 HSR m/min')
axes[1].set_ylabel('Q4 HSR m/min')
axes[1].set_title('Q1-Q3 vs Q4 HSR')
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'data_exploration.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: data_exploration.png")