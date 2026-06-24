import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from config import (FEATURES_PATH, CLUSTERS_PATH, SEQ_PATH,
                    SEQ_META_PATH, OUTPUTS_DIR, HIST_FEAT_HSR,
                    FEATURE_COLS, ACTION_ZONES)

feat_df  = pd.read_csv(FEATURES_PATH)
clusters = pd.read_csv(CLUSTERS_PATH)[['PlayerID','cluster','cluster_name']]
feat_df  = feat_df.merge(clusters, on='PlayerID', how='left')
seq_uv   = np.load(SEQ_PATH)
meta     = pd.read_csv(SEQ_META_PATH)
meta     = meta.merge(clusters, on='PlayerID', how='left')

C0 = '#378ADD'
C1 = '#D85A30'
CLUSTER_COLORS = {0: C0, 1: C1}

# Figure 1: Q4 HSR distribution + cluster overlay

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Figure 1: Q4 HSR Distribution and Inter-Player Variability',
             fontsize=11, fontweight='bold')

# Histogram with cluster overlay
ax = axes[0]
for c, grp in feat_df.groupby('cluster'):
    cname = grp['cluster_name'].iloc[0]
    ax.hist(grp['Q4_HSR_per_min'], bins=14, alpha=0.6,
            color=CLUSTER_COLORS[c], label=cname, edgecolor='white')
ax.axvline(feat_df['Q4_HSR_per_min'].mean(), color='black',
           linestyle='--', linewidth=1, alpha=0.6, label='Mean (26.35)')
ax.set_xlabel('Q4 HSR (m/min)')
ax.set_ylabel('Count')
ax.set_title('Q4 HSR Distribution by Cluster')
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# Per-player boxplot sorted by median
ax = axes[1]
player_data = [feat_df[feat_df['PlayerID']==pid]['Q4_HSR_per_min'].values
               for pid in feat_df.groupby('PlayerID')['Q4_HSR_per_min']
                                  .median().sort_values().index]
player_ids  = feat_df.groupby('PlayerID')['Q4_HSR_per_min'].median().sort_values().index.tolist()
player_clusters = [feat_df[feat_df['PlayerID']==pid]['cluster'].iloc[0]
                   for pid in player_ids]
bp = ax.boxplot(player_data, patch_artist=True, medianprops=dict(color='white',linewidth=1.5))
for patch, c in zip(bp['boxes'], player_clusters):
    patch.set_facecolor(CLUSTER_COLORS[c]); patch.set_alpha(0.75)
ax.set_xlabel('Player (sorted by median Q4 HSR)')
ax.set_ylabel('Q4 HSR (m/min)')
ax.set_title('Per-Player Q4 HSR Variability')
ax.set_xticks([])
ax.grid(axis='y', alpha=0.2)
patches = [mpatches.Patch(color=C0, alpha=0.75, label='C0: Low-intensity'),
           mpatches.Patch(color=C1, alpha=0.75, label='C1: High-intensity')]
ax.legend(handles=patches, fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'appendix_fig1_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: appendix_fig1_distribution.png")



# Figure 2: Pre-Q4 rolling window sequences by cluster

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Figure 2: Mean Pre-Q4 HSR Sequence by Cluster',
             fontsize=11, fontweight='bold')

steps = np.arange(50)
for ax, c in zip(axes, [0, 1]):
    idx   = meta[meta['cluster']==c].index.tolist()
    seqs  = seq_uv[idx, :, 0]
    mean  = seqs.mean(axis=0)
    std   = seqs.std(axis=0)
    cname = clusters[clusters['cluster']==c]['cluster_name'].iloc[0]
    color = CLUSTER_COLORS[c]
    ax.plot(steps, mean, color=color, linewidth=2, label='Mean')
    ax.fill_between(steps, mean-std, mean+std, alpha=0.2, color=color, label='Mean +/- SD')
    ax.axvline(49, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Window (50 = closest to Q4)')
    ax.set_ylabel('HSR (m/min)')
    ax.set_title(cname)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'appendix_fig2_sequences.png', dpi=150, bbox_inches='tight')
print("Saved: appendix_fig2_sequences.png")


# Figure 3: Ablation bar chart
ablation_data = {
    'hist_mean_q4_mpm':  +0.634,
    'series_mean':       +0.065,
    'series_last':       +0.033,
    'last10_slope':      +0.026,
    'series_std':        -0.054,
    'series_slope':      -0.073,
}
labels = list(ablation_data.keys())
values = list(ablation_data.values())
colors = ['#E53935' if v > 0 else '#43A047' for v in values]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height()/2,
            f'{val:+.3f}', va='center',
            ha='left' if val >= 0 else 'right', fontsize=9)
ax.set_xlabel('ΔMAE when feature removed (positive = worse)')
ax.set_title('Figure 3: Feature Ablation -- Ridge hc+hist all_others',
             fontweight='bold')
ax.grid(axis='x', alpha=0.2)
red_patch   = mpatches.Patch(color='#E53935', alpha=0.85, label='Harmful to remove')
green_patch = mpatches.Patch(color='#43A047', alpha=0.85, label='Beneficial to remove')
ax.legend(handles=[red_patch, green_patch], fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'appendix_fig3_ablation.png', dpi=150, bbox_inches='tight')
print("Saved: appendix_fig3_ablation.png")


# Figure 4: Cold vs personalised delta MAE and F1

models = ['ARIMA','LSTM','Ridge\nhc+hist','RF\nhc+hist',
          'XGBoost\nhc+hist','Ridge\ntsfresh+hist','RF\ntsfresh+hist']
delta_mae = [-0.836, -0.999, -0.270, -1.014, -0.828, -0.827, -1.524]
delta_f1  = [-0.039, +0.028, +0.018, +0.041, +0.049, +0.092, +0.095]

x = np.arange(len(models))
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('Figure 4: Cold-Start vs Personalised -- Performance Delta',
             fontsize=11, fontweight='bold')

ax = axes[0]
colors = ['#43A047' if v < 0 else '#E53935' for v in delta_mae]
bars = ax.bar(x, delta_mae, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
ax.set_ylabel('ΔMAE (negative = improvement)')
ax.set_title('Regression MAE Change')
ax.grid(axis='y', alpha=0.2)
for bar, val in zip(bars, delta_mae):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height() + (0.01 if val >= 0 else -0.04),
            f'{val:+.3f}', ha='center', fontsize=7.5)

ax = axes[1]
colors = ['#43A047' if v > 0 else '#E53935' for v in delta_f1]
bars = ax.bar(x, delta_f1, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
ax.set_ylabel('ΔF1 (positive = improvement)')
ax.set_title('Classification F1 Change')
ax.grid(axis='y', alpha=0.2)
for bar, val in zip(bars, delta_f1):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height() + (0.001 if val >= 0 else -0.004),
            f'{val:+.3f}', ha='center', fontsize=7.5)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'appendix_fig4_cold_vs_personal.png', dpi=150, bbox_inches='tight')
print("Saved: appendix_fig4_cold_vs_personal.png")


# Figure 5: TSFresh feature frequency top 10

tsf_features = [
    'value__c3__lag_3',
    'value__quantile__q_0.7',
    'value__quantile__q_0.3',
    'value__agg_linear_trend..chunk_5_min',
    'value__quantile__q_0.9',
    'value__benford_correlation',
    'value__median',
    'value__ar_coefficient__k_10',
    'value__c3__lag_2',
    'value__quantile__q_0.1',
]
pct = [65.9, 42.9, 33.0, 29.7, 22.0, 20.9, 19.8, 16.5, 11.0, 8.8]
avg_rank = [1.65, 2.67, 3.43, 2.67, 1.20, 2.58, 2.06, 1.67, 3.30, 4.00]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('Figure 5: TSFresh Feature Frequency -- Ridge tsfresh+hist cluster_only',
             fontsize=11, fontweight='bold')

ax = axes[0]
ax.barh(tsf_features[::-1], pct[::-1], color='#378ADD', alpha=0.85, edgecolor='white')
ax.axvline(50, color='red', linestyle='--', alpha=0.4, linewidth=1, label='50% threshold')
ax.set_xlabel('% of LOPO folds selected in')
ax.set_title('Selection Frequency (Top 10)')
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.2)

ax = axes[1]
ax.barh(tsf_features[::-1], avg_rank[::-1], color='#1D9E75', alpha=0.85, edgecolor='white')
ax.set_xlabel('Average importance rank (1 = most important)')
ax.set_title('Average Importance Rank')
ax.grid(axis='x', alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'appendix_fig5_tsfresh_frequency.png', dpi=150, bbox_inches='tight')
print("Saved: appendix_fig5_tsfresh_frequency.png")


# Figure 6: LSTM zone permutation importance

zones = ['Standing','Walking','Jogging','Running','High Intensity Running','Sprint']
drops_ao = [0.000, 0.0018, 0.000, 0.0090, -0.0130, -0.0130]
drops_co = [0.000, 0.0036, 0.000, 0.0054, +0.0150, 0.000]

x = np.arange(len(zones))
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Figure 6: LSTM Zone Permutation Importance',
             fontsize=11, fontweight='bold')

for ax, drops, title in zip(
    axes,
    [drops_ao, drops_co],
    ['mv+hist all_others (cold start)', 'mv combined (personalised)']
):
    colors = ['#E53935' if v > 0 else '#94a3b8' if v == 0 else '#378ADD'
              for v in drops]
    bars = ax.bar(x, drops, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Mean F1 drop when zone permuted')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.2)
    for bar, val in zip(bars, drops):
        if val != 0:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + (0.0005 if val > 0 else -0.001),
                    f'{val:+.4f}', ha='center', fontsize=7.5)

red_patch  = mpatches.Patch(color='#E53935', alpha=0.85, label='Positive contribution')
blue_patch = mpatches.Patch(color='#378ADD', alpha=0.85, label='Harmful when included')
grey_patch = mpatches.Patch(color='#94a3b8', alpha=0.85, label='No contribution')
axes[0].legend(handles=[red_patch, blue_patch, grey_patch], fontsize=7)
axes[1].legend(handles=[red_patch, blue_patch, grey_patch], fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'appendix_fig6_lstm_zones.png', dpi=150, bbox_inches='tight')
print("Saved: appendix_fig6_lstm_zones.png")


# Figure 7: TSFresh k-sweep

k_vals = [1, 2, 3, 4, 5, 6, 8, 10]
mae_k  = [4.865, 4.841, 4.822, 4.804, 4.801, 4.811, 4.836, 4.948]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_vals, mae_k, 'o-', color='#378ADD', linewidth=2, markersize=7)
ax.axvline(5, color='#E53935', linestyle='--', alpha=0.6, linewidth=1.5, label='Best k=5')
ax.axvline(4, color='#94a3b8', linestyle=':', alpha=0.6, linewidth=1.5, label='Default k=4')
for k, m in zip(k_vals, mae_k):
    ax.annotate(f'{m:.3f}', (k, m), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=8)
ax.set_xlabel('Number of TSFresh features (k)')
ax.set_ylabel('MAE (m/min)')
ax.set_title('Figure 7: TSFresh Feature Count Sweep -- cluster_only',
             fontweight='bold')
ax.set_xticks(k_vals)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'appendix_fig7_ksweep.png', dpi=150, bbox_inches='tight')
print("Saved: appendix_fig7_ksweep.png")

print("\nAll appendix figures saved.")