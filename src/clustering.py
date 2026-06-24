import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import (DATA_PATH, FEATURES_PATH, CLUSTERS_PATH,
                    OUTPUTS_DIR, Q_SPLIT_SECONDS, HSR_ACTIONS)

feat_df = pd.read_csv(FEATURES_PATH)
df      = pd.read_csv(DATA_PATH)

# Compute Q1-Q3 and Q4 HSR per player-game from raw data
h2s = df[df['Half']==2].groupby('GameID')['Start_Second'].min().rename('h2_start')
df  = df.merge(h2s, on='GameID')
df['q4_cut'] = df['h2_start'] + Q_SPLIT_SECONDS

records = []
for (gid, pid), grp in df.groupby(['GameID','PlayerID']):
    q4c = grp['q4_cut'].iloc[0]
    q123 = grp[~((grp['Half']==2) & (grp['Start_Second'] >= q4c))]
    q4   = grp[(grp['Half']==2) & (grp['Start_Second'] >= q4c)]
    q123_dur = q123['Duration'].sum()
    q4_dur   = q4['Duration'].sum()
    if q123_dur == 0 or q4_dur == 0: continue
    q123_hsr = q123[q123['Action'].isin(HSR_ACTIONS)]['Distance'].sum()
    q4_hsr   = q4[q4['Action'].isin(HSR_ACTIONS)]['Distance'].sum()
    records.append({'GameID': gid, 'PlayerID': pid,
                    'Q1Q3_HSR_mpm': q123_hsr / (q123_dur/60),
                    'Q4_HSR_mpm':   q4_hsr   / (q4_dur/60)})

hsr_df = pd.DataFrame(records)
hsr_df = hsr_df.merge(feat_df[['GameID','PlayerID']].drop_duplicates(),
                       on=['GameID','PlayerID'])

player_profiles = hsr_df.groupby('PlayerID').agg(
    mean_q1q3_hsr=('Q1Q3_HSR_mpm', 'mean'),
    mean_q4_hsr=('Q4_HSR_mpm', 'mean'),
    std_q4_hsr=('Q4_HSR_mpm', 'std'),
).fillna(0).reset_index()

profile_feats = ['mean_q1q3_hsr', 'mean_q4_hsr', 'std_q4_hsr']
sc = StandardScaler()
X = sc.fit_transform(player_profiles[profile_feats].values)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
player_profiles['cluster'] = kmeans.fit_predict(X)

# Label by mean Q4 HSR
cluster_means = player_profiles.groupby('cluster')['mean_q4_hsr'].mean()
high_cluster  = cluster_means.idxmax()
player_profiles['cluster_name'] = player_profiles['cluster'].apply(
    lambda c: 'C1: High-intensity Consistent' if c == high_cluster
              else 'C0: Low-intensity Consistent')

print("Cluster sizes:")
print(player_profiles['cluster_name'].value_counts().to_string())
print("\nCluster profiles:")
print(player_profiles.groupby('cluster_name')[profile_feats].mean().round(2).to_string())

player_profiles[['PlayerID','cluster','cluster_name']].to_csv(
    CLUSTERS_PATH, index=False)
print(f"\nSaved: {CLUSTERS_PATH}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = {0: '#378ADD', 1: '#D85A30'}
for c in [0, 1]:
    sub = player_profiles[player_profiles['cluster'] == c]
    ax.scatter(sub['mean_q1q3_hsr'], sub['mean_q4_hsr'],
               color=colors[c], label=sub['cluster_name'].iloc[0],
               alpha=0.8, s=80, edgecolors='white')
    for _, r in sub.iterrows():
        ax.annotate(str(int(r.PlayerID)), (r.mean_q1q3_hsr, r.mean_q4_hsr),
                    fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Mean Q1-Q3 HSR m/min')
ax.set_ylabel('Mean Q4 HSR m/min')
ax.set_title('Player Archetypes (K=2)')
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'clustering.png', dpi=150, bbox_inches='tight')
print("Saved: clustering.png")