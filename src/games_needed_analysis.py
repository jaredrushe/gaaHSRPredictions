import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters

from config import (FEATURES_PATH, CLUSTERS_PATH, SEQ_PATH, SEQ_MV_PATH,
                    SEQ_META_PATH, HIST_FEAT_HSR, FEATURE_COLS,
                    LABEL_ORDER, MIN_HIST_GAMES, TOP_FEATURES, OUTPUTS_DIR)
from utils import load_corrected_hist

ALPHA_AO   = 10.0
ALPHA_CO   = 10.0
LSTM_AO_H, LSTM_AO_D, LSTM_AO_LR = 32, 0.3, 0.001
LSTM_CO_H, LSTM_CO_D, LSTM_CO_LR = 16, 0.5, 0.002

feat_df  = pd.read_csv(FEATURES_PATH)
clusters = pd.read_csv(CLUSTERS_PATH)[['PlayerID','cluster']]
feat_df  = feat_df.merge(clusters, on='PlayerID', how='left')
hist_tr_lookup, hist_te_lookup = load_corrected_hist()
games_per_player = feat_df.groupby('PlayerID')['GameID'].count()

seq_uv = np.load(SEQ_PATH)
meta   = pd.read_csv(SEQ_META_PATH)
meta   = meta.merge(clusters, on='PlayerID', how='left')
meta   = meta.merge(feat_df[['GameID','PlayerID', HIST_FEAT_HSR]],
                    on=['GameID','PlayerID'], how='left')

n_samples  = len(meta)
n_steps    = seq_uv.shape[1]
players_seq  = meta['PlayerID'].values
game_ids_seq = meta['GameID'].values
clusters_seq = meta['cluster'].values
y_hsr_seq    = meta['Q4_HSR_per_min'].values
y_clf_seq    = meta['Q4_label_binary'].values
hist_hsr_seq = meta[HIST_FEAT_HSR].values.astype(np.float32)


# Model 1: Ridge hc+hist all_others
print("Model 1: Ridge hc+hist all_others...")
recs_1 = []
for idx, row in feat_df.iterrows():
    pid = row['PlayerID']; gid = row['GameID']
    if games_per_player[pid] < MIN_HIST_GAMES: continue
    train_df = feat_df[(feat_df['PlayerID']!=pid)&(feat_df['GameID']!=gid)]
    test_row = feat_df.loc[[idx]].copy()
    test_row[HIST_FEAT_HSR] = hist_te_lookup.get((pid,gid), np.nan)
    if len(train_df) == 0: continue
    X_tr = train_df[FEATURE_COLS].values.astype(float)
    X_te = test_row[FEATURE_COLS].values.astype(float)
    h_tr = train_df[HIST_FEAT_HSR].values.reshape(-1,1).astype(float)
    h_te = test_row[HIST_FEAT_HSR].values.reshape(-1,1).astype(float)
    hm = np.nanmean(h_tr)
    h_tr = np.where(np.isnan(h_tr), hm, h_tr)
    h_te = np.where(np.isnan(h_te), hm, h_te)
    sc = StandardScaler()
    Xtr = sc.fit_transform(np.hstack([X_tr,h_tr]))
    Xte = sc.transform(np.hstack([X_te,h_te]))
    m = Ridge(alpha=ALPHA_AO)
    m.fit(Xtr, train_df['Q4_HSR_per_min'].values)
    pred = float(m.predict(Xte)[0])
    recs_1.append({'PlayerID': pid, 'n_games': int(games_per_player[pid]),
                   'abs_error': abs(pred - float(row['Q4_HSR_per_min']))})
df1 = pd.DataFrame(recs_1)


# Model 2: Ridge tsfresh+hist cluster_only
print("Model 2: Ridge tsfresh+hist cluster_only (extracting TSFresh)...")
rows_ts = [{'id': i, 'time': t, 'value': float(seq_uv[i,t,0])}
           for i in range(n_samples) for t in range(n_steps)]
extracted = extract_features(
    pd.DataFrame(rows_ts), column_id='id', column_sort='time',
    column_value='value', default_fc_parameters=EfficientFCParameters(),
    n_jobs=1, disable_progressbar=True)
extracted = (extracted.astype(float)
             .replace([np.inf,-np.inf], np.nan)
             .fillna(extracted.median())
             .dropna(axis=1, how='all'))
extracted = extracted.loc[:, extracted.std() > 0]

recs_2 = []
for i in range(n_samples):
    pid = players_seq[i]; gid = game_ids_seq[i]; c = clusters_seq[i]
    if games_per_player.get(pid,0) < MIN_HIST_GAMES: continue
    tr = np.where((clusters_seq==c)&(game_ids_seq!=gid))[0]
    if len(tr) == 0: continue
    if len(np.unique(y_clf_seq[tr])) < 2: continue
    fold_le = LabelEncoder(); fold_le.fit(y_clf_seq[tr])
    y_enc   = fold_le.transform(y_clf_seq[tr])
    ext_tr  = extracted.iloc[tr]; ext_te = extracted.iloc[[i]]
    try:
        fdr = select_features(ext_tr, pd.Series(y_enc, index=ext_tr.index),
                               fdr_level=0.05, n_jobs=1)
        if fdr.shape[1] < TOP_FEATURES:
            fdr = select_features(ext_tr, pd.Series(y_enc, index=ext_tr.index),
                                   fdr_level=0.5, n_jobs=1)
        if fdr.shape[1] == 0: fdr = ext_tr
    except Exception: fdr = ext_tr
    sc0 = StandardScaler(); Xf = sc0.fit_transform(fdr.values)
    m0 = Ridge(alpha=1.0); m0.fit(Xf, y_hsr_seq[tr].astype(float))
    top_f = pd.Series(np.abs(m0.coef_), index=fdr.columns).nlargest(
        min(TOP_FEATURES-1, fdr.shape[1])).index.tolist()
    h_tr = hist_hsr_seq[tr].copy().astype(float)
    for j, tr_idx in enumerate(tr):
        if players_seq[tr_idx] != pid: continue
        h_tr[j] = hist_tr_lookup.get((pid, game_ids_seq[tr_idx], gid), np.nan)
    h_te = hist_te_lookup.get((pid, gid), np.nan)
    mv = np.nanmean(h_tr)
    h_tr = np.where(np.isnan(h_tr), mv, h_tr)
    h_te = mv if np.isnan(h_te) else h_te
    base_tr = fdr[top_f].values.astype(np.float32)
    base_te = ext_te[top_f].values.astype(np.float32)
    Xtr_raw = np.hstack([base_tr, h_tr.reshape(-1,1)])
    Xte_raw = np.hstack([base_te, [[h_te]]])
    sc = StandardScaler()
    m  = Ridge(alpha=ALPHA_CO)
    m.fit(sc.fit_transform(Xtr_raw), y_hsr_seq[tr].astype(float))
    pred = float(m.predict(sc.transform(Xte_raw))[0])
    recs_2.append({'PlayerID': pid, 'n_games': int(games_per_player[pid]),
                   'abs_error': abs(pred - float(y_hsr_seq[i]))})
df2 = pd.DataFrame(recs_2)


# LSTM per-player accuracy stubs (from previous run)
_player_ngames = {194:3,229:3,321:3,182:4,236:5,283:5,
                  174:6,331:6,244:7,203:8,130:9,146:10,152:11,201:11}
_lstm_ao_acc   = {194:0.667,229:0.667,321:0.667,182:1.000,
                  236:0.800,283:0.800,174:0.833,331:0.667,
                  244:0.857,203:0.750,130:0.778,146:0.700,
                  152:0.818,201:0.636}
_lstm_co_acc   = {194:1.000,229:0.667,321:1.000,182:1.000,
                  236:0.800,283:1.000,174:0.833,331:0.667,
                  244:0.857,203:0.750,130:0.889,146:0.800,
                  152:0.818,201:0.636}

df3_rows, df4_rows = [], []
for pid, ng in _player_ngames.items():
    for _ in range(ng):
        df3_rows.append({'PlayerID':pid,'n_games':ng,'correct':_lstm_ao_acc[pid]})
        df4_rows.append({'PlayerID':pid,'n_games':ng,'correct':_lstm_co_acc[pid]})
df3 = pd.DataFrame(df3_rows)
df4 = pd.DataFrame(df4_rows)


def player_summary(df, metric):
    col = 'abs_error' if metric == 'mae' else 'correct'
    return df.groupby(['PlayerID','n_games']).agg(
        score=(col,'mean'), n_obs=(col,'count')).reset_index()

def bucket_summary(df, metric):
    col = 'abs_error' if metric == 'mae' else 'correct'
    df = df.copy()
    df['bucket'] = pd.cut(df['n_games'], bins=[2,4,6,8,11],
                          labels=['3-4','5-6','7-8','9-11'], right=True)
    return df.groupby('bucket').agg(score=(col,'mean'), n=(col,'count')).reset_index()

pp1 = player_summary(df1,'mae'); pp2 = player_summary(df2,'mae')
pp3 = player_summary(df3,'acc'); pp4 = player_summary(df4,'acc')
b1  = bucket_summary(df1,'mae'); b2  = bucket_summary(df2,'mae')
b3  = bucket_summary(df3,'acc'); b4  = bucket_summary(df4,'acc')

print("\nSpearman correlations:")
for label, pp, metric in [
    ("Ridge hc+hist all_others (MAE)",      pp1, 'mae'),
    ("Ridge tsfresh+hist cluster_only (MAE)",pp2, 'mae'),
    ("LSTM mv+hist all_others (Acc)",        pp3, 'acc'),
    ("LSTM mv combined (Acc)",               pp4, 'acc'),
]:
    rho, p = stats.spearmanr(pp['n_games'], pp['score'])
    print(f"  {label}: rho={rho:+.3f}  p={p:.4f}  "
          f"({'significant' if p<0.05 else 'not significant'})")

print("\nBucket summaries:")
for label, b, metric in [
    ("Ridge hc+hist all_others", b1,'MAE'),
    ("Ridge tsfresh+hist cluster_only", b2,'MAE'),
    ("LSTM mv+hist all_others", b3,'Acc'),
    ("LSTM mv combined", b4,'Acc'),
]:
    print(f"\n  {label} ({metric}):")
    for _, r in b.iterrows():
        print(f"    {str(r.bucket):<8} score={r.score:.3f}  n={int(r.n)}")


# Plot
COLORS = ['#378ADD','#1D9E75','#D85A30','#9333ea']
MODEL_LABELS = ['Ridge hc+hist\nall_others (MAE)',
                'Ridge tsfresh+hist\ncluster_only (MAE)',
                'LSTM mv+hist\nall_others (Acc)',
                'LSTM mv\ncombined (Acc)']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance vs Historical Games -- All Four Headline Models',
             fontsize=12, fontweight='bold')

for ax, pp, b, metric, label, color in zip(
    axes.flat, [pp1,pp2,pp3,pp4], [b1,b2,b3,b4],
    ['mae','mae','acc','acc'], MODEL_LABELS, COLORS
):
    ax.scatter(pp['n_games'], pp['score'], s=pp['n_obs']*15,
               alpha=0.75, color=color, edgecolors='white', linewidths=0.5)
    for _, r in pp.iterrows():
        ax.annotate(str(int(r.PlayerID)), (r.n_games, r.score),
                    textcoords='offset points', xytext=(4,3), fontsize=6.5, alpha=0.7)
    z = np.polyfit(pp['n_games'], pp['score'], 1)
    x_line = np.linspace(pp['n_games'].min(), pp['n_games'].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), '--', color=color, alpha=0.5, linewidth=1.5)
    rho, p = stats.spearmanr(pp['n_games'], pp['score'])
    ax.set_xlabel('n_games'); ax.set_ylabel('MAE (m/min)' if metric=='mae' else 'Accuracy')
    ax.set_title(f'{label}\nrho={rho:+.3f}  p={p:.3f}'); ax.grid(alpha=0.2)

    ax2 = ax.inset_axes([0.62, 0.62, 0.36, 0.35])
    x_pos = range(len(b))
    ax2.bar(list(x_pos), b['score'].tolist(), color=color, alpha=0.7, width=0.6)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(b['bucket'].astype(str).tolist(), fontsize=6)
    ax2.tick_params(labelsize=6); ax2.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'games_needed_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: games_needed_analysis.png")