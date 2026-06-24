import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters

from config import (FEATURES_PATH, CLUSTERS_PATH, SEQ_PATH, SEQ_MV_PATH,
                    SEQ_META_PATH, HIST_FEAT_HSR, FEATURE_COLS, ACTION_ZONES,
                    LABEL_ORDER, MIN_HIST_GAMES, TOP_FEATURES, OUTPUTS_DIR)
from utils import reg_metrics, clf_metrics, load_corrected_hist

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
seq_mv = np.load(SEQ_MV_PATH)
meta   = pd.read_csv(SEQ_META_PATH)
meta   = meta.merge(clusters, on='PlayerID', how='left')
meta   = meta.merge(feat_df[['GameID','PlayerID', HIST_FEAT_HSR]],
                    on=['GameID','PlayerID'], how='left')

n_samples, n_steps, n_zones = seq_mv.shape
players_seq  = meta['PlayerID'].values
game_ids_seq = meta['GameID'].values
clusters_seq = meta['cluster'].values
y_hsr_seq    = meta['Q4_HSR_per_min'].values.astype(np.float32)
y_clf_seq    = meta['Q4_label_binary'].values
hist_hsr_seq = meta[HIST_FEAT_HSR].values.astype(np.float32)

ALL_FEATS = FEATURE_COLS + [HIST_FEAT_HSR]


# LSTM architecture
class Attention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.w = nn.Linear(h, 1)
    def forward(self, x):
        a = F.softmax(self.w(x).squeeze(-1), dim=-1)
        return (a.unsqueeze(-1) * x).sum(1)

class LSTMClf(nn.Module):
    def __init__(self, in_sz, hidden=16, dropout=0.5, n_cls=2, use_hist=False):
        super().__init__()
        self.use_hist = use_hist
        self.lstm = nn.LSTM(in_sz, hidden, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden*2)
        self.drop = nn.Dropout(dropout)
        fc_in = hidden*2 + 1 if use_hist else hidden*2
        self.fc1 = nn.Linear(fc_in, 16); self.fc2 = nn.Linear(16, n_cls)
    def forward(self, x, h=None):
        ctx = self.drop(self.attn(self.lstm(x)[0]))
        if self.use_hist and h is not None:
            ctx = torch.cat([ctx, h.view(-1,1)], dim=1)
        return self.fc2(F.relu(self.fc1(ctx)))

def augment(x):
    return x * torch.empty(x.shape[0],1,1).uniform_(0.9,1.1) + torch.randn_like(x)*0.05

def train_lstm(model, Xtr, ytr, loss_fn, lr, hist_tr=None, epochs=300, patience=25):
    nv = max(1, int(len(Xtr)*0.15))
    Xv,yv,hv = Xtr[-nv:],ytr[-nv:],(hist_tr[-nv:] if hist_tr is not None else None)
    Xt,yt,ht = Xtr[:-nv],ytr[:-nv],(hist_tr[:-nv] if hist_tr is not None else None)
    if len(Xt) == 0: return model
    tensors = [torch.tensor(Xt), torch.tensor(yt)]
    if ht is not None: tensors.append(torch.tensor(ht))
    dl = DataLoader(TensorDataset(*tensors), batch_size=min(16,len(Xt)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bv, bs, wait = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for batch in dl:
            xb,yb = batch[0],batch[1]; hb = batch[2] if ht is not None else None
            opt.zero_grad(); loss_fn(model(augment(xb),hb),yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            hv_t = torch.tensor(hv) if hv is not None else None
            vl = loss_fn(model(torch.tensor(Xv),hv_t),torch.tensor(yv)).item()
        if vl < bv:
            bv=vl; bs={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait += 1
            if wait >= patience: break
    if bs: model.load_state_dict(bs)
    return model


# Section 1: Ridge hc+hist all_others -- global importance + ablation
print("="*55)
print("1. Ridge hc+hist all_others -- importance + ablation")

df_elig = feat_df[feat_df[HIST_FEAT_HSR].notna()].copy()
sc_g = StandardScaler()
m_g  = Ridge(alpha=ALPHA_AO)
m_g.fit(sc_g.fit_transform(df_elig[ALL_FEATS].values.astype(float)),
        df_elig['Q4_HSR_per_min'].values)
imp_global = pd.Series(np.abs(m_g.coef_), index=ALL_FEATS).sort_values(ascending=False)
print("\nGlobal importance:")
for f, v in imp_global.items():
    print(f"  {f:<25} {v:.4f}")

def ridge_lopo_ao(feature_list):
    use_hist = HIST_FEAT_HSR in feature_list
    hc_feats = [f for f in feature_list if f != HIST_FEAT_HSR]
    t, p = [], []
    for idx, row in feat_df.iterrows():
        pid = row['PlayerID']; gid = row['GameID']
        if use_hist and games_per_player[pid] < MIN_HIST_GAMES: continue
        train_df = feat_df[(feat_df['PlayerID']!=pid)&(feat_df['GameID']!=gid)]
        test_row = feat_df.loc[[idx]].copy()
        if use_hist:
            test_row[HIST_FEAT_HSR] = hist_te_lookup.get((pid,gid), np.nan)
        if len(train_df) == 0: continue
        X_tr = train_df[hc_feats].values.astype(float) if hc_feats else np.zeros((len(train_df),1))
        X_te = test_row[hc_feats].values.astype(float) if hc_feats else np.zeros((1,1))
        if use_hist:
            h_tr = train_df[HIST_FEAT_HSR].values.reshape(-1,1).astype(float)
            h_te = test_row[HIST_FEAT_HSR].values.reshape(-1,1).astype(float)
            hm = np.nanmean(h_tr)
            h_tr = np.where(np.isnan(h_tr),hm,h_tr)
            h_te = np.where(np.isnan(h_te),hm,h_te)
            X_tr = np.hstack([X_tr,h_tr]); X_te = np.hstack([X_te,h_te])
        sc = StandardScaler()
        m = Ridge(alpha=ALPHA_AO)
        m.fit(sc.fit_transform(X_tr), train_df['Q4_HSR_per_min'].values)
        t.append(float(row['Q4_HSR_per_min']))
        p.append(float(m.predict(sc.transform(X_te))[0]))
    return reg_metrics(t, p), len(t)

base_m, base_n = ridge_lopo_ao(ALL_FEATS)
print(f"\nBaseline: MAE {base_m['mae']}  R2 {base_m['r2']}  n={base_n}")
print(f"\nAblation:")
print(f"  {'Feature':<25} {'MAE':>7}  {'dMAE':>8}")
ablation = []
for feat in ALL_FEATS:
    reduced = [f for f in ALL_FEATS if f != feat]
    m, _ = ridge_lopo_ao(reduced)
    d = m['mae'] - base_m['mae']
    ablation.append({'feature': feat, 'mae': m['mae'], 'delta_mae': d})
    print(f"  {feat:<25} {m['mae']:>7.3f}  {d:>+8.3f}")


# Section 2: Ridge tsfresh+hist cluster_only -- fold frequency
print("\n"+"="*55)
print("2. Ridge tsfresh+hist cluster_only -- fold frequency")
print("  Extracting TSFresh...")
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
print(f"  Features: {extracted.shape[1]}")

sel_counts = Counter(); rank_sums = Counter(); n_folds = 0
t_ts, p_ts = [], []
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
    m0 = Ridge(alpha=ALPHA_CO); m0.fit(Xf, y_hsr_seq[tr].astype(float))
    imps = pd.Series(np.abs(m0.coef_), index=fdr.columns)
    top_feats = imps.nlargest(min(TOP_FEATURES-1, len(imps)))
    for rank, feat in enumerate(top_feats.index, 1):
        sel_counts[feat] += 1; rank_sums[feat] += rank
    h_tr = hist_hsr_seq[tr].copy().astype(float)
    for j, tr_idx in enumerate(tr):
        if players_seq[tr_idx] != pid: continue
        h_tr[j] = hist_tr_lookup.get((pid, game_ids_seq[tr_idx], gid), np.nan)
    h_te = hist_te_lookup.get((pid, gid), np.nan)
    mv = np.nanmean(h_tr)
    h_tr = np.where(np.isnan(h_tr),mv,h_tr); h_te = mv if np.isnan(h_te) else h_te
    base_tr = fdr[top_feats.index].values.astype(np.float32)
    base_te = ext_te[top_feats.index].values.astype(np.float32)
    Xtr_raw = np.hstack([base_tr, h_tr.reshape(-1,1)])
    Xte_raw = np.hstack([base_te, [[h_te]]])
    sc = StandardScaler()
    m_r = Ridge(alpha=ALPHA_CO)
    m_r.fit(sc.fit_transform(Xtr_raw), y_hsr_seq[tr].astype(float))
    t_ts.append(float(y_hsr_seq[i])); p_ts.append(float(m_r.predict(sc.transform(Xte_raw))[0]))
    n_folds += 1

print(f"\n  LOPO: {n_folds} folds  MAE {reg_metrics(t_ts,p_ts)['mae']}")
freq_df = pd.DataFrame({
    'feature': list(sel_counts.keys()),
    'n_selected': list(sel_counts.values()),
    'pct_folds': [v/n_folds*100 for v in sel_counts.values()],
    'avg_rank': [rank_sums[f]/sel_counts[f] for f in sel_counts],
}).sort_values('n_selected', ascending=False).reset_index(drop=True)
print(f"\n  Top 15 features:")
print(f"  {'Feature':<45} {'Folds':>5}  {'%':>6}  {'AvgRank':>8}")
for _, row in freq_df.head(15).iterrows():
    print(f"  {row.feature:<45} {int(row.n_selected):>5}  "
          f"{row.pct_folds:>5.1f}%  {row.avg_rank:>8.2f}")
freq_df.to_csv(OUTPUTS_DIR + 'tsfresh_feature_frequency.csv', index=False)


# Section 3+4: LSTM permutation importance
def lstm_lopo_fold_data(split_type, use_hist, hidden, dropout, lr):
    ct, cp, fold_data = [], [], []
    for i in range(n_samples):
        pid = players_seq[i]; gid = game_ids_seq[i]
        if use_hist and games_per_player.get(pid,0) < MIN_HIST_GAMES: continue
        if split_type == 'all_others':
            tr = np.where((players_seq!=pid)&(game_ids_seq!=gid))[0]
        else:
            tr = np.where(game_ids_seq!=gid)[0]
        if len(tr) == 0: continue
        if len(np.unique(y_clf_seq[tr])) < 2: continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(seq_mv[tr].reshape(-1,n_zones)).reshape(
                len(tr),n_steps,n_zones).astype(np.float32)
        Xte = sc.transform(seq_mv[[i]].reshape(-1,n_zones)).reshape(
                1,n_steps,n_zones).astype(np.float32)
        fold_use_hist = use_hist; hm_tr = hm_te = None
        if use_hist:
            hm_tr_arr = hist_hsr_seq[tr].copy().astype(np.float32)
            hm_te_val = hist_te_lookup.get((pid,gid), np.nan)
            mv = np.nanmean(hm_tr_arr)
            if np.isnan(mv): fold_use_hist = False
            else:
                hm_tr_arr = np.where(np.isnan(hm_tr_arr),mv,hm_tr_arr).astype(np.float32)
                hm_te_val = mv if np.isnan(hm_te_val) else hm_te_val
                sc_h = StandardScaler()
                hm_tr = sc_h.fit_transform(hm_tr_arr.reshape(-1,1)).flatten().astype(np.float32)
                hm_te = sc_h.transform([[hm_te_val]]).flatten().astype(np.float32)
        fold_le = LabelEncoder(); fold_le.fit(y_clf_seq[tr])
        y_enc   = fold_le.transform(y_clf_seq[tr]).astype(np.int64)
        n_cls   = len(fold_le.classes_)
        counts  = np.bincount(y_enc, minlength=n_cls).astype(np.float32)
        weights = torch.tensor(1.0/np.sqrt(counts+1e-6)); weights=(weights/weights.sum())*n_cls
        torch.manual_seed(42)
        model = train_lstm(LSTMClf(n_zones, hidden=hidden, dropout=dropout,
                                    n_cls=n_cls, use_hist=fold_use_hist),
                           Xtr,y_enc,nn.CrossEntropyLoss(weight=weights),
                           lr=lr,hist_tr=hm_tr)
        model.eval()
        with torch.no_grad():
            ht = torch.tensor(hm_te) if hm_te is not None else None
            pred = int(torch.argmax(model(torch.tensor(Xte),ht),dim=1).item())
        cp.append(fold_le.inverse_transform([pred])[0]); ct.append(y_clf_seq[i])
        fold_data.append({'model':model,'Xte':Xte,'hm_te':hm_te,
                          'fold_le':fold_le,'true':y_clf_seq[i]})
    return ct, cp, fold_data

def permutation_importance(fold_data, n_repeats=5):
    ct_base = [fd['true'] for fd in fold_data]
    cp_base = []
    for fd in fold_data:
        fd['model'].eval()
        with torch.no_grad():
            ht = torch.tensor(fd['hm_te']) if fd['hm_te'] is not None else None
            pred = int(torch.argmax(fd['model'](torch.tensor(fd['Xte']),ht),dim=1).item())
        cp_base.append(fd['fold_le'].inverse_transform([pred])[0])
    base_f1 = clf_metrics(ct_base, cp_base)['f1']
    zone_imp = {}
    for zi, zname in enumerate(ACTION_ZONES):
        drops = []
        for rep in range(n_repeats):
            np.random.seed(rep)
            cp_perm = []
            for fd in fold_data:
                Xp = fd['Xte'].copy()
                Xp[0,:,zi] = np.random.permutation(Xp[0,:,zi])
                fd['model'].eval()
                with torch.no_grad():
                    ht = torch.tensor(fd['hm_te']) if fd['hm_te'] is not None else None
                    pred = int(torch.argmax(fd['model'](torch.tensor(Xp),ht),dim=1).item())
                cp_perm.append(fd['fold_le'].inverse_transform([pred])[0])
            drops.append(base_f1 - clf_metrics(ct_base,cp_perm)['f1'])
        zone_imp[zname] = {'mean_drop': float(np.mean(drops)), 'std_drop': float(np.std(drops))}
    return base_f1, zone_imp

print("\n"+"="*55)
print("3. LSTM mv+hist all_others -- zone permutation importance")
ct_ao, cp_ao, fold_ao = lstm_lopo_fold_data('all_others',True,LSTM_AO_H,LSTM_AO_D,LSTM_AO_LR)
base_ao, imp_ao = permutation_importance(fold_ao)
print(f"  Baseline F1: {base_ao:.3f}")
for z, v in sorted(imp_ao.items(), key=lambda x: x[1]['mean_drop'], reverse=True):
    print(f"  {z:<28} {v['mean_drop']:>+.4f}  std={v['std_drop']:.4f}")

print("\n"+"="*55)
print("4. LSTM mv combined -- zone permutation importance")
ct_co, cp_co, fold_co = lstm_lopo_fold_data('combined',False,LSTM_CO_H,LSTM_CO_D,LSTM_CO_LR)
base_co, imp_co = permutation_importance(fold_co)
print(f"  Baseline F1: {base_co:.3f}")
for z, v in sorted(imp_co.items(), key=lambda x: x[1]['mean_drop'], reverse=True):
    print(f"  {z:<28} {v['mean_drop']:>+.4f}  std={v['std_drop']:.4f}")


# Plot
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Feature Importance and Ablation', fontsize=13, fontweight='bold')

ax = axes[0,0]
imp_s = imp_global.sort_values()
ax.barh(imp_s.index, imp_s.values,
        color=['#D85A30' if f==HIST_FEAT_HSR else '#378ADD' for f in imp_s.index], alpha=0.85)
ax.set_xlabel('|Coefficient|'); ax.set_title('Ridge hc+hist all_others\nGlobal Importance')
ax.grid(axis='x', alpha=0.3)

ax = axes[0,1]
abl_df = pd.DataFrame(ablation).sort_values('delta_mae', ascending=False)
bars = ax.barh(abl_df['feature'], abl_df['delta_mae'],
               color=['#E53935' if v>0 else '#43A047' for v in abl_df['delta_mae']], alpha=0.85)
ax.axvline(0, color='black', linewidth=0.8)
for bar, row in zip(bars, abl_df.itertuples()):
    ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
            f'{row.delta_mae:+.3f}', va='center', fontsize=8)
ax.set_xlabel('dMAE'); ax.set_title('Ridge hc+hist all_others\nAblation')
ax.grid(axis='x', alpha=0.3)

ax = axes[0,2]
top20 = freq_df.head(20).sort_values('pct_folds')
ax.barh(top20['feature'], top20['pct_folds'], color='#378ADD', alpha=0.85)
ax.axvline(50, color='red', linestyle='--', alpha=0.4, linewidth=1)
ax.set_xlabel('% of folds'); ax.set_title('TSFresh cluster_only\nTop 20 Selected Features')
ax.grid(axis='x', alpha=0.3)

for ax, imp, base_f1, title in [
    (axes[1,0], imp_ao, base_ao, f'LSTM mv+hist all_others\nPermutation (baseline F1={base_ao:.3f})'),
    (axes[1,1], imp_co, base_co, f'LSTM mv combined\nPermutation (baseline F1={base_co:.3f})'),
]:
    zones = list(imp.keys())
    drops = [imp[z]['mean_drop'] for z in zones]
    stds  = [imp[z]['std_drop']  for z in zones]
    idx   = np.argsort(drops)[::-1]
    zs    = [zones[i] for i in idx]; ds = [drops[i] for i in idx]; ss = [stds[i] for i in idx]
    ax.barh(zs[::-1], ds[::-1], xerr=ss[::-1],
            color=['#E53935' if v>0 else '#94a3b8' for v in ds[::-1]], alpha=0.85, capsize=3)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Mean F1 drop'); ax.set_title(title); ax.grid(axis='x', alpha=0.3)

top15 = freq_df[freq_df['n_selected']>=5].nsmallest(15,'avg_rank').sort_values('avg_rank',ascending=False)
axes[1,2].barh(top15['feature'], top15['avg_rank'], color='#1D9E75', alpha=0.85)
axes[1,2].set_xlabel('Avg rank (1=most important)')
axes[1,2].set_title('TSFresh cluster_only\nAvg Rank of Consistent Features')
axes[1,2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'feature_importance.png', dpi=150, bbox_inches='tight')
print("\nSaved: feature_importance.png")
print("Done.")