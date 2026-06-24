import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters

from config import (FEATURES_PATH, CLUSTERS_PATH, SEQ_PATH, SEQ_MV_PATH,
                    SEQ_META_PATH, HIST_FEAT_HSR, FEATURE_COLS,
                    LABEL_ORDER, MIN_HIST_GAMES, TOP_FEATURES, OUTPUTS_DIR)
from utils import reg_metrics, clf_metrics, save_results, load_corrected_hist

ALPHA_GRID = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
LSTM_GRID  = [(16,0.3,0.001),(16,0.5,0.001),(16,0.5,0.002),
              (32,0.3,0.001),(32,0.5,0.001),(32,0.3,0.002)]

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
y_hsr_seq    = meta['Q4_HSR_per_min'].values
y_clf_seq    = meta['Q4_label_binary'].values
hist_hsr_seq = meta[HIST_FEAT_HSR].values.astype(np.float32)


# Section 1: Ridge hc+hist all_others alpha sweep
print("="*55)
print("1. Ridge hc+hist all_others -- alpha tuning")
def run_ridge_ao(alpha):
    t, p = [], []
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
        m = Ridge(alpha=alpha)
        m.fit(Xtr, train_df['Q4_HSR_per_min'].values)
        t.append(float(row['Q4_HSR_per_min']))
        p.append(float(m.predict(Xte)[0]))
    return reg_metrics(t, p), len(t)

best_ao = {'alpha': 1.0, 'mae': float('inf')}
print(f"  {'Alpha':>8}  {'MAE':>7}  {'R2':>7}")
for alpha in ALPHA_GRID:
    m, n = run_ridge_ao(alpha)
    flag = ' <--' if m['mae'] < best_ao['mae'] else ''
    print(f"  {alpha:>8.3f}  {m['mae']:>7.3f}  {m['r2']:>7.3f}{flag}")
    if m['mae'] < best_ao['mae']:
        best_ao = {'alpha': alpha, 'mae': m['mae']}
print(f"  Best alpha: {best_ao['alpha']}  MAE: {best_ao['mae']:.3f}")
m_best, n = run_ridge_ao(best_ao['alpha'])
save_results('tuning', 'Ridge_tuned', 'hc+hist', 'all_others', m_best,
             {'accuracy':0,'f1':0,'precision':0,'recall':0})


# Section 2: Ridge tsfresh+hist cluster_only alpha sweep
print("\n"+"="*55)
print("2. Ridge tsfresh+hist cluster_only -- alpha tuning")
print("  Extracting TSFresh features...")
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
print(f"  Features extracted: {extracted.shape[1]}")

def run_tsfresh_co(alpha):
    t, p = [], []
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
        imps  = pd.Series(np.abs(m0.coef_), index=fdr.columns)
        top_f = imps.nlargest(min(TOP_FEATURES-1, len(imps))).index.tolist()
        h_tr = hist_hsr_seq[tr].copy().astype(float)
        for j, tr_idx in enumerate(tr):
            if players_seq[tr_idx] != pid: continue
            tr_gid = game_ids_seq[tr_idx]
            h_tr[j] = hist_tr_lookup.get((pid, tr_gid, gid), np.nan)
        h_te = hist_te_lookup.get((pid, gid), np.nan)
        mv = np.nanmean(h_tr)
        h_tr = np.where(np.isnan(h_tr), mv, h_tr)
        h_te = mv if np.isnan(h_te) else h_te
        base_tr = fdr[top_f].values.astype(np.float32)
        base_te = ext_te[top_f].values.astype(np.float32)
        Xtr_raw = np.hstack([base_tr, h_tr.reshape(-1,1)])
        Xte_raw = np.hstack([base_te, [[h_te]]])
        sc = StandardScaler()
        m  = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(Xtr_raw), y_hsr_seq[tr].astype(float))
        t.append(float(y_hsr_seq[i]))
        p.append(float(m.predict(sc.transform(Xte_raw))[0]))
    return reg_metrics(t, p), len(t)

best_co = {'alpha': 1.0, 'mae': float('inf')}
print(f"  {'Alpha':>8}  {'MAE':>7}  {'R2':>7}")
for alpha in ALPHA_GRID:
    m, n = run_tsfresh_co(alpha)
    flag = ' <--' if m['mae'] < best_co['mae'] else ''
    print(f"  {alpha:>8.3f}  {m['mae']:>7.3f}  {m['r2']:>7.3f}{flag}")
    if m['mae'] < best_co['mae']:
        best_co = {'alpha': alpha, 'mae': m['mae']}
print(f"  Best alpha: {best_co['alpha']}  MAE: {best_co['mae']:.3f}")
m_best, n = run_tsfresh_co(best_co['alpha'])
save_results('tuning', 'Ridge_tsfresh_tuned', 'tsfresh+hist', 'cluster_only', m_best,
             {'accuracy':0,'f1':0,'precision':0,'recall':0})


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
        self.fc1 = nn.Linear(fc_in, 16)
        self.fc2 = nn.Linear(16, n_cls)
    def forward(self, x, h=None):
        ctx = self.drop(self.attn(self.lstm(x)[0]))
        if self.use_hist and h is not None:
            ctx = torch.cat([ctx, h.view(-1,1)], dim=1)
        return self.fc2(F.relu(self.fc1(ctx)))

def augment(x):
    return x * torch.empty(x.shape[0],1,1).uniform_(0.9,1.1) + torch.randn_like(x)*0.05

def train_lstm(model, Xtr, ytr, loss_fn, lr, hist_tr=None, epochs=300, patience=25):
    nv = max(1, int(len(Xtr)*0.15))
    Xv,yv,hv = Xtr[-nv:], ytr[-nv:], (hist_tr[-nv:] if hist_tr is not None else None)
    Xt,yt,ht = Xtr[:-nv], ytr[:-nv], (hist_tr[:-nv] if hist_tr is not None else None)
    if len(Xt) == 0: return model
    tensors = [torch.tensor(Xt), torch.tensor(yt)]
    if ht is not None: tensors.append(torch.tensor(ht))
    dl = DataLoader(TensorDataset(*tensors), batch_size=min(16,len(Xt)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bv, bs, wait = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for batch in dl:
            xb, yb = batch[0], batch[1]
            hb = batch[2] if ht is not None else None
            opt.zero_grad()
            loss_fn(model(augment(xb), hb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hv_t = torch.tensor(hv) if hv is not None else None
            vl = loss_fn(model(torch.tensor(Xv), hv_t), torch.tensor(yv)).item()
        if vl < bv:
            bv = vl; bs = {k: v.clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience: break
    if bs: model.load_state_dict(bs)
    return model

def run_lstm_clf(split_type, use_hist, hidden, dropout, lr):
    ct, cp = [], []
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
        weights = torch.tensor(1.0/np.sqrt(counts+1e-6)); weights = (weights/weights.sum())*n_cls
        torch.manual_seed(42)
        model = train_lstm(LSTMClf(n_zones, hidden=hidden, dropout=dropout,
                                    n_cls=n_cls, use_hist=fold_use_hist),
                           Xtr, y_enc, nn.CrossEntropyLoss(weight=weights),
                           lr=lr, hist_tr=hm_tr)
        model.eval()
        with torch.no_grad():
            ht = torch.tensor(hm_te) if hm_te is not None else None
            pred = int(torch.argmax(model(torch.tensor(Xte),ht),dim=1).item())
        cp.append(fold_le.inverse_transform([pred])[0])
        ct.append(y_clf_seq[i])
    return clf_metrics(ct, cp), len(ct)


# Section 3: LSTM all_others tuning
print("\n"+"="*55)
print("3. LSTM multivariate+hist all_others -- tuning")
print(f"  {'hidden':>6} {'drop':>5} {'lr':>7}  {'F1':>7}")
best_ao_lstm = {'f1': 0.0, 'params': (16,0.5,0.001)}
for hidden, dropout, lr in LSTM_GRID:
    cm, n = run_lstm_clf('all_others', True, hidden, dropout, lr)
    flag = ' <--' if cm['f1'] > best_ao_lstm['f1'] else ''
    print(f"  {hidden:>6} {dropout:>5.1f} {lr:>7.4f}  {cm['f1']:>7.3f}{flag}")
    if cm['f1'] > best_ao_lstm['f1']:
        best_ao_lstm = {'f1': cm['f1'], 'params': (hidden, dropout, lr)}
h,d,l = best_ao_lstm['params']
print(f"  Best: hidden={h} drop={d} lr={l} F1={best_ao_lstm['f1']:.3f}")
cm_best, _ = run_lstm_clf('all_others', True, h, d, l)
save_results('tuning', f'LSTM_ao_h{h}_d{int(d*10)}_lr{int(l*10000)}',
             'multivariate+hist', 'all_others',
             {'mae':0,'rmse':0,'r2':0}, cm_best)


# Section 4: LSTM combined tuning
print("\n"+"="*55)
print("4. LSTM multivariate combined -- tuning")
print(f"  {'hidden':>6} {'drop':>5} {'lr':>7}  {'F1':>7}")
best_co_lstm = {'f1': 0.0, 'params': (16,0.5,0.002)}
for hidden, dropout, lr in LSTM_GRID:
    cm, n = run_lstm_clf('combined', False, hidden, dropout, lr)
    flag = ' <--' if cm['f1'] > best_co_lstm['f1'] else ''
    print(f"  {hidden:>6} {dropout:>5.1f} {lr:>7.4f}  {cm['f1']:>7.3f}{flag}")
    if cm['f1'] > best_co_lstm['f1']:
        best_co_lstm = {'f1': cm['f1'], 'params': (hidden, dropout, lr)}
h,d,l = best_co_lstm['params']
print(f"  Best: hidden={h} drop={d} lr={l} F1={best_co_lstm['f1']:.3f}")
cm_best, _ = run_lstm_clf('combined', False, h, d, l)
save_results('tuning', f'LSTM_co_h{h}_d{int(d*10)}_lr{int(l*10000)}',
             'multivariate', 'combined',
             {'mae':0,'rmse':0,'r2':0}, cm_best)

print("\nDone.")