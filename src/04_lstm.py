import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (SEQ_MV_PATH, SEQ_META_PATH, FEATURES_PATH, CLUSTERS_PATH,
                    LABEL_ORDER, MIN_TRAIN_GAMES, MIN_HIST_GAMES, HIST_FEAT_HSR)
from utils import reg_metrics, clf_metrics, save_results, load_corrected_hist

sequences = np.load(SEQ_MV_PATH)
meta      = pd.read_csv(SEQ_META_PATH)
feat_df   = pd.read_csv(FEATURES_PATH)
clusters  = pd.read_csv(CLUSTERS_PATH)[['PlayerID','cluster']]

meta = meta.merge(clusters, on='PlayerID', how='left')
meta = meta.merge(
    feat_df[['GameID','PlayerID', HIST_FEAT_HSR]],
    on=['GameID','PlayerID'], how='left')

n_samples, n_steps, n_zones = sequences.shape

y_hsr        = meta['Q4_HSR_per_min'].values.astype(np.float32)
y_clf_lbl    = meta['Q4_label_binary'].values
players      = meta['PlayerID'].values
game_ids     = meta['GameID'].values
clusters_arr = meta['cluster'].values
hist_hsr     = meta[HIST_FEAT_HSR].values.astype(np.float32)

games_per_player = feat_df.groupby('PlayerID')['GameID'].count()

# Load corrected hist lookups
hist_tr_lookup, hist_te_lookup = load_corrected_hist()

torch.manual_seed(42)
print(f"Samples: {n_samples} | Timesteps: {n_steps} | Zones: {n_zones}")
print(f"Labels: {pd.Series(y_clf_lbl).value_counts().to_dict()}")

class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.w = nn.Linear(hidden, 1)
    def forward(self, x):
        s = self.w(x).squeeze(-1)
        a = F.softmax(s, dim=-1)
        return (a.unsqueeze(-1) * x).sum(1)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden=16, use_hist=False):
        super().__init__()
        self.use_hist = use_hist
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True,
                             bidirectional=True)
        self.attn = Attention(hidden * 2)
        self.drop = nn.Dropout(0.5)
        fc_in = hidden * 2 + 1 if use_hist else hidden * 2
        self.fc1 = nn.Linear(fc_in, 16)
        self.fc2 = nn.Linear(16, 1)
    def forward(self, x, h=None):
        out, _ = self.lstm(x)
        ctx = self.drop(self.attn(out))
        if self.use_hist and h is not None:
            ctx = torch.cat([ctx, h.view(-1, 1)], dim=1)
        return self.fc2(F.relu(self.fc1(ctx))).squeeze(-1)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden=16, n_classes=2, use_hist=False):
        super().__init__()
        self.use_hist = use_hist
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True,
                             bidirectional=True)
        self.attn = Attention(hidden * 2)
        self.drop = nn.Dropout(0.5)
        fc_in = hidden * 2 + 1 if use_hist else hidden * 2
        self.fc1 = nn.Linear(fc_in, 16)
        self.fc2 = nn.Linear(16, n_classes)
    def forward(self, x, h=None):
        out, _ = self.lstm(x)
        ctx = self.drop(self.attn(out))
        if self.use_hist and h is not None:
            ctx = torch.cat([ctx, h.view(-1, 1)], dim=1)
        return self.fc2(F.relu(self.fc1(ctx)))

def augment(x):
    scale = torch.empty(x.shape[0], 1, 1).uniform_(0.9, 1.1)
    return x * scale + torch.randn_like(x) * 0.05

def train_model(model, Xtr, ytr, loss_fn, hist_tr=None,
                epochs=200, patience=20, augment_on=False):
    nv = max(1, int(len(Xtr) * 0.15))
    Xv, yv = Xtr[-nv:], ytr[-nv:]
    Xt, yt = Xtr[:-nv], ytr[:-nv]
    hv = hist_tr[-nv:] if hist_tr is not None else None
    ht = hist_tr[:-nv] if hist_tr is not None else None
    if len(Xt) == 0:
        return model

    tensors = [torch.tensor(Xt), torch.tensor(yt)]
    if ht is not None:
        tensors.append(torch.tensor(ht))
    ds = TensorDataset(*tensors)
    dl = DataLoader(ds, batch_size=min(16, len(Xt)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    bv, bs, wait = float('inf'), None, 0

    for _ in range(epochs):
        model.train()
        for batch in dl:
            xb, yb = batch[0], batch[1]
            hb = batch[2] if ht is not None else None
            if augment_on:
                xb = augment(xb)
            opt.zero_grad()
            loss_fn(model(xb, hb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hv_t = torch.tensor(hv) if hv is not None else None
            vl   = loss_fn(model(torch.tensor(Xv), hv_t),
                           torch.tensor(yv)).item()
        if vl < bv:
            bv   = vl
            bs   = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if bs:
        model.load_state_dict(bs)
    return model

WEIGHT_GRID = [2, 5, 10, 20]

def run_lopo(split_type, use_hist):
    reg_t, reg_p = [], []
    clf_t, clf_p = [], []
    skipped = 0

    for i in range(n_samples):
        pid = players[i]
        gid = game_ids[i]
        c   = clusters_arr[i]

        # Skip players with < MIN_HIST_GAMES for hist experiments
        if use_hist and games_per_player.get(pid, 0) < MIN_HIST_GAMES:
            skipped += 1; continue

        if split_type == 'all_others':
            tr        = np.where((players!=pid)&(game_ids!=gid))[0]
            same_mask = np.zeros(len(tr), dtype=bool)
        elif split_type == 'same_player':
            tr = np.where((players==pid)&(game_ids!=gid))[0]
            if len(tr) < MIN_TRAIN_GAMES: skipped+=1; continue
            same_mask = np.ones(len(tr), dtype=bool)
        elif split_type == 'combined':
            same      = np.where((players==pid)&(game_ids!=gid))[0]
            othr      = np.where((players!=pid)&(game_ids!=gid))[0]
            tr        = np.concatenate([same, othr])
        elif split_type == 'cluster_only':
            # Include target player's other games + cluster peers
            tr        = np.where((clusters_arr==c)&(game_ids!=gid))[0]
            same_mask = np.zeros(len(tr), dtype=bool)

        if len(tr) == 0: skipped+=1; continue

        # Scale sequences
        sc     = StandardScaler()
        Xtr_2d = sequences[tr].reshape(-1, n_zones)
        Xte_2d = sequences[[i]].reshape(-1, n_zones)
        Xtr    = sc.fit_transform(Xtr_2d).reshape(
                   len(tr), n_steps, n_zones).astype(np.float32)
        Xte    = sc.transform(Xte_2d).reshape(
                   1, n_steps, n_zones).astype(np.float32)

        fold_use_hist = use_hist
        hm_tr = hm_te = None

        if use_hist:
            feat_pid  = feat_df[feat_df['PlayerID'] == pid]
            hm_tr_arr = hist_hsr[tr].copy().astype(np.float32)

            if split_type in ('same_player', 'combined'):
                for j, tr_idx in enumerate(tr):
                    if players[tr_idx] != pid:
                        continue  # other players - keep precomputed
                    tr_gid = game_ids[tr_idx]
                    hm_tr_arr[j] = hist_tr_lookup.get(
                        (pid, tr_gid, gid), np.nan)

                # Test row hist from lookup
                hm_te_val = hist_te_lookup.get((pid, gid), np.nan)
            else:
                hm_te_val = float(hist_hsr[i])

            mv = np.nanmean(hm_tr_arr)
            if np.isnan(mv):
                fold_use_hist = False
            else:
                hm_tr_arr = np.where(np.isnan(hm_tr_arr),
                                      mv, hm_tr_arr).astype(np.float32)
                hm_te_val = mv if np.isnan(hm_te_val) else hm_te_val
                sc_h  = StandardScaler()
                hm_tr = sc_h.fit_transform(
                    hm_tr_arr.reshape(-1, 1)).flatten().astype(np.float32)
                hm_te = sc_h.transform(
                    np.array([[hm_te_val]])).flatten().astype(np.float32)

                torch.manual_seed(42)
        m1 = train_model(
            LSTMRegressor(n_zones, use_hist=fold_use_hist),
            Xtr, y_hsr[tr], nn.L1Loss(), hist_tr=hm_tr,
            epochs=200, patience=20)
        m1.eval()
        with torch.no_grad():
            ht = torch.tensor(hm_te) if hm_te is not None else None
            reg_p.append(float(m1(torch.tensor(Xte), ht).item()))
        reg_t.append(float(y_hsr[i]))

        if len(np.unique(y_clf_lbl[tr])) < 2:
            clf_t.append(y_clf_lbl[i])
            clf_p.append('Decline')
        else:
            fold_le   = LabelEncoder()
            fold_le.fit(y_clf_lbl[tr])
            y_clf_enc = fold_le.transform(
                y_clf_lbl[tr]).astype(np.int64)
            n_cls     = len(fold_le.classes_)
            counts    = np.bincount(
                y_clf_enc, minlength=n_cls).astype(np.float32)
            weights   = torch.tensor(1.0 / np.sqrt(counts + 1e-6))
            weights   = (weights / weights.sum()) * n_cls
            torch.manual_seed(42)
            m3 = train_model(
                LSTMClassifier(n_zones, n_classes=n_cls,
                                use_hist=fold_use_hist),
                Xtr, y_clf_enc,
                nn.CrossEntropyLoss(weight=weights),
                hist_tr=hm_tr,
                epochs=300, patience=25, augment_on=True)
            m3.eval()
            with torch.no_grad():
                ht   = torch.tensor(hm_te) if hm_te is not None else None
                pred = int(torch.argmax(
                    m3(torch.tensor(Xte), ht), dim=1).item())
            clf_p.append(fold_le.inverse_transform([pred])[0])
            clf_t.append(y_clf_lbl[i])

        if (i + 1) % 10 == 0:
            h_flag = '|hist' if fold_use_hist else ''
            print(f"  [{split_type}{h_flag}] "
                  f"{i+1}/{n_samples} n_train={len(tr)}", flush=True)

    if skipped:
        print(f"  [{split_type}{'|hist' if use_hist else ''}] skipped {skipped}")
    return reg_t, reg_p, clf_t, clf_p

# Note: weighted split excluded - per-sample weight application via batch
# index arithmetic is unreliable with DataLoader shuffling. PyTorch
# WeightedRandomSampler would be required for a correct implementation.
# Tabular model weighted results (scripts 02, 03) remain valid.
SPLITS = ['all_others', 'same_player', 'combined', 'cluster_only']

for use_hist, feat_type in [(False, 'multivariate'),
                             (True,  'multivariate+hist')]:
    print(f"\n{'='*55}")
    print(f"Features: {feat_type}")
    print(f"Architecture: BiLSTM hidden=16 | attention | dropout=0.5")

    for split in SPLITS:
        print(f"\nSplit: {split}")
        rt, rp, ct, cp = run_lopo(split, use_hist)
        if not rt:
            print("  No predictions.")
            continue
        hsr_m = reg_metrics(rt, rp)
        clf_m = clf_metrics(ct, cp)
        print(f"  LSTM (n={len(rt)}): "
              f"MAE {hsr_m['mae']} R² {hsr_m['r2']} | "
              f"F1 {clf_m['f1']}")
        save_results('04', 'LSTM', feat_type, split, hsr_m, clf_m)

print("\nDone.")