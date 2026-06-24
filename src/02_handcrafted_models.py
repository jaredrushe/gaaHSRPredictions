import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor, XGBClassifier

from config import (FEATURES_PATH, CLUSTERS_PATH, LABEL_ORDER,
                    FEATURE_COLS, HIST_FEAT_HSR,
                    MIN_TRAIN_GAMES, MIN_HIST_GAMES)
from utils import reg_metrics, clf_metrics, save_results, load_corrected_hist

df       = pd.read_csv(FEATURES_PATH)
clusters = pd.read_csv(CLUSTERS_PATH)[['PlayerID','cluster']]
df       = df.merge(clusters, on='PlayerID', how='left')

le = LabelEncoder(); le.fit(LABEL_ORDER)

print(f"Samples: {len(df)} | Players: {df['PlayerID'].nunique()}")
print(f"Labels:\n{df['Q4_label_binary'].value_counts()}\n")

#Load corrected hist lookups (precomputed by 00b_precompute_hist.py)
hist_tr_lookup, hist_te_lookup = load_corrected_hist()

# Games per player - used to filter hist experiments
games_per_player = df.groupby('PlayerID')['GameID'].count()

WEIGHT_GRID = [2, 5, 10, 20]

def make_ridge():     return Ridge(alpha=1.0)
def make_ridge_clf(): return LogisticRegression(max_iter=1000, C=0.1,
                                                 class_weight='balanced')
def make_rf():        return RandomForestRegressor(n_estimators=200,
                                                    max_depth=5, random_state=42)
def make_rf_clf():    return RandomForestClassifier(n_estimators=200,
                                                     max_depth=5,
                                                     class_weight='balanced',
                                                     random_state=42)
def make_xgb():       return XGBRegressor(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, subsample=0.8,
                                           random_state=42, verbosity=0)
def make_xgb_clf():   return XGBClassifier(n_estimators=100, max_depth=3,
                                            learning_rate=0.1, subsample=0.8,
                                            random_state=42, verbosity=0,
                                            eval_metric='logloss')

FIXED_MODELS = [
    ('Ridge',   make_ridge,   make_ridge_clf),
    ('RF',      make_rf,      make_rf_clf),
    ('XGBoost', make_xgb,     make_xgb_clf),
]

def tune_weight(Xtr, ytr, same_mask, task):
    if same_mask.sum() == 0 or len(Xtr) < 6:
        return 1
    nv    = max(1, int(len(Xtr) * 0.2))
    Xv,yv = Xtr[-nv:], ytr[-nv:]
    Xt,yt = Xtr[:-nv], ytr[:-nv]
    sm    = same_mask[:-nv]
    best_w, best_s = 1, float('inf')
    for w in WEIGHT_GRID:
        sw = np.where(sm, float(w), 1.0)
        try:
            if task == 'clf':
                m = LogisticRegression(max_iter=500, C=0.1,
                                       class_weight='balanced')
                m.fit(Xt, yt, sample_weight=sw)
                score = -np.mean(m.predict(Xv) == yv)
            else:
                m = Ridge(alpha=1.0)
                m.fit(Xt, yt.astype(float), sample_weight=sw)
                score = np.mean(np.abs(m.predict(Xv) - yv.astype(float)))
            if score < best_s:
                best_s, best_w = score, w
        except Exception:
            continue
    return best_w

def run_lopo(split_type, use_hist):
    res     = {}
    skipped = 0

    for idx, row in df.iterrows():
        pid = row['PlayerID']
        gid = row['GameID']
        c   = row['cluster']

        # Skip players with < MIN_HIST_GAMES for hist experiments
        if use_hist and games_per_player[pid] < MIN_HIST_GAMES:
            skipped += 1; continue

        if split_type == 'all_others':
            train_df  = df[(df['PlayerID'] != pid) & (df['GameID'] != gid)]
            same_mask = np.zeros(len(train_df), dtype=bool)
        elif split_type == 'same_player':
            train_df  = df[(df['PlayerID'] == pid) & (df['GameID'] != gid)]
            same_mask = np.ones(len(train_df), dtype=bool)
            if len(train_df) < MIN_TRAIN_GAMES:
                skipped += 1; continue
        elif split_type == 'combined':
            same      = df[(df['PlayerID'] == pid) & (df['GameID'] != gid)]
            others    = df[(df['PlayerID'] != pid) & (df['GameID'] != gid)]
            train_df  = pd.concat([same, others]).reset_index(drop=True)
            same_mask = np.concatenate([np.ones(len(same),  dtype=bool),
                                        np.zeros(len(others),dtype=bool)])
        elif split_type == 'weighted':
            same      = df[(df['PlayerID'] == pid) & (df['GameID'] != gid)]
            others    = df[(df['PlayerID'] != pid) & (df['GameID'] != gid)]
            train_df  = pd.concat([same, others]).reset_index(drop=True)
            same_mask = np.concatenate([np.ones(len(same),  dtype=bool),
                                        np.zeros(len(others),dtype=bool)])
        elif split_type == 'cluster_only':
            # Include target player's other games + cluster peers
            train_df  = df[(df['cluster'] == c) &
                           (df['GameID']  != gid)]
            same_mask = np.zeros(len(train_df), dtype=bool)

        if len(train_df) == 0:
            skipped += 1; continue
        if len(np.unique(train_df['Q4_label_binary'].values)) < 2:
            skipped += 1; continue

            test_row = df.loc[[idx]].copy()

        if use_hist and split_type in ('same_player', 'combined'):
            train_df = train_df.copy().reset_index(drop=True)

            for i in range(len(train_df)):
                tr_pid = train_df.loc[i, 'PlayerID']
                if tr_pid != pid:
                    continue  # other players - already clean
                tr_gid = train_df.loc[i, 'GameID']
                # O(1) lookup - NaN for 2-game players
                val = hist_tr_lookup.get((pid, tr_gid, gid), np.nan)
                train_df.loc[i, HIST_FEAT_HSR] = val

            # Test row hist from precomputed lookup
            te_val = hist_te_lookup.get((pid, gid), np.nan)
            test_row[HIST_FEAT_HSR] = te_val

        fold_le = LabelEncoder()
        fold_le.fit(train_df['Q4_label_binary'].values)
        y_enc   = fold_le.transform(train_df['Q4_label_binary'].values)

        for mname, reg_fn, clf_fn in FIXED_MODELS:
            if mname not in res:
                res[mname] = {t: {'t':[],'p':[]}
                               for t in ['hsr','clf']}

            for task, target in [('hsr', 'Q4_HSR_per_min'),
                                  ('clf', 'Q4_label_binary')]:
                X_tr = train_df[FEATURE_COLS].values.astype(np.float64)
                X_te = test_row[FEATURE_COLS].values.astype(np.float64)

                if use_hist:
                    h_tr = train_df[HIST_FEAT_HSR].values.astype(
                        np.float64).reshape(-1, 1)
                    h_te = test_row[HIST_FEAT_HSR].values.astype(
                        np.float64).reshape(-1, 1)
                    hm   = np.nanmean(h_tr)
                    h_tr = np.where(np.isnan(h_tr), hm, h_tr)
                    h_te = np.where(np.isnan(h_te), hm, h_te)
                    X_tr = np.hstack([X_tr, h_tr])
                    X_te = np.hstack([X_te, h_te])

                sc  = StandardScaler()
                Xtr = sc.fit_transform(X_tr)
                Xte = sc.transform(X_te)

                sw = None
                if split_type == 'weighted':
                    y_tune = (train_df['Q4_label_binary'].values
                              if task == 'clf'
                              else train_df[target].values.astype(float))
                    sw = np.where(same_mask, float(
                        tune_weight(Xtr, y_tune, same_mask, task)), 1.0)

                if task == 'clf':
                    m  = clf_fn()
                    kw = {'sample_weight': sw} if sw is not None else {}
                    if isinstance(m, LogisticRegression):
                        m.fit(Xtr, train_df['Q4_label_binary'].values, **kw)
                        pred = m.predict(Xte)[0]
                    else:
                        m.fit(Xtr, y_enc, **kw)
                        pred = fold_le.inverse_transform(m.predict(Xte))[0]
                    res[mname]['clf']['t'].append(row['Q4_label_binary'])
                    res[mname]['clf']['p'].append(pred)
                else:
                    m  = reg_fn()
                    kw = {'sample_weight': sw} if sw is not None else {}
                    m.fit(Xtr, train_df[target].values.astype(float), **kw)
                    res[mname]['hsr']['t'].append(float(row[target]))
                    res[mname]['hsr']['p'].append(float(m.predict(Xte)[0]))

    if skipped:
        print(f"  [{split_type}{'|hist' if use_hist else ''}] skipped {skipped}")
    return res

SPLITS    = ['all_others','same_player','combined','weighted','cluster_only']
FEAT_SETS = [('hc', False), ('hc+hist', True)]

for split in SPLITS:
    print(f"\n{'='*55}\nSplit: {split}")
    for feat_name, use_hist in FEAT_SETS:
        print(f"  Features: {feat_name}")
        res = run_lopo(split, use_hist)
        for mname, tasks in res.items():
            if not tasks['hsr']['t']: continue
            hsr_m = reg_metrics(tasks['hsr']['t'], tasks['hsr']['p'])
            clf_m = clf_metrics(tasks['clf']['t'],  tasks['clf']['p'])
            n = len(tasks['hsr']['t'])
            print(f"    {mname} (n={n}): "
                  f"MAE {hsr_m['mae']} R² {hsr_m['r2']} | "
                  f"F1 {clf_m['f1']}")
            save_results('02', mname, feat_name, split, hsr_m, clf_m)

print("\nDone.")