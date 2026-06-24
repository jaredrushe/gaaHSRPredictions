import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor, XGBClassifier

from config import (SEQ_PATH, SEQ_META_PATH, FEATURES_PATH, CLUSTERS_PATH,
                    LABEL_ORDER, HIST_FEAT_HSR,
                    MIN_TRAIN_GAMES, MIN_HIST_GAMES, TOP_FEATURES)
from utils import reg_metrics, clf_metrics, save_results, load_corrected_hist

sequences = np.load(SEQ_PATH)
meta      = pd.read_csv(SEQ_META_PATH)
feat_df   = pd.read_csv(FEATURES_PATH)
clusters  = pd.read_csv(CLUSTERS_PATH)[['PlayerID','cluster']]

meta = meta.merge(clusters, on='PlayerID', how='left')
meta = meta.merge(
    feat_df[['GameID','PlayerID', HIST_FEAT_HSR]],
    on=['GameID','PlayerID'], how='left')

# Q4_HSR_per_min comes from gaa_sequences_meta.csv directly
# Q4_label_binary comes from gaa_sequences_meta.csv directly

n_samples, n_steps, _ = sequences.shape
players      = meta['PlayerID'].values
game_ids     = meta['GameID'].values
clusters_arr = meta['cluster'].values
y_hsr        = meta['Q4_HSR_per_min'].values
y_clf        = meta['Q4_label_binary'].values
hist_hsr_arr = meta[HIST_FEAT_HSR].values.astype(float)

# Games per player - used to filter hist experiments
games_per_player = feat_df.groupby('PlayerID')['GameID'].count()

# Load corrected hist lookups
hist_tr_lookup, hist_te_lookup = load_corrected_hist()

print(f"Samples: {n_samples} | Timesteps: {n_steps}")

print("Extracting TSFresh features...")
rows_ts = [{'id': i, 'time': t, 'value': float(sequences[i,t,0])}
           for i in range(n_samples) for t in range(n_steps)]
extracted = extract_features(
    pd.DataFrame(rows_ts), column_id='id', column_sort='time',
    column_value='value',
    default_fc_parameters=EfficientFCParameters(),
    n_jobs=1, disable_progressbar=False)
extracted = (extracted.astype(float)
             .replace([np.inf,-np.inf], np.nan)
             .fillna(extracted.median())
             .dropna(axis=1, how='all'))
extracted = extracted.loc[:, extracted.std() > 0]
print(f"Extracted: {extracted.shape[1]} features after cleaning")

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

def fdr_filter(ext_tr, y_enc_tr):
    try:
        fdr = select_features(ext_tr,
                               pd.Series(y_enc_tr, index=ext_tr.index),
                               fdr_level=0.05, n_jobs=1)
        if fdr.shape[1] < TOP_FEATURES:
            fdr = select_features(ext_tr,
                                   pd.Series(y_enc_tr, index=ext_tr.index),
                                   fdr_level=0.5, n_jobs=1)
        if fdr.shape[1] == 0:
            fdr = ext_tr
    except Exception:
        fdr = ext_tr
    return fdr

def top_k(fdr_tr, ytr, task, k):
    sc = StandardScaler()
    Xf = sc.fit_transform(fdr_tr.values)
    if task == 'clf':
        m = LogisticRegression(max_iter=200, C=0.1, class_weight='balanced')
        m.fit(Xf, ytr)
        imps = pd.Series(np.abs(m.coef_).mean(axis=0), index=fdr_tr.columns)
    else:
        m = Ridge(alpha=1.0)
        m.fit(Xf, ytr)
        imps = pd.Series(np.abs(m.coef_), index=fdr_tr.columns)
    return imps.nlargest(min(k, len(imps))).index.tolist()

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
            same_mask = np.concatenate([np.ones(len(same), dtype=bool),
                                        np.zeros(len(othr), dtype=bool)])
        elif split_type == 'weighted':
            same      = np.where((players==pid)&(game_ids!=gid))[0]
            othr      = np.where((players!=pid)&(game_ids!=gid))[0]
            tr        = np.concatenate([same, othr])
            same_mask = np.concatenate([np.ones(len(same), dtype=bool),
                                        np.zeros(len(othr), dtype=bool)])
        elif split_type == 'cluster_only':
            # Include target player's other games + cluster peers
            tr        = np.where((clusters_arr==c)&(game_ids!=gid))[0]
            same_mask = np.zeros(len(tr), dtype=bool)

        if len(tr) == 0: skipped+=1; continue
        if len(np.unique(y_clf[tr])) < 2: skipped+=1; continue

        k_ts = TOP_FEATURES - 1 if use_hist else TOP_FEATURES

        hist_hsr_tr  = hist_hsr_arr[tr].copy().astype(float)
        hist_hsr_te  = float(hist_hsr_arr[i])

        if use_hist and split_type in ('same_player', 'combined'):
            for j, tr_idx in enumerate(tr):
                if players[tr_idx] != pid:
                    continue  # other players - already clean
                tr_gid = game_ids[tr_idx]
                hist_hsr_tr[j] = hist_tr_lookup.get(
                    (pid, tr_gid, gid), np.nan)

            # Test row hist from lookup
            hist_hsr_te = hist_te_lookup.get((pid, gid), np.nan)

        hist_hsr_tr_col = hist_hsr_tr.reshape(-1, 1)
        mv = np.nanmean(hist_hsr_tr_col)
        hist_hsr_tr_col = np.where(np.isnan(hist_hsr_tr_col),
                                    mv, hist_hsr_tr_col)
        hist_hsr_te_val = mv if np.isnan(hist_hsr_te) else hist_hsr_te

        fold_le  = LabelEncoder()
        fold_le.fit(y_clf[tr])
        y_enc_tr = fold_le.transform(y_clf[tr])

        ext_tr = extracted.iloc[tr]
        ext_te = extracted.iloc[[i]]
        fdr    = fdr_filter(ext_tr, y_enc_tr)
        fdr_te = ext_te[fdr.columns]

        for mname, reg_fn, clf_fn in FIXED_MODELS:
            if mname not in res:
                res[mname] = {t: {'t':[],'p':[]} for t in ['hsr','clf']}

            for task, y_tr_arr, y_te_val in [
                ('hsr', y_hsr[tr], y_hsr[i]),
                ('clf', y_clf[tr], y_clf[i]),
            ]:
                is_clf = task == 'clf'

                f = top_k(fdr,
                           y_enc_tr if is_clf else y_tr_arr.astype(float),
                           task, k_ts)
                base_tr = fdr[f].values.astype(np.float32)
                base_te = fdr_te[f].values.astype(np.float32)

                if use_hist:
                    Xtr_raw = np.hstack([base_tr, hist_hsr_tr_col])
                    Xte_raw = np.hstack([base_te,
                                         [[hist_hsr_te_val]]])
                else:
                    Xtr_raw, Xte_raw = base_tr, base_te

                sc  = StandardScaler()
                Xtr = sc.fit_transform(Xtr_raw)
                Xte = sc.transform(Xte_raw)

                sw = None
                if split_type == 'weighted':
                    y_tune = (y_clf[tr] if is_clf
                              else y_tr_arr.astype(float))
                    sw = np.where(same_mask, float(
                        tune_weight(Xtr, y_tune, same_mask, task)), 1.0)

                if is_clf:
                    m  = clf_fn()
                    kw = {'sample_weight': sw} if sw is not None else {}
                    if isinstance(m, LogisticRegression):
                        m.fit(Xtr, y_clf[tr], **kw)
                        pred = m.predict(Xte)[0]
                    else:
                        m.fit(Xtr, y_enc_tr, **kw)
                        pred = fold_le.inverse_transform(m.predict(Xte))[0]
                    res[mname]['clf']['t'].append(y_te_val)
                    res[mname]['clf']['p'].append(pred)
                else:
                    m  = reg_fn()
                    kw = {'sample_weight': sw} if sw is not None else {}
                    m.fit(Xtr, y_tr_arr.astype(float), **kw)
                    res[mname]['hsr']['t'].append(float(y_te_val))
                    res[mname]['hsr']['p'].append(float(m.predict(Xte)[0]))

        if (i+1) % 10 == 0:
            print(f"  [{split_type}{'|hist' if use_hist else ''}] "
                  f"{i+1}/{n_samples} n_train={len(tr)} k={k_ts}",
                  flush=True)

    if skipped:
        print(f"  [{split_type}] skipped {skipped}")
    return res

SPLITS    = ['all_others','same_player','combined','weighted','cluster_only']
FEAT_SETS = [('tsfresh', False), ('tsfresh+hist', True)]

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
            save_results('03', mname, feat_name, split, hsr_m, clf_m)

print("\nDone.")