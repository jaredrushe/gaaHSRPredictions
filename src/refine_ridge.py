import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, accuracy_score

from config import (FEATURES_PATH, CLUSTERS_PATH, HIST_FEAT_HSR,
                    FEATURE_COLS, LABEL_ORDER, MIN_HIST_GAMES, OUTPUTS_DIR)
from utils import reg_metrics, clf_metrics, load_corrected_hist

ALPHA_AO = 10.0
ALPHA_CO = 10.0
N_BOOTSTRAP = 1000
REFINED_FEATS  = ['series_mean', 'series_last', 'last10_slope']
ORIGINAL_FEATS = FEATURE_COLS

feat_df  = pd.read_csv(FEATURES_PATH)
clusters = pd.read_csv(CLUSTERS_PATH)[['PlayerID','cluster']]
feat_df  = feat_df.merge(clusters, on='PlayerID', how='left')
hist_tr_lookup, hist_te_lookup = load_corrected_hist()
games_per_player = feat_df.groupby('PlayerID')['GameID'].count()


def bootstrap_ci(trues, preds, metric_fn, n=N_BOOTSTRAP, seed=42):
    np.random.seed(seed)
    t, p = np.array(trues), np.array(preds)
    scores = [metric_fn(t[idx], p[idx])
              for idx in (np.random.choice(len(t),len(t),replace=True) for _ in range(n))]
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores,2.5)), float(np.percentile(scores,97.5))

def mae_fn(t,p):  return mean_absolute_error(t,p)
def r2_fn(t,p):   return r2_score(t,p)
def f1_fn(t,p):   return f1_score(t,p,labels=LABEL_ORDER,average='weighted',zero_division=0)
def acc_fn(t,p):  return accuracy_score(t,p)
def fmt(mean,lo,hi,d=3): return f"{mean:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def ridge_hc_lopo(feat_list, split_type, alpha):
    use_hist = HIST_FEAT_HSR in feat_list
    hc_feats = [f for f in feat_list if f != HIST_FEAT_HSR]
    t_hsr, p_hsr, t_clf, p_clf = [], [], [], []
    for idx, row in feat_df.iterrows():
        pid = row['PlayerID']; gid = row['GameID']
        if use_hist and games_per_player[pid] < MIN_HIST_GAMES: continue
        if split_type == 'all_others':
            train_df = feat_df[(feat_df['PlayerID']!=pid)&(feat_df['GameID']!=gid)]
        elif split_type == 'combined':
            same   = feat_df[(feat_df['PlayerID']==pid)&(feat_df['GameID']!=gid)]
            others = feat_df[(feat_df['PlayerID']!=pid)&(feat_df['GameID']!=gid)]
            train_df = pd.concat([same,others]).reset_index(drop=True)
            if use_hist:
                train_df = train_df.copy()
                for i in range(len(train_df)):
                    if train_df.loc[i,'PlayerID'] != pid: continue
                    tr_gid = train_df.loc[i,'GameID']
                    train_df.loc[i,HIST_FEAT_HSR] = hist_tr_lookup.get((pid,tr_gid,gid),np.nan)
        test_row = feat_df.loc[[idx]].copy()
        if use_hist:
            test_row[HIST_FEAT_HSR] = hist_te_lookup.get((pid,gid), np.nan)
        if len(train_df) == 0: continue
        if len(np.unique(train_df['Q4_label_binary'].values)) < 2: continue
        X_tr = train_df[hc_feats].values.astype(float) if hc_feats else np.zeros((len(train_df),1))
        X_te = test_row[hc_feats].values.astype(float) if hc_feats else np.zeros((1,1))
        if use_hist:
            h_tr = train_df[HIST_FEAT_HSR].values.reshape(-1,1).astype(float)
            h_te = test_row[HIST_FEAT_HSR].values.reshape(-1,1).astype(float)
            hm = np.nanmean(h_tr)
            h_tr = np.where(np.isnan(h_tr),hm,h_tr); h_te = np.where(np.isnan(h_te),hm,h_te)
            X_tr = np.hstack([X_tr,h_tr]); X_te = np.hstack([X_te,h_te])
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_tr); Xte = sc.transform(X_te)
        m_r = Ridge(alpha=alpha); m_r.fit(Xtr, train_df['Q4_HSR_per_min'].values)
        t_hsr.append(float(row['Q4_HSR_per_min'])); p_hsr.append(float(m_r.predict(Xte)[0]))
        m_c = LogisticRegression(max_iter=1000, C=1/alpha, class_weight='balanced')
        m_c.fit(Xtr, train_df['Q4_label_binary'].values)
        t_clf.append(row['Q4_label_binary']); p_clf.append(m_c.predict(Xte)[0])
    return (reg_metrics(t_hsr,p_hsr), clf_metrics(t_clf,p_clf),
            len(t_hsr), t_hsr, p_hsr, t_clf, p_clf)


# Run original and refined for both splits
results = []
for feat_name, feats in [('Original (5 feats)', ORIGINAL_FEATS+[HIST_FEAT_HSR]),
                          ('Refined (3 feats)',  REFINED_FEATS+[HIST_FEAT_HSR])]:
    for split in ['all_others', 'combined']:
        alpha = ALPHA_AO if split == 'all_others' else ALPHA_CO
        hm, cm, n, t_h, p_h, t_c, p_c = ridge_hc_lopo(feats, split, alpha)

        mae_ci  = bootstrap_ci(t_h, p_h, mae_fn)
        r2_ci   = bootstrap_ci(t_h, p_h, r2_fn)
        f1_ci   = bootstrap_ci(t_c, p_c, f1_fn)
        acc_ci  = bootstrap_ci(t_c, p_c, acc_fn)

        results.append({'Model': feat_name, 'Split': split, 'n': n,
                        'MAE': hm['mae'], 'R2': hm['r2'],
                        'F1': cm['f1'], 'Acc': cm['accuracy'],
                        'MAE_CI': fmt(*mae_ci), 'R2_CI': fmt(*r2_ci),
                        'F1_CI': fmt(*f1_ci), 'Acc_CI': fmt(*acc_ci)})
        print(f"{feat_name} | {split} (n={n}):")
        print(f"  MAE {fmt(*mae_ci)}  R2 {fmt(*r2_ci)}")
        print(f"  F1  {fmt(*f1_ci)}  Acc {fmt(*acc_ci)}")

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUTS_DIR + 'refine_ridge_results.csv', index=False)
print(f"\nSaved: refine_ridge_results.csv")


# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Refined Ridge HC -- Original vs Refined Features', fontsize=12, fontweight='bold')

x = np.arange(2); width = 0.3
orig_mae = [df_res[(df_res['Model']=='Original (5 feats)')&(df_res['Split']==s)]['MAE'].values[0]
            for s in ['all_others','combined']]
ref_mae  = [df_res[(df_res['Model']=='Refined (3 feats)')&(df_res['Split']==s)]['MAE'].values[0]
            for s in ['all_others','combined']]

axes[0].bar(x-width/2, orig_mae, width, label='Original (5 feats)', color='#94a3b8', alpha=0.85)
axes[0].bar(x+width/2, ref_mae,  width, label='Refined (3 feats)',  color='#378ADD', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(['all_others','combined'])
axes[0].set_ylabel('MAE (m/min)'); axes[0].set_title('Regression MAE')
axes[0].legend(); axes[0].grid(axis='y', alpha=0.2)
for bars in [axes[0].containers[0], axes[0].containers[1]]:
    for bar in bars:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

orig_f1 = [df_res[(df_res['Model']=='Original (5 feats)')&(df_res['Split']==s)]['F1'].values[0]
           for s in ['all_others','combined']]
ref_f1  = [df_res[(df_res['Model']=='Refined (3 feats)')&(df_res['Split']==s)]['F1'].values[0]
           for s in ['all_others','combined']]
axes[1].bar(x-width/2, orig_f1, width, label='Original (5 feats)', color='#94a3b8', alpha=0.85)
axes[1].bar(x+width/2, ref_f1,  width, label='Refined (3 feats)',  color='#378ADD', alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(['all_others','combined'])
axes[1].set_ylabel('F1 (weighted)'); axes[1].set_title('Classification F1')
axes[1].legend(); axes[1].grid(axis='y', alpha=0.2)
for bars in [axes[1].containers[0], axes[1].containers[1]]:
    for bar in bars:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'refine_ridge.png', dpi=150, bbox_inches='tight')
print("Saved: refine_ridge.png")
print("Done.")