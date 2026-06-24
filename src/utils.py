import os
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, f1_score, accuracy_score,
                              precision_score, recall_score)
from config import RESULTS_PATH, LABEL_ORDER, HIST_PATH

STAT_PREFIX  = '---'
METRIC_COLS  = ['reg_mae', 'reg_rmse', 'reg_r2',
                'clf_accuracy', 'clf_f1', 'clf_precision', 'clf_recall']
LOWER_BETTER = {'reg_mae', 'reg_rmse'}


def ratio_to_label(ratio):
    return 'Improve' if float(ratio) >= 1.0 else 'Decline'


def reg_metrics(true, pred):
    t, p = np.array(true, dtype=float), np.array(pred, dtype=float)
    return {
        'mae':  round(float(mean_absolute_error(t, p)), 3),
        'rmse': round(float(np.sqrt(mean_squared_error(t, p))), 3),
        'r2':   round(float(r2_score(t, p)), 3),
    }


def clf_metrics(true, pred):
    return {
        'accuracy':  round(float(accuracy_score(true, pred)), 3),
        'f1':        round(float(f1_score(true, pred, labels=LABEL_ORDER,
                                          average='weighted', zero_division=0)), 3),
        'precision': round(float(precision_score(true, pred, labels=LABEL_ORDER,
                                                  average='weighted', zero_division=0)), 3),
        'recall':    round(float(recall_score(true, pred, labels=LABEL_ORDER,
                                               average='weighted', zero_division=0)), 3),
    }


def _load():
    if not os.path.exists(RESULTS_PATH):
        return None
    return pd.read_csv(RESULTS_PATH)


def save_results(script, model_type, features_type, split_type, hsr_m, clf_m):
    new_row = {
        'script': script, 'model_type': model_type,
        'features_type': features_type, 'split_type': split_type,
        'reg_mae': hsr_m['mae'], 'reg_rmse': hsr_m['rmse'],
        'reg_r2': hsr_m['r2'], 'clf_accuracy': clf_m['accuracy'],
        'clf_f1': clf_m['f1'], 'clf_precision': clf_m['precision'],
        'clf_recall': clf_m['recall'],
    }
    df = _load()
    if df is not None:
        stat_rows  = df[df['model_type'].astype(str).str.startswith(STAT_PREFIX)]
        model_rows = df[~df['model_type'].astype(str).str.startswith(STAT_PREFIX)]
        mask = (
            (model_rows['script'].astype(str)        == str(script)) &
            (model_rows['model_type'].astype(str)    == str(model_type)) &
            (model_rows['features_type'].astype(str) == str(features_type)) &
            (model_rows['split_type'].astype(str)    == str(split_type))
        )
        model_rows = pd.concat([model_rows[~mask], pd.DataFrame([new_row])],
                               ignore_index=True)
        df = pd.concat([model_rows, stat_rows], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(RESULTS_PATH, index=False, encoding='utf-8-sig')
    print(f"  -> [{script} | {model_type} | {features_type} | {split_type}]")


def update_stat_rows(flat_df):
    hsr = flat_df['Q4_HSR_per_min']
    lbl = flat_df['Q4_label_binary']
    stat_rows = [
        {
            'script': STAT_PREFIX,
            'model_type': '--- Q4 HSR per min (m/min) ---',
            'features_type': f'n={len(hsr)}',
            'split_type': '',
            'reg_mae': f'mean={hsr.mean():.2f}',
            'reg_rmse': f'std={hsr.std():.2f}',
            'reg_r2': f'median={hsr.median():.2f}',
            'clf_accuracy': f'min={hsr.min():.2f} max={hsr.max():.2f}',
            'clf_f1': f'IQR={hsr.quantile(0.25):.2f}-{hsr.quantile(0.75):.2f}',
            'clf_precision': f'skew={hsr.skew():.3f}',
            'clf_recall': '',
        },
        {
            'script': STAT_PREFIX,
            'model_type': '--- Q4 label (binary) ---',
            'features_type': f'n={len(lbl)}',
            'split_type': '',
            'reg_mae': f'Decline={(lbl=="Decline").sum()} ({(lbl=="Decline").mean():.1%})',
            'reg_rmse': f'Improve={(lbl=="Improve").sum()} ({(lbl=="Improve").mean():.1%})',
            'reg_r2': f'majority_acc={(lbl=="Decline").mean():.3f}',
            'clf_accuracy': 'threshold: ratio>=1=Improve',
            'clf_f1': 'ratio<1=Decline',
            'clf_precision': 'LOPO throughout',
            'clf_recall': '',
        },
    ]
    df = _load()
    if df is not None:
        df = df[~df['model_type'].astype(str).str.startswith(STAT_PREFIX)]
        df = pd.concat([df, pd.DataFrame(stat_rows)], ignore_index=True)
    else:
        df = pd.DataFrame(stat_rows)
    df.to_csv(RESULTS_PATH, index=False, encoding='utf-8-sig')


def load_corrected_hist():
    hdf = pd.read_csv(HIST_PATH)
    hist_tr_lookup = {}
    hist_te_lookup = {}
    for _, row in hdf.iterrows():
        pid    = row['PlayerID']
        tr_gid = row['tr_gid']
        te_gid = row['te_gid']
        hist_tr_lookup[(pid, tr_gid, te_gid)] = (
            float(row['hist_mpm']) if pd.notna(row['hist_mpm']) else np.nan)
        hist_te_lookup[(pid, te_gid)] = (
            float(row['hist_mpm_te']) if pd.notna(row['hist_mpm_te']) else np.nan)
    return hist_tr_lookup, hist_te_lookup