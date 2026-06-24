import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from collections import Counter

from pmdarima import auto_arima

from config import (SEQ_PATH, SEQ_META_PATH, FEATURES_PATH,
                    HIST_FEAT_HSR, FEATURE_COLS)
from utils import reg_metrics, clf_metrics, save_results

sequences = np.load(SEQ_PATH)
meta      = pd.read_csv(SEQ_META_PATH)
feat_df   = pd.read_csv(FEATURES_PATH)

meta = meta.merge(
    feat_df[['GameID','PlayerID', HIST_FEAT_HSR, 'series_mean']],
    on=['GameID','PlayerID'], how='left')

n_samples, n_steps, _ = sequences.shape
y_hsr       = meta['Q4_HSR_per_min'].values
y_clf_lbl   = meta['Q4_label_binary'].values
hist_hsr    = meta[HIST_FEAT_HSR].values.astype(float)
series_mean = meta['series_mean'].values.astype(float)

Q4_STEPS = 18  # ~18 min of Q4 at 1-min resolution

# Exponential decay weights - step 1 weighted most, step 18 least
_w = np.exp(-0.1 * np.arange(Q4_STEPS))
EXP_WEIGHTS = _w / _w.sum()

print(f"Samples: {n_samples} | Sequence length: {n_steps}")
print(f"Forecasting {Q4_STEPS} steps")
print(f"With hist: {(~np.isnan(hist_hsr)).sum()}/{n_samples}")

def hsr_to_label(pred_hsr, series_m):
    """Classify: Improve if predicted Q4 >= current game Q1-Q3, else Decline."""
    return 'Improve' if pred_hsr >= series_m else 'Decline'

VARIANTS = ['arima_next1', 'arima_equal', 'arima_weighted',
            'arimax_equal', 'arimax_weighted']
store = {v: {'hsr_t':[], 'hsr_p':[], 'clf_t':[], 'clf_p':[]}
         for v in VARIANTS}
orders = []
failed = 0
arimax_fb = 0

print("\nRunning ARIMA variants...")

for i in range(n_samples):
    series   = sequences[i,:,0].astype(np.float64)
    hist_val = hist_hsr[i]
    s_mean   = series_mean[i]
    has_hist = not np.isnan(hist_val)

    try:
        m = auto_arima(series, start_p=0, max_p=4,
                       start_q=0, max_q=4, d=None,
                       seasonal=False, information_criterion='aic',
                       stepwise=True, suppress_warnings=True,
                       error_action='ignore', max_order=6)
        orders.append(m.order)
        fc      = m.predict(n_periods=Q4_STEPS)
        pred_n1 = float(np.clip(fc[0], 0, None))                    # next-1
        pred_a  = float(np.mean(np.clip(fc, 0, None)))              # equal
        pred_b  = float(np.dot(np.clip(fc, 0, None), EXP_WEIGHTS))  # weighted
    except Exception:
        pred_n1 = pred_a = pred_b = float(np.mean(series))
        orders.append((0,0,0))
        failed += 1

    try:
        if has_hist:
            X_tr = np.full((len(series), 1), hist_val)
            X_fu = np.full((Q4_STEPS, 1),   hist_val)
            mx   = auto_arima(series, X=X_tr, start_p=0, max_p=4,
                              start_q=0, max_q=4, d=None,
                              seasonal=False, information_criterion='aic',
                              stepwise=True, suppress_warnings=True,
                              error_action='ignore', max_order=6)
            fc_x    = mx.predict(n_periods=Q4_STEPS, X=X_fu)
            pred_xe = float(np.mean(np.clip(fc_x, 0, None)))
            pred_xw = float(np.dot(np.clip(fc_x, 0, None), EXP_WEIGHTS))
        else:
            pred_xe = pred_a
            pred_xw = pred_b
            arimax_fb += 1
    except Exception:
        pred_xe = pred_a
        pred_xw = pred_b
        arimax_fb += 1

        for key, pred in [('arima_next1',    pred_n1),
                       ('arima_equal',    pred_a),
                       ('arima_weighted', pred_b),
                       ('arimax_equal',   pred_xe),
                       ('arimax_weighted',pred_xw)]:
            store[key]['hsr_t'].append(float(y_hsr[i]))
            store[key]['hsr_p'].append(pred)
            store[key]['clf_t'].append(y_clf_lbl[i])
            store[key]['clf_p'].append(hsr_to_label(pred, s_mean))

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{n_samples} (order: {orders[-1]})", flush=True)

print(f"\nFailed: {failed} | ARIMAX fallback (no hist): {arimax_fb}")
print(f"Top orders: {Counter(orders).most_common(5)}")

print(f"\n{'='*55}")
print(f"{'Variant':<22} {'MAE':>7} {'R²':>7} {'F1':>7}")
print(f"{'-'*55}")

metrics = {}
for key, label in [
    ('arima_next1',    'ARIMA next-1'),
    ('arima_equal',    'ARIMA equal'),
    ('arima_weighted', 'ARIMA weighted'),
    ('arimax_equal',   'ARIMAX equal'),
    ('arimax_weighted','ARIMAX weighted'),
]:
    s          = store[key]
    hm         = reg_metrics(s['hsr_t'], s['hsr_p'])
    cm         = clf_metrics(s['clf_t'], s['clf_p'])
    metrics[key] = (hm, cm)
    model_type = 'ARIMAX' if 'arimax' in key else 'ARIMA'
    print(f"  {label:<22} {hm['mae']:>7.3f} {hm['r2']:>7.3f} {cm['f1']:>7.3f}")
    save_results('05', model_type, key, 'arima_forecast', hm, cm)

best_key = min(metrics, key=lambda k: metrics[k][0]['mae'])
print(f"\n  Best: {best_key} (MAE {metrics[best_key][0]['mae']})")
print(f"  Naive series_mean baseline: MAE ~5.779  R² ~0.273")

print(f"\n-- Residual bias (+ = overprediction):")
for key, label in [
    ('arima_next1',    'ARIMA next-1'),
    ('arima_equal',    'ARIMA equal'),
    ('arima_weighted', 'ARIMA weighted'),
    ('arimax_equal',   'ARIMAX equal'),
    ('arimax_weighted','ARIMAX weighted'),
]:
    res = np.array(store[key]['hsr_p']) - np.array(store[key]['hsr_t'])
    print(f"  {label:<22}: bias={res.mean():+.3f}  std={res.std():.3f}")

print("\nDone.")