import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from config import FEATURES_PATH
from utils import reg_metrics, clf_metrics, save_results

df = pd.read_csv(FEATURES_PATH)

print(f"Samples: {len(df)} | Players: {df['PlayerID'].nunique()}")
print(f"Labels:\n{df['Q4_label_binary'].value_counts()}\n")

y_hsr = df['Q4_HSR_per_min'].values
y_clf = df['Q4_label_binary'].values

p_hsr = df['series_mean'].values
p_clf = np.array(['Decline'] * len(df))

hsr_m = reg_metrics(y_hsr, p_hsr)
clf_m = clf_metrics(y_clf, p_clf)

print("Naive Baseline:")
print(f"  Regression  MAE {hsr_m['mae']}  RMSE {hsr_m['rmse']}  R² {hsr_m['r2']}")
print(f"  Clf         Acc {clf_m['accuracy']}  F1 {clf_m['f1']}  "
      f"Prec {clf_m['precision']}  Rec {clf_m['recall']}")

save_results('01', 'Naive', 'series_mean', 'all', hsr_m, clf_m)

print("\nDone.")