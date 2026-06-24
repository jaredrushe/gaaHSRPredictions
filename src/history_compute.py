import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from config import FEATURES_PATH, OUTPUTS_DIR, MIN_HIST_GAMES, HIST_FEAT_HSR

feat_df = pd.read_csv(FEATURES_PATH)
players = feat_df['PlayerID'].unique()
games   = feat_df['GameID'].unique()

rows = []
for pid in players:
    p_df = feat_df[feat_df['PlayerID'] == pid]
    p_games = p_df['GameID'].tolist()
    for te_gid in p_games:
        other_games = [g for g in p_games if g != te_gid]
        if len(other_games) == 0:
            hist_mpm_te = np.nan
        else:
            hist_mpm_te = p_df[p_df['GameID'].isin(other_games)]['Q4_HSR_per_min'].mean()

        for tr_gid in p_games:
            if tr_gid == te_gid:
                continue
            excl = [te_gid, tr_gid]
            remaining = [g for g in p_games if g not in excl]
            hist_mpm = np.nan if len(remaining) < 1 else \
                       p_df[p_df['GameID'].isin(remaining)]['Q4_HSR_per_min'].mean()
            rows.append({
                'PlayerID':   pid,
                'tr_gid':     tr_gid,
                'te_gid':     te_gid,
                'hist_mpm':   hist_mpm,
                'hist_mpm_te': hist_mpm_te,
            })

out = pd.DataFrame(rows)
out.to_csv(OUTPUTS_DIR + 'hist_corrected.csv', index=False)
print(f"Saved hist_corrected.csv ({len(out)} rows)")