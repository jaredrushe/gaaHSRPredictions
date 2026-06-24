import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from config import OUTPUTS_DIR, SEQ_PATH, SEQ_META_PATH

seq = np.load(SEQ_PATH)
meta = pd.read_csv(SEQ_META_PATH)

# Pick a representative sequence (most median Q4 HSR)
target_idx = (meta['Q4_HSR_per_min'] - meta['Q4_HSR_per_min'].median()).abs().idxmin()
series = seq[target_idx, :, 0]

fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(series, lags=40, ax=ax, color='#378ADD', vlines_kwargs={'colors': '#378ADD'})
ax.set_title('ACF of Pre-Q4 HSR Sequence')
ax.set_xlabel('Lag (windows)')
ax.set_ylabel('Autocorrelation')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'acf_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: acf_analysis.png")