import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'gaa_actions.csv')
OUTPUTS_DIR  = os.path.join(BASE_DIR, '..', 'outputs') + os.sep
RESULTS_PATH = OUTPUTS_DIR + 'results.csv'

SEQ_PATH      = OUTPUTS_DIR + 'gaa_sequences.npy'
SEQ_MV_PATH   = OUTPUTS_DIR + 'gaa_sequences_mv.npy'
SEQ_META_PATH = OUTPUTS_DIR + 'gaa_sequences_meta.csv'
FEATURES_PATH = OUTPUTS_DIR + 'gaa_features_final.csv'
CLUSTERS_PATH = OUTPUTS_DIR + 'player_clusters.csv'
HIST_PATH     = OUTPUTS_DIR + 'hist_corrected.csv'

COVERAGE_THRESHOLD = 0.98
MZ_THRESHOLD       = 3.5
Q_SPLIT_SECONDS    = 1050
WINDOW_S           = 300
STEP_S             = 60
N_STEPS            = 50

HSR_ACTIONS  = {'Running', 'High Intensity Running', 'Sprint'}
ACTION_ZONES = ['Standing', 'Walking', 'Jogging',
                'Running', 'High Intensity Running', 'Sprint']

LABEL_ORDER  = ['Decline', 'Improve']
FEATURE_COLS = ['series_mean', 'series_std', 'series_last',
                'series_slope', 'last10_slope']
HIST_FEAT_HSR   = 'hist_mean_q4_mpm'
TOP_FEATURES    = 5
MIN_TRAIN_GAMES = 2
MIN_HIST_GAMES  = 3