# =============================================================
# config.py — Global Configuration for ESED Framework
# =============================================================
# ESED: Explainable Stacked Ensemble for Disease Diagnosis
# Paper: "ESED: An Explainable Stacked Ensemble Framework
#         for Multi-Class Pulmonary Disease Diagnosis
#         from Chest X-Rays"
# =============================================================

import os

# ── Random seed ───────────────────────────────────────────────
SEED = 42

# ── Classes ───────────────────────────────────────────────────
CLASSES     = ['COVID', 'Pneumonia', 'TB', 'Normal']
NUM_CLASSES = 4

# ── Dataset ───────────────────────────────────────────────────
DATASET_SIZE     = 15898   # after MD5 deduplication
IMAGES_PER_CLASS = 4000    # target per class
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO        = 0.15

# ── Image sizes ───────────────────────────────────────────────
IMG_SIZE_DEFAULT  = 224    # DenseNet201, EfficientNetB4,
                           # ResNet50V2, ConvNeXtTiny
IMG_SIZE_INCEPTION= 299    # InceptionV3 only

# ── Training hyperparameters ──────────────────────────────────
BATCH_SIZE        = 32
PHASE1_EPOCHS     = 20     # frozen base
PHASE2_EPOCHS     = 10     # partial fine-tune
PHASE1_LR         = 1e-4
PHASE2_LR         = 1e-5
UNFREEZE_LAYERS   = 30     # last N layers unfrozen in Phase 2
DROPOUT_RATE_1    = 0.4
DROPOUT_RATE_2    = 0.3
DENSE_UNITS_1     = 512
DENSE_UNITS_2     = 256
EARLY_STOP_PATIENCE = 5
LR_REDUCE_FACTOR  = 0.3
LR_REDUCE_PATIENCE= 3

# ── Ensemble ──────────────────────────────────────────────────
LEVEL1_FEATURES   = NUM_CLASSES * 5  # 20-dim feature vector
META_LEARNER      = 'XGBoost'
CV_FOLDS          = 5

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators'    : 500,
    'max_depth'       : 3,
    'learning_rate'   : 0.05,
    'subsample'       : 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 5,
    'random_state'    : SEED,
    'use_label_encoder': False,
    'eval_metric'     : 'mlogloss',
}

# ── XAI ───────────────────────────────────────────────────────
GRADCAM_LAYERS = {
    'DenseNet201'   : 'conv5_block32_concat',
    'EfficientNetB4': 'top_conv',
    'ResNet50V2'    : 'post_bn',
    'InceptionV3'   : 'mixed10',
    'ConvNeXtTiny'  : 'convnext_tiny_stage_3_block_2_depthwise_conv',
}
SHAP_BACKGROUND_N = 50
LIME_SAMPLES      = 1000
LIME_FEATURES     = 10

# ── Clinical referral ─────────────────────────────────────────
REFERRAL_THRESHOLD     = 0.95
REFERRAL_THRESHOLDS    = [0.70, 0.80, 0.90, 0.95, 0.99]
HIGH_CONF_ERROR_THRESH = 0.90

# ── Kaggle paths (update if running locally) ──────────────────
WORK         = '/kaggle/working/'
MODELS_DIR   = WORK + 'models/'
METRICS_DIR  = WORK + 'metrics/'
FIGURES_DIR  = WORK + 'figures/'
SPLITS_DIR   = WORK
LOGS_DIR     = WORK + 'logs/'

MODELS_INPUT = (
    '/kaggle/input/datasets/mdkawshermahbub/'
    'pulmonary-model-outputs/models/'
)
DATA_INPUT   = (
    '/kaggle/input/datasets/mdkawshermahbub/'
    'pulmonary-merged-data/merged/'
)
SPLITS_INPUT = (
    '/kaggle/input/datasets/mdkawshermahbub/'
    'pulmonary-splits/'
)

# ── Figure subdirectories ─────────────────────────────────────
FIGURE_SUBDIRS = [
    'confusion_matrices', 'gradcam', 'shap',
    'lime', 'roc_curves', 'misclassification',
    'xai_agreement',
]

# ── Model configs ─────────────────────────────────────────────
# (model_name, preprocess_fn_import, img_size, optimiser)
MODEL_CONFIGS = [
    {
        'name'      : 'DenseNet201',
        'img_size'  : 224,
        'optimiser' : 'adam',
        'pre_import': (
            'tensorflow.keras.applications.densenet',
            'preprocess_input'),
    },
    {
        'name'      : 'EfficientNetB4',
        'img_size'  : 224,
        'optimiser' : 'adam',
        'pre_import': (
            'tensorflow.keras.applications.efficientnet',
            'preprocess_input'),
    },
    {
        'name'      : 'ResNet50V2',
        'img_size'  : 224,
        'optimiser' : 'adam',
        'pre_import': (
            'tensorflow.keras.applications.resnet_v2',
            'preprocess_input'),
    },
    {
        'name'      : 'InceptionV3',
        'img_size'  : 299,
        'optimiser' : 'adam',
        'pre_import': (
            'tensorflow.keras.applications.inception_v3',
            'preprocess_input'),
    },
    {
        'name'      : 'ConvNeXtTiny',
        'img_size'  : 224,
        'optimiser' : 'adamw',
        'pre_import': (
            'tensorflow.keras.applications.convnext',
            'preprocess_input'),
    },
]

def make_dirs():
    """Create all output directories."""
    for d in [MODELS_DIR, METRICS_DIR,
              FIGURES_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    for sub in FIGURE_SUBDIRS:
        os.makedirs(FIGURES_DIR + sub, exist_ok=True)
    print("✅ Output directories created")
