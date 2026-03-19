# =============================================================
# external_validation.py — Multi-Institutional Generalisation
# =============================================================
# Tests ESED on two external datasets:
#   1. NIH ChestX-ray14 — USA adult patients
#      Task: Pneumonia vs Normal (2-class)
#      Finding: Severe domain shift (F1=0.022)
#
#   2. Epic Chittagong — Bangladesh adult patients
#      Task: Pneumonia vs Normal (2-class)
#      Finding: Partial generalisation (F1=0.838)
#
# Domain shift analysis and clinical interpretation included.
# =============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    accuracy_score)

from config import (
    CLASSES, NUM_CLASSES, SEED,
    MODELS_DIR, METRICS_DIR, FIGURES_DIR,
    BATCH_SIZE
)
from base_learner_training import PREPROCESS_FNS, IMG_SIZES


# ── Constants ─────────────────────────────────────────────────
PNEU_IDX   = CLASSES.index('Pneumonia')
NORMAL_IDX = CLASSES.index('Normal')

NIH_BASE   = (
    '/kaggle/input/datasets/organizations/'
    'nih-chest-xrays/data/')
EPIC_BASE  = (
    '/kaggle/input/datasets/mdkawshermahbub/'
    'epic-chittagong-xray/')

MODELS_INPUT = (
    '/kaggle/input/datasets/mdkawshermahbub/'
    'pulmonary-model-outputs/models/')


# ── Feature Extraction ────────────────────────────────────────

def extract_external_features(
        df          : pd.DataFrame,
        dataset_name: str = 'external'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Level-1 features from all 5 base models
    for an external dataset.

    Args:
        df          : DataFrame with 'filepath' and 'label'
        dataset_name: For logging

    Returns:
        (X_ext, y_ext) — features and labels
    """
    model_names = [
        'DenseNet201', 'EfficientNetB4',
        'ResNet50V2', 'InceptionV3', 'ConvNeXtTiny']

    all_probs = []
    labels    = None

    print(f"\nExtracting features: {dataset_name}")
    print(f"Images: {len(df):,}")

    for model_name in model_names:
        print(f"  {model_name}...", end=' ')
        img_size = IMG_SIZES[model_name]
        pre_fn   = PREPROCESS_FNS[model_name]

        model    = keras.models.load_model(
            MODELS_INPUT + f'{model_name}_final.keras')

        gen = ImageDataGenerator(
            preprocessing_function=pre_fn
        ).flow_from_dataframe(
            df,
            x_col      = 'filepath',
            y_col      = 'label',
            target_size= (img_size, img_size),
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            classes    = CLASSES,
            shuffle    = False,
        )

        probs  = model.predict(gen, verbose=0)
        labels = gen.classes
        all_probs.append(probs)
        print(f"✅ {probs.shape}")

        del model
        import gc; gc.collect()
        tf.keras.backend.clear_session()

    X_ext = np.concatenate(all_probs, axis=1)
    y_ext = np.array(labels)
    return X_ext, y_ext


# ── 2-Class Evaluation ────────────────────────────────────────

def evaluate_2class(
        final_clf : object,
        X_ext     : np.ndarray,
        y_ext     : np.ndarray,
        dataset   : str,
        institution: str,
        country   : str
) -> dict:
    """
    Evaluate ensemble on Pneumonia vs Normal only.

    Returns:
        Dict with all evaluation metrics
    """
    y_pred_4 = final_clf.predict(X_ext)
    y_prob_4 = final_clf.predict_proba(X_ext)

    # Filter to Pneumonia + Normal
    mask     = (y_ext == PNEU_IDX) | \
               (y_ext == NORMAL_IDX)
    y_true_2 = (y_ext[mask] == PNEU_IDX).astype(int)
    y_pred_2 = (y_pred_4[mask] == PNEU_IDX).astype(int)
    y_prob_2 = y_prob_4[mask, PNEU_IDX]

    acc  = accuracy_score(y_true_2, y_pred_2)
    f1   = f1_score(y_true_2, y_pred_2,
                    zero_division=0)
    auc  = roc_auc_score(y_true_2, y_prob_2)
    rep  = classification_report(
        y_true_2, y_pred_2,
        target_names=['Normal', 'Pneumonia'],
        output_dict=True)

    print(f"\n{'='*55}")
    print(f"EXTERNAL VALIDATION: {dataset}")
    print(f"{'='*55}")
    print(f"  Institution : {institution}")
    print(f"  Country     : {country}")
    print(f"  Test size   : {mask.sum()}")
    print(f"  Pneumonia   : {y_true_2.sum()}")
    print(f"  Normal      : {(y_true_2==0).sum()}")
    print(f"\n  Accuracy    : {acc:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"\n  Per-class:")
    for cls2 in ['Normal', 'Pneumonia']:
        r = rep[cls2]
        print(f"    {cls2:<12}: "
              f"P={r['precision']:.4f} "
              f"R={r['recall']:.4f} "
              f"F1={r['f1-score']:.4f}")

    # Model-level probability analysis
    print(f"\n  Per-model Pneumonia probabilities:")
    model_names = [
        'DenseNet201','EfficientNetB4',
        'ResNet50V2','InceptionV3','ConvNeXtTiny']
    for mi, mname in enumerate(model_names):
        p_mean = X_ext[:, mi*4+PNEU_IDX].mean()
        n_mean = X_ext[:, mi*4+NORMAL_IDX].mean()
        print(f"    {mname:<16}: "
              f"Pneumonia={p_mean:.4f} | "
              f"Normal={n_mean:.4f}")

    return {
        'dataset'    : dataset,
        'institution': institution,
        'country'    : country,
        'test_size'  : int(mask.sum()),
        'pneumonia_n': int(y_true_2.sum()),
        'normal_n'   : int((y_true_2==0).sum()),
        'accuracy'   : round(acc, 4),
        'f1'         : round(f1,  4),
        'auc'        : round(auc, 4),
        'precision'  : round(
            rep['Pneumonia']['precision'], 4),
        'recall'     : round(
            rep['Pneumonia']['recall'], 4),
        'y_true_2'   : y_true_2,
        'y_pred_2'   : y_pred_2,
        'y_prob_2'   : y_prob_2,
    }


# ── NIH ChestX-ray14 Validation ──────────────────────────────

def build_nih_test_set(n_per_class: int = 500
                        ) -> pd.DataFrame:
    """
    Build balanced NIH Pneumonia/Normal test set
    from official test_list.txt.

    Args:
        n_per_class: Max images per class

    Returns:
        DataFrame with filepath and label
    """
    print("\nBuilding NIH test set...")

    labels_df = pd.read_csv(
        NIH_BASE + 'Data_Entry_2017.csv')

    with open(NIH_BASE + 'test_list.txt') as f:
        test_images = set(
            f.read().strip().split('\n'))

    # Build image path lookup
    img_lookup = {}
    for i in range(1, 13):
        folder = NIH_BASE + f'images_{i:03d}/images/'
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                img_lookup[fname] = folder + fname

    # Pure-label Pneumonia and No Finding
    pneu_df = labels_df[
        (labels_df['Finding Labels'] == 'Pneumonia') &
        (labels_df['Image Index'].isin(test_images))
    ].copy()
    norm_df = labels_df[
        (labels_df['Finding Labels'] == 'No Finding') &
        (labels_df['Image Index'].isin(test_images))
    ].copy()

    pneu_df['filepath'] = pneu_df[
        'Image Index'].map(img_lookup)
    norm_df['filepath'] = norm_df[
        'Image Index'].map(img_lookup)
    pneu_df = pneu_df.dropna(subset=['filepath'])
    norm_df = norm_df.dropna(subset=['filepath'])

    N = min(n_per_class,
            len(pneu_df), len(norm_df))
    df = pd.concat([
        pneu_df.sample(N, random_state=SEED)\
               .assign(label='Pneumonia'),
        norm_df.sample(N, random_state=SEED)\
               .assign(label='Normal')
    ]).reset_index(drop=True)

    print(f"  Pneumonia: {(df['label']=='Pneumonia').sum()}")
    print(f"  Normal   : {(df['label']=='Normal').sum()}")
    return df


def validate_nih(final_clf: object,
                  save     : bool = True) -> dict:
    """Run NIH ChestX-ray14 external validation."""
    df        = build_nih_test_set()
    X, y      = extract_external_features(df, 'NIH')
    result    = evaluate_2class(
        final_clf, X, y,
        dataset     = 'NIH ChestX-ray14',
        institution = 'NIH Clinical Center',
        country     = 'USA')

    if save:
        # Remove arrays before JSON serialisation
        save_result = {k: v for k, v in result.items()
                       if not isinstance(v, np.ndarray)}
        with open(METRICS_DIR+'external_validation.json',
                  'w') as f:
            json.dump(save_result, f, indent=2)
        np.save(MODELS_DIR+'ens_nih_X.npy', X)
        np.save(MODELS_DIR+'ens_nih_y.npy', y)

    return result


# ── Epic Chittagong Validation ────────────────────────────────

def build_epic_test_set() -> pd.DataFrame:
    """
    Build Epic Chittagong test set from folder structure.

    Returns:
        DataFrame with filepath and label
    """
    print("\nBuilding Epic Chittagong test set...")
    rows = []
    for cls, folder in [
        ('Pneumonia', EPIC_BASE+'Testing/pneumonia/'),
        ('Normal',    EPIC_BASE+'Testing/normal/'),
    ]:
        if not os.path.exists(folder):
            print(f"  ❌ Not found: {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(
                    ('.png','.jpg','.jpeg')):
                rows.append({
                    'filepath': os.path.join(
                        folder, fname),
                    'label'   : cls,
                })

    df = pd.DataFrame(rows)
    print(f"  Pneumonia: "
          f"{(df['label']=='Pneumonia').sum()}")
    print(f"  Normal   : "
          f"{(df['label']=='Normal').sum()}")
    return df


def validate_epic_chittagong(
        final_clf: object,
        save     : bool = True
) -> dict:
    """Run Epic Chittagong external validation."""
    df     = build_epic_test_set()
    X, y   = extract_external_features(
        df, 'Epic Chittagong')
    result = evaluate_2class(
        final_clf, X, y,
        dataset     = 'Epic Chittagong Chest X-ray',
        institution = 'Epic Chittagong Hospital',
        country     = 'Bangladesh')

    if save:
        save_result = {k: v for k, v in result.items()
                       if not isinstance(v, np.ndarray)}
        with open(
                METRICS_DIR+
                'external_validation_chittagong.json',
                'w') as f:
            json.dump(save_result, f, indent=2)
        np.save(MODELS_DIR+'ens_epic_X.npy', X)
        np.save(MODELS_DIR+'ens_epic_y.npy', y)

    return result


# ── Three-Way Comparison ──────────────────────────────────────

def three_way_comparison(
        internal_f1  : float,
        internal_auc : float,
        nih_result   : dict,
        epic_result  : dict,
        save         : bool = True
) -> pd.DataFrame:
    """
    Generate three-way internal vs external summary table.

    Returns:
        Comparison DataFrame
    """
    rows = [
        {
            'Source'     : 'Internal (multi-source)',
            'Institution': 'Multi-institution',
            'Country'    : 'Multi',
            'F1'         : internal_f1,
            'AUC'        : internal_auc,
            'F1_Drop'    : 0.0,
        },
        {
            'Source'     : 'Epic Chittagong',
            'Institution': 'Epic Chittagong Hospital',
            'Country'    : 'Bangladesh',
            'F1'         : epic_result['f1'],
            'AUC'        : epic_result['auc'],
            'F1_Drop'    : round(
                epic_result['f1']-internal_f1, 4),
        },
        {
            'Source'     : 'NIH ChestX-ray14',
            'Institution': 'NIH Clinical Center',
            'Country'    : 'USA',
            'F1'         : nih_result['f1'],
            'AUC'        : nih_result['auc'],
            'F1_Drop'    : round(
                nih_result['f1']-internal_f1, 4),
        },
    ]

    df = pd.DataFrame(rows)

    print(f"\n{'='*65}")
    print(f"THREE-WAY COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Source':<28} {'F1':>8} "
          f"{'AUC':>8} {'F1 Drop':>10}")
    print(f"  {'-'*56}")
    for _, row in df.iterrows():
        print(f"  {row['Source']:<28} "
              f"{row['F1']:>8.4f} "
              f"{row['AUC']:>8.4f} "
              f"{row['F1_Drop']:>+9.4f}")

    if save:
        df.to_csv(
            METRICS_DIR+'external_validation_summary.csv',
            index=False)

    return df


if __name__ == '__main__':
    print("External validation module loaded.")
    print(f"NIH base  : {NIH_BASE}")
    print(f"Epic base : {EPIC_BASE}")
    print(f"Task      : Pneumonia vs Normal (2-class)")
