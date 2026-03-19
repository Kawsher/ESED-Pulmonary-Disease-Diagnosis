# =============================================================
# dataset_preparation.py — Dataset Construction and Splitting
# =============================================================
# Handles:
#   - Multi-source image collection
#   - MD5 deduplication
#   - Stratified train/val/test splitting
#   - Class weight computation
#   - Data augmentation configuration
# =============================================================

import os
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    CLASSES, NUM_CLASSES, SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SPLITS_DIR, METRICS_DIR, BATCH_SIZE
)


# ── MD5 Hash Utilities ────────────────────────────────────────

def get_image_hash(path: str) -> str | None:
    """Compute MD5 hash of an image file."""
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def deduplicate_dataset(manifest_df: pd.DataFrame
                         ) -> pd.DataFrame:
    """
    Remove duplicate images using MD5 hashing.

    Args:
        manifest_df: DataFrame with 'filepath' and 'label'

    Returns:
        Deduplicated DataFrame with unique images only
    """
    print("Computing MD5 hashes...")
    manifest_df = manifest_df.copy()
    manifest_df['hash'] = manifest_df['filepath'].apply(
        get_image_hash)

    n_before = len(manifest_df)
    manifest_df = manifest_df.drop_duplicates(
        subset='hash', keep='first')
    n_after  = len(manifest_df)

    print(f"  Images before : {n_before:,}")
    print(f"  Images after  : {n_after:,}")
    print(f"  Duplicates removed: {n_before - n_after}")
    print("\n  Class distribution after deduplication:")
    for cls in CLASSES:
        n = (manifest_df['label'] == cls).sum()
        print(f"    {cls:<12}: {n:,}")

    return manifest_df.drop(
        columns=['hash']).reset_index(drop=True)


def check_cross_split_leakage(
        train_df: pd.DataFrame,
        val_df  : pd.DataFrame,
        test_df : pd.DataFrame,
        manifest_df: pd.DataFrame
) -> int:
    """
    Verify zero cross-split image overlap using MD5 hashes.

    Returns:
        Number of cross-split duplicate pairs found
    """
    print("Verifying cross-split integrity...")
    manifest_df = manifest_df.copy()
    manifest_df['hash'] = manifest_df['filepath'].apply(
        get_image_hash)

    hash_map   = manifest_df.set_index(
        'filepath')['hash'].to_dict()

    train_hashes = set(
        train_df['filepath'].map(hash_map).dropna())
    val_hashes   = set(
        val_df['filepath'].map(hash_map).dropna())
    test_hashes  = set(
        test_df['filepath'].map(hash_map).dropna())

    tv = len(train_hashes & val_hashes)
    tt = len(train_hashes & test_hashes)
    vt = len(val_hashes   & test_hashes)

    print(f"  Train-Val overlap  : {tv}")
    print(f"  Train-Test overlap : {tt}")
    print(f"  Val-Test overlap   : {vt}")

    if tv == 0 and tt == 0 and vt == 0:
        print("  ✅ Zero cross-split leakage confirmed")
    else:
        print("  ❌ Cross-split leakage detected — re-split")

    return tv + tt + vt


# ── Stratified Splitting ──────────────────────────────────────

def create_splits(manifest_df: pd.DataFrame,
                  save: bool = True
                  ) -> tuple[pd.DataFrame,
                              pd.DataFrame,
                              pd.DataFrame]:
    """
    Create stratified train/val/test splits.

    Args:
        manifest_df: Deduplicated DataFrame
        save: Whether to save CSVs to SPLITS_DIR

    Returns:
        (train_df, val_df, test_df)
    """
    all_files  = manifest_df['filepath'].values
    all_labels = manifest_df['label'].values

    # Split train vs temp (val+test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=SEED)
    train_idx, temp_idx = next(
        sss1.split(all_files, all_labels))

    # Split temp into val and test
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED)
    val_rel, test_rel = next(
        sss2.split(all_files[temp_idx],
                   all_labels[temp_idx]))
    val_idx  = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    train_df = pd.DataFrame({
        'filepath': all_files[train_idx],
        'label'   : all_labels[train_idx]
    })
    val_df = pd.DataFrame({
        'filepath': all_files[val_idx],
        'label'   : all_labels[val_idx]
    })
    test_df = pd.DataFrame({
        'filepath': all_files[test_idx],
        'label'   : all_labels[test_idx]
    })

    print(f"\nSplit sizes:")
    for name, df in [('Train', train_df),
                      ('Val',   val_df),
                      ('Test',  test_df)]:
        print(f"  {name}: {len(df):,}")
        for cls in CLASSES:
            n = (df['label'] == cls).sum()
            print(f"    {cls:<12}: {n:,}")

    if save:
        os.makedirs(SPLITS_DIR, exist_ok=True)
        train_df.to_csv(
            SPLITS_DIR+'train_split.csv', index=False)
        val_df.to_csv(
            SPLITS_DIR+'val_split.csv',   index=False)
        test_df.to_csv(
            SPLITS_DIR+'test_split.csv',  index=False)
        print(f"\n✅ Splits saved to {SPLITS_DIR}")

    return train_df, val_df, test_df


# ── Class Weights ─────────────────────────────────────────────

def compute_weights(train_df: pd.DataFrame,
                    save: bool = True
                    ) -> dict[int, float]:
    """
    Compute balanced class weights for training.

    Returns:
        Dict mapping class index to weight
    """
    label_ints = [CLASSES.index(l)
                  for l in train_df['label']]
    weights    = compute_class_weight(
        'balanced',
        classes=np.arange(NUM_CLASSES),
        y=np.array(label_ints))
    class_weights = {i: float(w)
                     for i, w in enumerate(weights)}

    print(f"\nClass weights: {class_weights}")

    if save:
        os.makedirs(SPLITS_DIR, exist_ok=True)
        with open(SPLITS_DIR+'class_weights.json',
                  'w') as f:
            json.dump(class_weights, f, indent=2)
        print("✅ Class weights saved")

    return class_weights


# ── Data Generators ───────────────────────────────────────────

def get_train_generator(train_df : pd.DataFrame,
                         pre_fn   ,
                         img_size : int,
                         augment  : bool = True):
    """
    Create training data generator with augmentation.

    Args:
        train_df : DataFrame with filepath and label
        pre_fn   : Model-specific preprocessing function
        img_size : Target image size
        augment  : Whether to apply augmentation

    Returns:
        Keras ImageDataGenerator flow
    """
    aug_params = dict(
        preprocessing_function=pre_fn,
        rotation_range      = 10,
        width_shift_range   = 0.05,
        height_shift_range  = 0.05,
        zoom_range          = 0.05,
        horizontal_flip     = True,
    ) if augment else dict(
        preprocessing_function=pre_fn)

    gen = ImageDataGenerator(**aug_params)
    return gen.flow_from_dataframe(
        train_df,
        x_col      = 'filepath',
        y_col      = 'label',
        target_size= (img_size, img_size),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical',
        classes    = CLASSES,
        shuffle    = True,
        seed       = SEED,
    )


def get_val_generator(val_df  : pd.DataFrame,
                       pre_fn  ,
                       img_size: int):
    """Create validation/test data generator (no augment)."""
    gen = ImageDataGenerator(
        preprocessing_function=pre_fn)
    return gen.flow_from_dataframe(
        val_df,
        x_col      = 'filepath',
        y_col      = 'label',
        target_size= (img_size, img_size),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical',
        classes    = CLASSES,
        shuffle    = False,
    )


# ── Sensitivity Analysis ──────────────────────────────────────

def sensitivity_analysis(
        manifest_df : pd.DataFrame,
        train_df    : pd.DataFrame,
        test_df     : pd.DataFrame,
        cross_dupes : pd.DataFrame
) -> dict:
    """
    Quantify impact of duplicate contamination
    by evaluating on clean vs full test set.
    Requires ensemble to already be trained.

    Args:
        manifest_df  : Full dataset manifest
        train_df     : Training split
        test_df      : Test split
        cross_dupes  : DataFrame of cross-split duplicates

    Returns:
        Dict with original and clean metrics
    """
    contaminated_hashes = set()
    for _, row in cross_dupes.iterrows():
        h1 = get_image_hash(row['file1'])
        h2 = get_image_hash(row['file2'])
        if h1: contaminated_hashes.add(h1)
        if h2: contaminated_hashes.add(h2)

    test_hashes       = test_df['filepath'].apply(
        get_image_hash)
    contaminated_mask = test_hashes.isin(
        contaminated_hashes)
    clean_mask        = ~contaminated_mask

    print(f"Original test size  : {len(test_df):,}")
    print(f"Contaminated images : {contaminated_mask.sum()}")
    print(f"Clean test images   : {clean_mask.sum():,}")

    return {
        'original_test_n'    : int(len(test_df)),
        'clean_test_n'       : int(clean_mask.sum()),
        'contaminated_removed': int(contaminated_mask.sum()),
        'clean_indices'      : np.where(
            clean_mask.values)[0].tolist(),
    }


if __name__ == '__main__':
    print("Dataset preparation module loaded.")
    print(f"Classes: {CLASSES}")
    print(f"Split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
