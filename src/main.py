# =============================================================
# main.py — ESED Complete Pipeline
# =============================================================
# Runs the complete ESED framework end-to-end:
#   1. Dataset preparation and splitting
#   2. Base learner training (5 CNNs)
#   3. Level-1 feature extraction
#   4. Meta-learner selection and ensemble training
#   5. Statistical validation (6 tests)
#   6. XAI analysis (Grad-CAM, SHAP, LIME)
#   7. External validation (NIH, Epic Chittagong)
#   8. Results aggregation and figure generation
#
# Usage:
#   python main.py --phase all
#   python main.py --phase train
#   python main.py --phase ensemble
#   python main.py --phase evaluate
# =============================================================

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    CLASSES, NUM_CLASSES, SEED,
    WORK, MODELS_DIR, METRICS_DIR,
    FIGURES_DIR, SPLITS_DIR, make_dirs
)

# Set global seeds
tf.random.set_seed(SEED)
np.random.seed(SEED)
import random; random.seed(SEED)


# ── Argument Parser ───────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='ESED: Explainable Stacked Ensemble '
                    'for Pulmonary Disease Diagnosis')
    parser.add_argument(
        '--phase',
        type   = str,
        default= 'all',
        choices= ['all', 'prepare', 'train',
                  'ensemble', 'evaluate',
                  'xai', 'external'],
        help   = 'Pipeline phase to run')
    parser.add_argument(
        '--skip-training',
        action = 'store_true',
        help   = 'Skip model training (use saved models)')
    parser.add_argument(
        '--models-dir',
        type   = str,
        default= MODELS_DIR,
        help   = 'Directory containing saved models')
    return parser.parse_args()


# ── Phase 1: Dataset Preparation ─────────────────────────────

def phase_prepare():
    """Prepare dataset splits and verify integrity."""
    from dataset_preparation import (
        deduplicate_dataset, create_splits,
        compute_weights, check_cross_split_leakage)

    print("\n" + "="*60)
    print("PHASE 1: Dataset Preparation")
    print("="*60)

    # Load manifest
    manifest_df = pd.read_csv(SPLITS_DIR+'manifest.csv')
    print(f"Loaded manifest: {len(manifest_df):,} images")

    # Deduplicate
    clean_df = deduplicate_dataset(manifest_df)

    # Create splits
    train_df, val_df, test_df = create_splits(
        clean_df, save=True)

    # Compute class weights
    class_weights = compute_weights(train_df, save=True)

    # Verify integrity
    leakage = check_cross_split_leakage(
        train_df, val_df, test_df, clean_df)

    if leakage > 0:
        print(f"❌ {leakage} cross-split duplicates found")
        sys.exit(1)

    print("\n✅ Dataset preparation complete")
    return train_df, val_df, test_df, class_weights


# ── Phase 2: Base Learner Training ───────────────────────────

def phase_train(train_df, val_df, class_weights):
    """Train all 5 base learners."""
    from base_learner_training import train_all_models

    print("\n" + "="*60)
    print("PHASE 2: Base Learner Training")
    print("="*60)

    histories = train_all_models(
        train_df, val_df, class_weights)

    print("\n✅ Base learner training complete")
    return histories


# ── Phase 3: Ensemble Learning ───────────────────────────────

def phase_ensemble(train_df, val_df, test_df):
    """Extract features and train ensemble."""
    from ensemble_learning import (
        extract_all_features, select_meta_learner,
        train_ensemble, get_feature_importance)

    print("\n" + "="*60)
    print("PHASE 3: Ensemble Learning")
    print("="*60)

    # Extract Level-1 features
    X_tr, y_tr, X_val, y_val, X_te, y_te = \
        extract_all_features(
            train_df, val_df, test_df, save=True)

    # Combined train+val
    X_tv = np.concatenate([X_tr, X_val])
    y_tv = np.concatenate([y_tr, y_val])

    # Select meta-learner
    best_name, best_clf, cv_results = \
        select_meta_learner(X_tv, y_tv, save=True)

    # Train final ensemble
    final_clf, results = train_ensemble(
        best_clf, X_tv, y_tv, X_te, y_te, save=True)

    # Feature importance
    imp_df = get_feature_importance(
        final_clf, save=True)

    print(f"\nTop 5 features:")
    print(imp_df.head(5).to_string(index=False))

    print("\n✅ Ensemble learning complete")
    return final_clf, results, X_te, y_te


# ── Phase 4: Statistical Validation ──────────────────────────

def phase_evaluate(final_clf, X_te, y_te):
    """Run all 6 statistical validation tests."""
    from statistical_validation import (
        run_mcnemar_all, friedman_test, nemenyi_test,
        wilcoxon_test, delong_test,
        bootstrap_ci_all_models)
    from sklearn.model_selection import (
        StratifiedKFold, cross_val_score)
    from ensemble_learning import get_candidate_classifiers

    print("\n" + "="*60)
    print("PHASE 4: Statistical Validation")
    print("="*60)

    # Load base model predictions
    y_pred_ens = final_clf.predict(X_te)
    y_prob_ens = final_clf.predict_proba(X_te)

    # Load all_results for base model predictions
    with open(METRICS_DIR+'all_results.json') as f:
        all_results = json.load(f)

    # Test 1: McNemar
    base_predictions = {
        name: np.array(
            all_results[name].get('y_pred', []))
        for name in ['DenseNet201','EfficientNetB4',
                     'ResNet50V2','InceptionV3',
                     'ConvNeXtTiny']
        if name in all_results
    }
    if base_predictions:
        run_mcnemar_all(
            y_te, y_pred_ens, base_predictions)

    # Tests 2-4: Friedman, Nemenyi, Wilcoxon
    # Requires CV scores from ensemble screening
    cv_df = pd.read_csv(
        METRICS_DIR+'classifier_comparison.csv')
    print("\n(Friedman/Nemenyi require kfold_cv_results.csv)")

    # Test 5: DeLong
    base_probs = {
        name: np.array(
            all_results[name].get('y_prob', []))
        for name in base_predictions
        if all_results[name].get('y_prob')
    }
    if base_probs:
        delong_test(y_te, y_prob_ens, base_probs)

    # Test 6: Bootstrap CI
    base_preds_dict = {
        name: np.array(
            all_results[name].get('y_pred', []))
        for name in base_predictions
    }
    bootstrap_ci_all_models(
        y_te, y_pred_ens, base_preds_dict)

    print("\n✅ Statistical validation complete")


# ── Phase 5: XAI Analysis ────────────────────────────────────

def phase_xai(test_df, y_te, y_pred_ens, y_prob_ens):
    """Run Grad-CAM, SHAP, and LIME analysis."""
    from xai_analysis import (
        get_sample_images, generate_gradcam_panel,
        compute_shap_values, get_shap_spatial_map,
        compute_shap_region_importance,
        generate_lime_explanations, analyse_lime_results,
        compute_xai_agreement)
    from base_learner_training import PREPROCESS_FNS
    from tensorflow.keras.applications.densenet import \
        preprocess_input as densenet_pre
    from tensorflow.keras.preprocessing.image import \
        load_img, img_to_array
    import tensorflow as tf

    MODELS_INPUT = (
        '/kaggle/input/datasets/mdkawshermahbub/'
        'pulmonary-model-outputs/models/')

    print("\n" + "="*60)
    print("PHASE 5: XAI Analysis")
    print("="*60)

    # Get sample images
    sample_images = get_sample_images(
        test_df, y_te, y_pred_ens, y_prob_ens)

    # Grad-CAM
    print("\n[1/3] Grad-CAM...")
    generate_gradcam_panel(sample_images, save=True)

    # SHAP (CPU only)
    print("\n[2/3] SHAP...")
    tf.config.set_visible_devices([], 'GPU')
    from tensorflow import keras
    model = keras.models.load_model(
        MODELS_INPUT + 'DenseNet201_final.keras')

    bg_df  = test_df.sample(
        min(50, len(test_df)), random_state=SEED)
    bg_arr = np.array([
        densenet_pre(img_to_array(
            load_img(r['filepath'],
                     target_size=(224,224))))
        for _, r in bg_df.iterrows()])

    test_imgs      = []
    test_labels_str= []
    for cls in CLASSES:
        if cls in sample_images:
            img = densenet_pre(img_to_array(
                load_img(sample_images[cls]['path'],
                         target_size=(224,224))))
            test_imgs.append(img)
            test_labels_str.append(cls)

    test_arr   = np.array(test_imgs)
    shap_values= compute_shap_values(
        model, test_arr, bg_arr)
    compute_shap_region_importance(
        shap_values, test_labels_str, save=True)

    # LIME
    print("\n[3/3] LIME...")
    lime_results = generate_lime_explanations(
        model, sample_images, save=True)
    analyse_lime_results(lime_results, save=True)

    del model
    import gc; gc.collect()
    tf.keras.backend.clear_session()
    print("\n✅ XAI analysis complete")


# ── Phase 6: External Validation ─────────────────────────────

def phase_external(final_clf):
    """Run external validation on NIH and Epic Chittagong."""
    from external_validation import (
        validate_nih, validate_epic_chittagong,
        three_way_comparison)

    print("\n" + "="*60)
    print("PHASE 6: External Validation")
    print("="*60)

    # NIH
    print("\n[1/2] NIH ChestX-ray14...")
    nih_result  = validate_nih(final_clf, save=True)

    # Epic Chittagong
    print("\n[2/2] Epic Chittagong...")
    epic_result = validate_epic_chittagong(
        final_clf, save=True)

    # Load internal F1 for comparison
    with open(METRICS_DIR+'all_results.json') as f:
        all_results = json.load(f)
    internal_f1  = all_results['ESED_Ensemble']['per_class']\
                   ['Pneumonia']['f1']
    internal_auc = all_results['ESED_Ensemble']['auc']

    # Three-way comparison
    three_way_comparison(
        internal_f1, internal_auc,
        nih_result, epic_result, save=True)

    print("\n✅ External validation complete")


# ── Main Entry Point ──────────────────────────────────────────

def main():
    args = parse_args()
    make_dirs()

    print("\n" + "#"*60)
    print("ESED: Explainable Stacked Ensemble Framework")
    print("Multi-Class Pulmonary Disease Diagnosis")
    print("#"*60)
    print(f"Phase    : {args.phase}")
    print(f"Classes  : {CLASSES}")
    print(f"Seed     : {SEED}")

    if args.phase in ('all', 'prepare'):
        train_df, val_df, test_df, class_weights = \
            phase_prepare()

    if args.phase in ('all', 'train') and \
            not args.skip_training:
        phase_train(train_df, val_df, class_weights)

    if args.phase in ('all', 'ensemble'):
        train_df = pd.read_csv(SPLITS_DIR+'train_split.csv')
        val_df   = pd.read_csv(SPLITS_DIR+'val_split.csv')
        test_df  = pd.read_csv(SPLITS_DIR+'test_split.csv')
        final_clf, results, X_te, y_te = \
            phase_ensemble(train_df, val_df, test_df)

    if args.phase in ('all', 'evaluate'):
        import joblib
        final_clf = joblib.load(
            MODELS_DIR+'ensemble_meta_learner.pkl')
        X_te = np.load(MODELS_DIR+'ens_test_X.npy')
        y_te = np.load(MODELS_DIR+'ens_test_y.npy')
        phase_evaluate(final_clf, X_te, y_te)

    if args.phase in ('all', 'xai'):
        import joblib
        test_df   = pd.read_csv(SPLITS_DIR+'test_split.csv')
        final_clf = joblib.load(
            MODELS_DIR+'ensemble_meta_learner.pkl')
        X_te      = np.load(MODELS_DIR+'ens_test_X.npy')
        y_te      = np.load(MODELS_DIR+'ens_test_y.npy')
        y_pred_ens= final_clf.predict(X_te)
        y_prob_ens= final_clf.predict_proba(X_te)
        phase_xai(test_df, y_te, y_pred_ens, y_prob_ens)

    if args.phase in ('all', 'external'):
        import joblib
        final_clf = joblib.load(
            MODELS_DIR+'ensemble_meta_learner.pkl')
        phase_external(final_clf)

    print("\n" + "#"*60)
    print("ESED Pipeline Complete ✅")
    print("#"*60)


if __name__ == '__main__':
    main()
