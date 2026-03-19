# =============================================================
# ensemble_learning.py — Stacked Ensemble with Meta-Learner
# =============================================================
# Implements two-level stacking:
#   Level 1: Extract softmax probabilities from 5 base models
#   Level 2: XGBoost meta-learner on 20-dim feature vectors
#
# Also handles:
#   - Meta-learner candidate screening (12 classifiers)
#   - 5-fold cross-validation selection
#   - Feature importance analysis
# =============================================================

import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    classification_report)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.ensemble        import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier)
from sklearn.neural_network  import MLPClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

from config import (
    CLASSES, NUM_CLASSES, SEED,
    MODELS_DIR, METRICS_DIR,
    LEVEL1_FEATURES, CV_FOLDS, XGBOOST_PARAMS,
    IMG_SIZE_DEFAULT, IMG_SIZE_INCEPTION, BATCH_SIZE
)
from base_learner_training import PREPROCESS_FNS, IMG_SIZES


# ── Level-1 Feature Extraction ────────────────────────────────

def extract_level1_features(
        model_name : str,
        df         : pd.DataFrame,
        split_name : str = 'unknown'
) -> np.ndarray:
    """
    Extract softmax probability vectors from one base model.

    Args:
        model_name : Name of the base model
        df         : DataFrame with filepath and label
        split_name : For logging ('train','val','test')

    Returns:
        Array of shape (N, 4) — softmax probabilities
    """
    img_size = IMG_SIZES[model_name]
    pre_fn   = PREPROCESS_FNS[model_name]

    model_path = (MODELS_DIR +
                  f'{model_name}_final.keras')
    model      = keras.models.load_model(model_path)

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

    probs = model.predict(gen, verbose=0)
    print(f"  [{model_name}] {split_name}: "
          f"{probs.shape}")

    del model
    import gc
    gc.collect()
    tf.keras.backend.clear_session()

    return probs


def extract_all_features(
        train_df : pd.DataFrame,
        val_df   : pd.DataFrame,
        test_df  : pd.DataFrame,
        save     : bool = True
) -> tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Extract Level-1 features from all 5 base models
    for train, val, and test splits.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    model_names = [
        'DenseNet201', 'EfficientNetB4',
        'ResNet50V2', 'InceptionV3', 'ConvNeXtTiny']

    train_probs, val_probs, test_probs = [], [], []

    for model_name in model_names:
        print(f"\nExtracting: {model_name}")
        train_probs.append(extract_level1_features(
            model_name, train_df, 'train'))
        val_probs.append(extract_level1_features(
            model_name, val_df, 'val'))
        test_probs.append(extract_level1_features(
            model_name, test_df, 'test'))

    # Concatenate to (N, 20)
    X_train = np.concatenate(train_probs, axis=1)
    X_val   = np.concatenate(val_probs,   axis=1)
    X_test  = np.concatenate(test_probs,  axis=1)

    # Labels
    y_train = np.array([CLASSES.index(l)
                         for l in train_df['label']])
    y_val   = np.array([CLASSES.index(l)
                         for l in val_df['label']])
    y_test  = np.array([CLASSES.index(l)
                         for l in test_df['label']])

    print(f"\nLevel-1 feature shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val  : {X_val.shape}")
    print(f"  X_test : {X_test.shape}")

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        np.save(MODELS_DIR+'ens_train_X.npy', X_train)
        np.save(MODELS_DIR+'ens_train_y.npy', y_train)
        np.save(MODELS_DIR+'ens_val_X.npy',   X_val)
        np.save(MODELS_DIR+'ens_val_y.npy',   y_val)
        np.save(MODELS_DIR+'ens_test_X.npy',  X_test)
        np.save(MODELS_DIR+'ens_test_y.npy',  y_test)
        print("\n✅ Level-1 features saved")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Meta-Learner Candidates ───────────────────────────────────

def get_candidate_classifiers() -> dict:
    """
    Return all 12 candidate meta-learner classifiers.

    Returns:
        Dict mapping name to sklearn-compatible classifier
    """
    return {
        'XGBoost'          : XGBClassifier(
            **XGBOOST_PARAMS),
        'Logistic_Regression': LogisticRegression(
            max_iter=1000, random_state=SEED),
        'SVM_Linear'       : SVC(
            kernel='linear', probability=True,
            random_state=SEED),
        'SVM_RBF'          : SVC(
            kernel='rbf', probability=True,
            random_state=SEED),
        'Random_Forest'    : RandomForestClassifier(
            n_estimators=200, random_state=SEED),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=SEED),
        'Extra_Trees'      : ExtraTreesClassifier(
            n_estimators=200, random_state=SEED),
        'MLP'              : MLPClassifier(
            hidden_layer_sizes=(256,128),
            max_iter=500, random_state=SEED),
        'KNN_5'            : KNeighborsClassifier(n_neighbors=5),
        'KNN_9'            : KNeighborsClassifier(n_neighbors=9),
        'Naive_Bayes'      : GaussianNB(),
        'LDA'              : LinearDiscriminantAnalysis(),
    }


# ── Meta-Learner Selection ────────────────────────────────────

def select_meta_learner(
        X_tv : np.ndarray,
        y_tv : np.ndarray,
        save : bool = True
) -> tuple[str, object, pd.DataFrame]:
    """
    Screen 12 candidate classifiers using 5-fold CV
    on combined train+val Level-1 features.

    Args:
        X_tv: Combined train+val features (N, 20)
        y_tv: Combined train+val labels (N,)
        save: Whether to save results CSV

    Returns:
        (best_name, best_clf, results_df)
    """
    candidates = get_candidate_classifiers()
    cv         = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True,
        random_state=SEED)

    rows = []
    print(f"\nMeta-learner screening ({CV_FOLDS}-fold CV)...")
    print("=" * 55)

    for name, clf in candidates.items():
        print(f"  {name:<22}...", end=' ')
        try:
            scores = cross_val_score(
                clf, X_tv, y_tv,
                cv      = cv,
                scoring = 'f1_macro',
                n_jobs  = -1)
            mean_f1 = scores.mean()
            std_f1  = scores.std()
            print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")
            rows.append({
                'Classifier': name,
                'CV_F1_Mean': round(mean_f1, 4),
                'CV_F1_Std' : round(std_f1,  4),
            })
        except Exception as e:
            print(f"FAILED: {str(e)[:40]}")

    df_cv = pd.DataFrame(rows).sort_values(
        'CV_F1_Mean', ascending=False)
    print(f"\nTop 5 classifiers:")
    print(df_cv.head(5).to_string(index=False))

    best_name = df_cv.iloc[0]['Classifier']
    best_clf  = candidates[best_name]
    print(f"\n✅ Selected: {best_name} "
          f"(F1={df_cv.iloc[0]['CV_F1_Mean']:.4f})")

    if save:
        os.makedirs(METRICS_DIR, exist_ok=True)
        df_cv.to_csv(
            METRICS_DIR+'classifier_comparison.csv',
            index=False)

    return best_name, best_clf, df_cv


# ── Final Ensemble Training ───────────────────────────────────

def train_ensemble(
        best_clf    : object,
        X_tv        : np.ndarray,
        y_tv        : np.ndarray,
        X_test      : np.ndarray,
        y_test      : np.ndarray,
        save        : bool = True
) -> tuple[object, dict]:
    """
    Train final meta-learner on train+val features
    and evaluate on held-out test set.

    Returns:
        (fitted_clf, results_dict)
    """
    print("\nTraining final ensemble on train+val...")
    best_clf.fit(X_tv, y_tv)

    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)

    # Metrics
    acc    = accuracy_score(y_test, y_pred)
    rep    = classification_report(
        y_test, y_pred,
        target_names=CLASSES,
        output_dict=True)
    auc    = roc_auc_score(
        label_binarize(y_test,
                       classes=range(NUM_CLASSES)),
        y_prob,
        multi_class='ovr',
        average='macro')

    results = {
        'accuracy'  : round(acc, 4),
        'macro_f1'  : round(rep['macro avg']['f1-score'], 4),
        'macro_pre' : round(rep['macro avg']['precision'], 4),
        'macro_rec' : round(rep['macro avg']['recall'], 4),
        'auc'       : round(auc, 4),
        'per_class' : {
            cls: {
                'precision': round(
                    rep[cls]['precision'], 4),
                'recall'   : round(
                    rep[cls]['recall'], 4),
                'f1'       : round(
                    rep[cls]['f1-score'], 4),
            }
            for cls in CLASSES
        }
    }

    print(f"\n{'='*55}")
    print(f"ENSEMBLE RESULTS")
    print(f"{'='*55}")
    print(f"  Accuracy : {results['accuracy']}")
    print(f"  F1-Macro : {results['macro_f1']}")
    print(f"  AUC-ROC  : {results['auc']}")
    print(f"\n  Per-class F1:")
    for cls in CLASSES:
        print(f"    {cls:<12}: "
              f"{results['per_class'][cls]['f1']}")

    if save:
        os.makedirs(MODELS_DIR,  exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

        joblib.dump(
            best_clf,
            MODELS_DIR+'ensemble_meta_learner.pkl')

        np.save(MODELS_DIR+'ens_pred.npy', y_pred)
        np.save(MODELS_DIR+'ens_prob.npy', y_prob)

        print(f"\n✅ Ensemble saved: "
              f"ensemble_meta_learner.pkl")

    return best_clf, results


# ── Feature Importance ────────────────────────────────────────

def get_feature_importance(
        clf  : object,
        save : bool = True
) -> pd.DataFrame:
    """
    Extract XGBoost feature importance for meta-learner.

    Returns:
        DataFrame with feature names and importance scores
    """
    model_names = [
        'DenseNet201', 'EfficientNetB4',
        'ResNet50V2', 'InceptionV3', 'ConvNeXtTiny']

    feature_names = []
    for m in model_names:
        for cls in CLASSES:
            feature_names.append(f'{m[:6]}_{cls[:3]}')

    importance = clf.feature_importances_
    df_imp     = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': importance,
    }).sort_values('Importance', ascending=False)

    if save:
        df_imp.to_csv(
            METRICS_DIR+'feature_importance.csv',
            index=False)

    return df_imp


if __name__ == '__main__':
    print("Ensemble learning module loaded.")
    print(f"Level-1 feature dim: {LEVEL1_FEATURES}")
    print(f"Meta-learner: XGBoost")
    print(f"Candidates: 12 classifiers")
    print(f"CV folds: {CV_FOLDS}")
